# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Modified by Shaoshuai Shi 
# All Rights Reserved


"""
Reference: https://github.com/dvlab-research/DeepVision3D/blob/master/EQNet/eqnet/transformer/multi_head_attention.py
"""

from typing import Optional

import torch.nn.functional as F
from torch import nn, Tensor

from .multi_head_attention import MultiheadAttention
from .multi_head_attention_local import MultiheadAttentionLocal

from fmoe.layers import _fmoe_general_global_forward, fmoe_faster_schedule

# options for all models:
viz = False
decoder2x = False # (for baseline model) making the decoder twice the size
upsample_hard = False

# options for MOE models only:
moe = False
NUMEXPERTS = 32
TOPK = 2
SHARED = 0

first_agent = False # only use the first agent's tokens for routing
curr_timestep = False # only use the current timestep's tokens for routing
final_timestep = False # only use the final (in the decoder) timestep's tokens for routing
cls_token = False # add a CLS token and only use it for routing
T_OBS = 21 # don't hardcode, change later TODO
T_FUT = 60 # don't hardcode, change later
NUM_PREDS = 6 # don't hardcode, change later
NUM_AGENTS = 65 # don't hardcode, change later

two_layer_router = False 

harmonic = False # NUMEXPERTS needs to be divisible by 8  for this to work
# ratios = [0, 1.0/8, 1.0/4, 1.0/2, 3.0/4, 1.0, 1.5, 2.0] # for shared
ratios = [1.0/8, 1.0/4, 1.0/2, 3.0/4, 1.0, 1.5, 2.0, 2.5] # for not shared

diff_init = False # does not seem to help
concatdim = False #DON'T USE! KILLS MEMORY

moe_types = ['social', 'temporal', 'decoder'] # social, temporal, decoder

if moe:
    print("MoE is enabled. NUMEXPERTS:", NUMEXPERTS, "TOPK:", TOPK, "SHARED:", SHARED, "harmonic:", harmonic, "first_agent:", first_agent, "curr_timestep:", curr_timestep, "final_timestep:", final_timestep, "cls_token:", cls_token)
    print('MOE for Transformer types: {}'.format(moe_types))
    import tree
    import torch
    from fmoe.gates import NaiveGate
    from fmoe.transformer import FMoE, _Expert, FMoETransformerMLP

    class _ExpertDiffInit(_Expert):
        def __init__(self, *args, expert_idx=0, num_experts=NUMEXPERTS, **kwargs):
            super().__init__(*args, **kwargs)
            self.expert_idx = expert_idx
            self.num_experts = num_experts
            self._init_expert_weights()
        
        def _init_expert_weights(self):
            """Initialize expert weights with scaled variance and unique initialization"""
            scale = 1.0 / (self.num_experts ** 0.5)  # Scale by 1/sqrt(num_experts)
            
            # Initialize linear layers with scaled variance
            for module in [self.htoh4, self.h4toh]:
                if hasattr(module, 'weight'):
                    # Smaller variance initialization
                    nn.init.normal_(module.weight, mean=0.0, std=scale / (module.weight.shape[1] ** 0.5))
                    # Add unique offset per expert to break symmetry
                    module.weight.data += (self.expert_idx * 0.01 * scale) * torch.randn_like(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.normal_(module.bias, mean=0.0, std=scale * 0.1)

    class GSoftmaxGate(NaiveGate):
        r"""
        A gate that uses gumbel softmax to calculate the score of each expert.
        """
        def __init__(self, d_model, num_expert, world_size, top_k=2, gate_bias=True):
            super().__init__(d_model, num_expert, world_size, top_k, gate_bias)
            self.tau = 1.0
            if two_layer_router:
                self.gate = nn.Sequential(
                        nn.Linear(d_model, int(d_model / 2), bias = gate_bias),
                        # nn.Dropout(dropout),
                        nn.ReLU(inplace=True),
                        nn.Linear(int(d_model / 2), self.tot_expert, bias = gate_bias),
                    )

        def forward(self, inp, return_all_scores=False):
            r"""
            The naive implementation simply calculates the top-k of a linear layer's
            output.
            """
            gate = self.gate(inp)
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.top_k, dim=-1, largest=True, sorted=False
            )  # [.. x top_k]
            gate_top_k_val = gate_top_k_val.view(-1, self.top_k)

            gate_score = F.gumbel_softmax(gate_top_k_val, tau=self.tau, hard=(not self.training))

            # dummy loss
            self.set_loss(torch.zeros(1, requires_grad=True).to(inp.device))

            if return_all_scores:
                return gate_top_k_idx, gate_score, gate
            return gate_top_k_idx, gate_score

    class GSoftmaxHarmonicGate(NaiveGate):
        r"""
        A gate that uses gumbel softmax to calculate the score of each expert.
        """
        def __init__(self, d_model, num_expert, world_size, top_k=2, gate_bias=True):
            super().__init__(d_model, num_expert, world_size, top_k, gate_bias)
            self.tau = 1.0
            if two_layer_router:
                self.gate = nn.Sequential(
                        nn.Linear(d_model, int(d_model / 2), bias = gate_bias),
                        # nn.Dropout(dropout),
                        nn.ReLU(inplace=True),
                        nn.Linear(int(d_model / 2), self.tot_expert, bias = gate_bias),
                    )

        def forward(self, inp, return_all_scores=False):
            r"""
            The naive implementation simply calculates the top-k of a linear layer's
            output.
            """
            gate = self.gate(inp)
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.top_k, dim=-1, largest=True, sorted=False
            )  # [.. x top_k]
            gate_top_k_val = gate_top_k_val.view(-1, self.top_k)

            gate_score = F.gumbel_softmax(gate_top_k_val, tau=self.tau, hard=(not self.training))

            # Compute average expert index weighted by scores
            # Shape: gate_top_k_idx (batch*seq, top_k), gate_score (batch*seq, top_k)
            factor = NUMEXPERTS / len(ratios)  # Number of experts per ratio group
            ratios_based_idx = (gate_top_k_idx / factor).int() # Convert to (0, 7)
            avg_expert_idx = (ratios_based_idx.float() * gate_score).sum(dim=-1).mean()

            # dummy loss
            self.set_loss(torch.zeros(1, requires_grad=True).to(inp.device))

            if return_all_scores:
                return gate_top_k_idx, gate_score, gate, avg_expert_idx
            return gate_top_k_idx, gate_score, avg_expert_idx


    class FMoEViz(FMoE):
        def forward(self, moe_inp):
            r"""
            The FMoE module first computes gate output, and then conduct MoE forward
            according to the gate.  The score of the selected gate given by the
            expert is multiplied to the experts' output tensors as a weight.
            """

            moe_inp_batch_size = tree.flatten(
                tree.map_structure(lambda tensor: tensor.shape[0], moe_inp)
            )
            assert all(
                [batch_size == moe_inp_batch_size[0] for batch_size in moe_inp_batch_size]
            ), "MoE inputs must have the same batch size"

            if self.world_size > 1:

                def ensure_comm_func(tensor):
                    ensure_comm(tensor, self.moe_group)

                tree.map_structure(ensure_comm_func, moe_inp)
            if self.slice_size > 1:

                def slice_func(tensor):
                    return Slice.apply(
                        tensor, self.slice_rank, self.slice_size, self.slice_group
                    )

                moe_inp = tree.map_structure(slice_func, moe_inp)

            gate_top_k_idx, gate_score = self.gate(moe_inp)

            if self.gate_hook is not None:
                self.gate_hook(gate_top_k_idx, gate_score, None)

            # delete masked tensors
            if self.mask is not None and self.mask_dict is not None:
                # TODO: to fix
                def delete_mask_func(tensor):
                    # to: (BxL') x d_model
                    tensor = tensor[mask == 0, :]
                    return tensor

                mask = self.mask.view(-1)
                moe_inp = tree.map_structure(delete_mask_func, moe_inp)
                gate_top_k_idx = gate_top_k_idx[mask == 0, :]

            fwd = _fmoe_general_global_forward(
                moe_inp, gate_top_k_idx, self.expert_fn_single if fmoe_faster_schedule else self.expert_fn,
                self.num_expert, self.world_size,
                experts=self.experts
            )

            # recover deleted tensors
            if self.mask is not None and self.mask_dict is not None:

                def recover_func(tensor):
                    # to: (BxL') x top_k x dim
                    dim = tensor.shape[-1]
                    tensor = tensor.view(-1, self.top_k, dim)
                    # to: (BxL) x top_k x d_model
                    x = torch.zeros(
                        mask.shape[0],
                        self.top_k,
                        dim,
                        device=tensor.device,
                        dtype=tensor.dtype,
                    )
                    # recover
                    x[mask == 0] = tensor
                    for k, v in self.mask_dict.items():
                        x[mask == k] = v
                    return x

                moe_outp = tree.map_structure(recover_func, fwd)
            else:

                def view_func(tensor):
                    dim = tensor.shape[-1]
                    tensor = tensor.view(-1, self.top_k, dim)
                    return tensor

                moe_outp = tree.map_structure(view_func, fwd)

            gate_score = gate_score.view(-1, 1, self.top_k)

            def bmm_func(tensor):
                dim = tensor.shape[-1]
                tensor = torch.bmm(gate_score, tensor).reshape(-1, dim)
                return tensor

            moe_outp = tree.map_structure(bmm_func, moe_outp)

            if self.slice_size > 1:

                def all_gather_func(tensor):
                    return AllGather.apply(
                        tensor, self.slice_rank, self.slice_size, self.slice_group
                    )

                moe_outp = tree.map_structure(all_gather_func, moe_outp)

            moe_outp_batch_size = tree.flatten(
                tree.map_structure(lambda tensor: tensor.shape[0], moe_outp)
            )
            assert all(
                [batch_size == moe_outp_batch_size[0] for batch_size in moe_outp_batch_size]
            ), "MoE outputs must have the same batch size"
            return moe_outp, gate_score, gate_top_k_idx

    class FMoETransformerMLPViz(FMoEViz):
        r"""
        A complete MoE MLP module in a Transformer block.
        * `activation` is the activation function to be used in MLP in each expert.
        * `d_hidden` is the dimension of the MLP layer.
        """

        def __init__(
            self,
            num_expert=32,
            d_model=1024,
            d_hidden=4096,
            activation=torch.nn.GELU(),
            expert_dp_comm="none",
            expert_rank=0,
            **kwargs
        ):
            def one_expert(d_model):
                return _Expert(1, d_model, d_hidden, activation, rank=0)
            
            expert = one_expert
            super().__init__(num_expert=num_expert, d_model=d_model, expert=expert, **kwargs)
            self.mark_parallel_comm(expert_dp_comm)

        def forward(self, inp: torch.Tensor, ret={}):
            r"""
            This module wraps up the FMoE module with reshape, residual and layer
            normalization.
            """
            original_shape = inp.shape
            inp = inp.reshape(-1, self.d_model)
            output, gate_score, top_k_idx = super().forward(inp)
            ret['gate_score'] = gate_score.reshape(original_shape[0], -1, TOPK)
            ret['top_k_idx'] = top_k_idx.reshape(original_shape[0], -1, TOPK)
            return output.reshape(original_shape)

    class FMoETransformerMLPDiffInit(FMoE):
        def __init__(
            self,
            num_expert=32,
            d_model=1024,
            d_hidden=4096,
            activation=torch.nn.GELU(),
            expert_dp_comm="none",
            expert_rank=0,
            **kwargs
        ):
            def one_expert(d_model, expert_idx):
                return _ExpertDiffInit(1, d_model, d_hidden, activation, rank=0, 
                                expert_idx=expert_idx, num_experts=num_expert)
            
            # Create experts with different indices
            expert_list = nn.ModuleList([one_expert(d_model, i) for i in range(num_expert)])
            
            # Manually call parent init with a dummy expert function
            # The parent will create experts, but we'll replace them
            def dummy_expert(d_model):
                pass
            
            super().__init__(num_expert=num_expert, d_model=d_model, expert=dummy_expert, **kwargs)
            
            # Replace the auto-created experts with our custom ones
            self.experts = expert_list
            self.mark_parallel_comm(expert_dp_comm)

        def forward(self, inp: torch.Tensor, ret={}):
            r"""
            Forward pass with proper reshaping
            """
            original_shape = inp.shape
            inp = inp.reshape(-1, self.d_model)
            output = super().forward(inp)
            return output.reshape(original_shape)

    class FMoEHarmonic(FMoE):
        def __init__(
            self,
            num_expert=32,
            d_model=1024,
            world_size=1,
            mp_group=None,
            slice_group=None,
            moe_group=None,
            top_k=2,
            gate=NaiveGate,
            expert=None,
            gate_hook=None,
            mask=None,
            mask_dict=None,
            gate_bias=True,
            d_hidden=None,
            expert_list=None,  # NEW: pass pre-built experts directly
        ):
            super().__init__()
            self.num_expert = num_expert
            self.d_model = d_model
            self.world_size = world_size

            self.slice_group = slice_group
            if mp_group is not None:
                print("[Warning] mp_group is being deprecated")
                self.slice_group = mp_group
            if self.slice_group is None:
                self.slice_size = 1
                self.slice_rank = 0
            else:
                self.slice_size = self.slice_group.size()
                self.slice_rank = self.slice_group.rank()

            self.top_k = top_k
            
            # NEW: support pre-built heterogeneous expert list
            if expert_list is not None:
                self.experts = nn.ModuleList(expert_list)
                self.experts_fused = False
                self.num_expert = len(expert_list)
            elif type(expert) is list:
                self.experts = nn.ModuleList([e(d_model) for e in expert])
                self.experts_fused = False
                self.num_expert = num_expert = len(expert)
            elif expert is not None:
                self.experts = nn.ModuleList([expert(d_model) for _ in range(num_expert)])
                self.experts_fused = False
            else:
                self.experts_fused = True

            if issubclass(gate, NaiveGate):
                self.gate = gate(d_model, num_expert, world_size, top_k, gate_bias=gate_bias)
            else:
                self.gate = gate(d_model, num_expert, world_size, top_k)
            self.gate_hook = gate_hook
            self.mask = mask
            self.mask_dict = mask_dict
            self.moe_group = moe_group
        # def expert_fn(self, inp, fwd_expert_count):
        #     r"""
        #     Optimized expert function that batches operations where possible.
        #     Avoids sequential expert calls by grouping experts with same hidden size.
        #     """
        #     if self.experts_fused:
        #         return self.experts(inp, fwd_expert_count)
            
        #     if isinstance(fwd_expert_count, torch.Tensor):
        #         fwd_expert_count_cpu = fwd_expert_count.cpu().numpy()
            
        #     outputs = []
        #     base_idx = 0
            
        #     for i in range(self.num_expert):
        #         batch_size = fwd_expert_count_cpu[i]
        #         if batch_size > 0:
        #             inp_slice = inp[base_idx : base_idx + batch_size]
        #             # Use fwd_expert_count[i:i+1] to keep tensor shape for FMoELinear
        #             expert_out = self.experts[i](inp_slice, fwd_expert_count[i:i+1])
        #             outputs.append(expert_out)
        #             base_idx += batch_size
            
        #     return torch.cat(outputs, dim=0) if outputs else inp[:0]
        def forward(self, moe_inp):
            r"""
            The FMoE module first computes gate output, and then conduct MoE forward
            according to the gate.  The score of the selected gate given by the
            expert is multiplied to the experts' output tensors as a weight.
            """
            inp_shape = moe_inp.shape

            moe_inp_batch_size = tree.flatten(
                tree.map_structure(lambda tensor: tensor.shape[0], moe_inp)
            )
            assert all(
                [batch_size == moe_inp_batch_size[0] for batch_size in moe_inp_batch_size]
            ), "MoE inputs must have the same batch size"

            if self.world_size > 1:

                def ensure_comm_func(tensor):
                    ensure_comm(tensor, self.moe_group)

                tree.map_structure(ensure_comm_func, moe_inp)
            if self.slice_size > 1:

                def slice_func(tensor):
                    return Slice.apply(
                        tensor, self.slice_rank, self.slice_size, self.slice_group
                    )

                moe_inp = tree.map_structure(slice_func, moe_inp)

            gate_top_k_idx, gate_score, avg_expert_idx = self.gate(moe_inp)

            if self.gate_hook is not None:
                self.gate_hook(gate_top_k_idx, gate_score, None)

            # delete masked tensors
            if self.mask is not None and self.mask_dict is not None:
                # TODO: to fix
                def delete_mask_func(tensor):
                    # to: (BxL') x d_model
                    tensor = tensor[mask == 0, :]
                    return tensor

                mask = self.mask.view(-1)
                moe_inp = tree.map_structure(delete_mask_func, moe_inp)
                gate_top_k_idx = gate_top_k_idx[mask == 0, :]

            fwd = _fmoe_general_global_forward(
                moe_inp, gate_top_k_idx, self.expert_fn_single if fmoe_faster_schedule else self.expert_fn,
                self.num_expert, self.world_size,
                experts=self.experts
            )

            # recover deleted tensors
            if self.mask is not None and self.mask_dict is not None:

                def recover_func(tensor):
                    # to: (BxL') x top_k x dim
                    dim = tensor.shape[-1]
                    tensor = tensor.view(-1, self.top_k, dim)
                    # to: (BxL) x top_k x d_model
                    x = torch.zeros(
                        mask.shape[0],
                        self.top_k,
                        dim,
                        device=tensor.device,
                        dtype=tensor.dtype,
                    )
                    # recover
                    x[mask == 0] = tensor
                    for k, v in self.mask_dict.items():
                        x[mask == k] = v
                    return x

                moe_outp = tree.map_structure(recover_func, fwd)
            else:

                def view_func(tensor):
                    dim = tensor.shape[-1]
                    tensor = tensor.view(-1, self.top_k, dim)
                    return tensor

                moe_outp = tree.map_structure(view_func, fwd)

            gate_score = gate_score.view(-1, 1, self.top_k)

            def bmm_func(tensor):
                dim = tensor.shape[-1]
                tensor = torch.bmm(gate_score, tensor).reshape(-1, dim)
                return tensor

            moe_outp = tree.map_structure(bmm_func, moe_outp)

            if self.slice_size > 1:

                def all_gather_func(tensor):
                    return AllGather.apply(
                        tensor, self.slice_rank, self.slice_size, self.slice_group
                    )

                moe_outp = tree.map_structure(all_gather_func, moe_outp)

            moe_outp_batch_size = tree.flatten(
                tree.map_structure(lambda tensor: tensor.shape[0], moe_outp)
            )
            assert all(
                [batch_size == moe_outp_batch_size[0] for batch_size in moe_outp_batch_size]
            ), "MoE outputs must have the same batch size"

            if viz: return moe_outp, avg_expert_idx, gate_score, gate_top_k_idx
            return moe_outp, avg_expert_idx
    class FMoETransformerMLPHarmonic(FMoEHarmonic):
        r"""
        Optimized heterogeneous MoE MLP with 8 experts of varying hidden sizes.
        """
        def __init__(
            self,
            num_expert=8,
            d_model=1024,
            d_hidden=4096,
            activation=torch.nn.GELU(),
            expert_dp_comm="none",
            expert_rank=0,
            top_k=2,
            **kwargs
        ):
            # Build heterogeneous expert list
            expert_list = []  # Expert 0: identity
            
            # Experts 1-7: variable hidden sizes
            for ratio in ratios:
                hidden = int(d_hidden * ratio)
                for i in range(int(NUMEXPERTS / len(ratios))):  # Repeat each expert type to fill up num_expert
                    if ratio == 0:  # Identity expert
                        expert_list.append(_IdentityExpert())
                    else:
                        expert_list.append(_Expert(1, d_model, hidden, activation, rank=expert_rank))
            
            # Pass pre-built experts directly for efficiency
            super().__init__(
                num_expert=num_expert,
                d_model=d_model,
                expert_list=expert_list,
                top_k=top_k,
                **kwargs
            )
            self.mark_parallel_comm(expert_dp_comm)

        def forward(self, inp: torch.Tensor, ret={}):
            original_shape = inp.shape
            inp = inp.reshape(-1, self.d_model)
            if viz: 
                output, avg_expert_idx, gate_score, top_k_idx = super().forward(inp)
                ret["gate_score"] = gate_score.reshape(original_shape[0], -1, 2)
                ret["top_k_idx"] = top_k_idx.reshape(original_shape[0], -1, 2)
            else: output, avg_expert_idx = super().forward(inp)
            ret["avg_expert_idx"] = avg_expert_idx
            return output.reshape(original_shape)

    class FMoEConcatDim(FMoE):
        def forward(self, moe_inp, layer_type=None):
            r"""
            The FMoE module first computes gate output, and then conduct MoE forward
            according to the gate.  The score of the selected gate given by the
            expert is multiplied to the experts' output tensors as a weight.
            """

            inp_shape = moe_inp.shape
            if len(inp_shape) > 2: # done for curr_timestep, first_agent, or cls_token
                index = -1 if ((layer_type == 'social' and curr_timestep) or (layer_type == 'decoder' and final_timestep)) else 0
                inp_orig = moe_inp
                moe_inp = moe_inp[:, index, :]

            moe_inp_batch_size = tree.flatten(
                tree.map_structure(lambda tensor: tensor.shape[0], moe_inp)
            )
            assert all(
                [batch_size == moe_inp_batch_size[0] for batch_size in moe_inp_batch_size]
            ), "MoE inputs must have the same batch size"

            if self.world_size > 1:

                def ensure_comm_func(tensor):
                    ensure_comm(tensor, self.moe_group)

                tree.map_structure(ensure_comm_func, moe_inp)
            if self.slice_size > 1:

                def slice_func(tensor):
                    return Slice.apply(
                        tensor, self.slice_rank, self.slice_size, self.slice_group
                    )

                moe_inp = tree.map_structure(slice_func, moe_inp)

            gate_top_k_idx, gate_score = self.gate(moe_inp)
            gate_score_ret = gate_score.clone().detach()
            top_k_idx_ret = gate_top_k_idx.clone().detach()
            if len(inp_shape) > 2: # done for cls_token, curr_timestep, or first_agent
                num_tokens = inp_shape[1] #if len(inp_shape) == 3 else inp_shape[1] * inp_shape[1] # should always be 3 
                gate_top_k_idx = gate_top_k_idx.unsqueeze(1).expand(-1, num_tokens, -1).reshape(-1, gate_top_k_idx.shape[-1])  # (batch_size, top_k)
                gate_score = gate_score.unsqueeze(1).expand(-1, num_tokens, -1).reshape(-1, gate_score.shape[-1])  # (batch_size, top_k)
                moe_inp = inp_orig.reshape(-1, inp_shape[-1]) # use original input for expert forward

            if self.gate_hook is not None:
                self.gate_hook(gate_top_k_idx, gate_score, None)

            # delete masked tensors
            if self.mask is not None and self.mask_dict is not None:
                # TODO: to fix
                def delete_mask_func(tensor):
                    # to: (BxL') x d_model
                    tensor = tensor[mask == 0, :]
                    return tensor

                mask = self.mask.view(-1)
                moe_inp = tree.map_structure(delete_mask_func, moe_inp)
                gate_top_k_idx = gate_top_k_idx[mask == 0, :]

            fwd = _fmoe_general_global_forward(
                moe_inp, gate_top_k_idx, self.expert_fn_single if fmoe_faster_schedule else self.expert_fn,
                self.num_expert, self.world_size,
                experts=self.experts
            )

            # recover deleted tensors
            if self.mask is not None and self.mask_dict is not None:

                def recover_func(tensor):
                    # to: (BxL') x top_k x dim
                    dim = tensor.shape[-1]
                    tensor = tensor.view(-1, self.top_k, dim)
                    # to: (BxL) x top_k x d_model
                    x = torch.zeros(
                        mask.shape[0],
                        self.top_k,
                        dim,
                        device=tensor.device,
                        dtype=tensor.dtype,
                    )
                    # recover
                    x[mask == 0] = tensor
                    for k, v in self.mask_dict.items():
                        x[mask == k] = v
                    return x

                moe_outp = tree.map_structure(recover_func, fwd)
            else:

                def view_func(tensor):
                    dim = tensor.shape[-1]
                    tensor = tensor.view(-1, self.top_k, dim)
                    return tensor

                moe_outp = tree.map_structure(view_func, fwd)

            gate_score = gate_score.view(-1, 1, self.top_k)
            gate_score_ret = gate_score_ret.view(-1, 1, self.top_k)

            def bmm_func(tensor):
                dim = tensor.shape[-1]
                tensor = torch.bmm(gate_score, tensor).reshape(-1, dim)
                return tensor

            moe_outp = tree.map_structure(bmm_func, moe_outp)

            if self.slice_size > 1:

                def all_gather_func(tensor):
                    return AllGather.apply(
                        tensor, self.slice_rank, self.slice_size, self.slice_group
                    )

                moe_outp = tree.map_structure(all_gather_func, moe_outp)

            moe_outp_batch_size = tree.flatten(
                tree.map_structure(lambda tensor: tensor.shape[0], moe_outp)
            )
            assert all(
                [batch_size == moe_outp_batch_size[0] for batch_size in moe_outp_batch_size]
            ), "MoE outputs must have the same batch size"
            return moe_outp, gate_score_ret, top_k_idx_ret

    class FMoETransformerMLPConcatDim(FMoEConcatDim):
        r"""
        A complete MoE MLP module in a Transformer block.
        * `activation` is the activation function to be used in MLP in each expert.
        * `d_hidden` is the dimension of the MLP layer.
        """

        def __init__(
            self,
            num_expert=32,
            d_model=1024,
            d_hidden=4096,
            activation=torch.nn.GELU(),
            expert_dp_comm="none",
            expert_rank=0,
            **kwargs
        ):
            def one_expert(d_model):
                return _Expert(1, d_model, d_hidden, activation, rank=0)
            
            expert = one_expert
            super().__init__(num_expert=num_expert, d_model=d_model, expert=expert, **kwargs)
            self.mark_parallel_comm(expert_dp_comm)

        def forward(self, inp: torch.Tensor, ret=None):
            r"""
            This module wraps up the FMoE module with reshape, residual and layer
            normalization.
            """
            original_shape = inp.shape

            if cls_token:
                t_obs, t_fut, num_agents = T_OBS + 1, T_FUT + 1, NUM_AGENTS + 1
            else:
                t_obs, t_fut, num_agents = T_OBS, T_FUT, NUM_AGENTS

            layer_type = None
            if original_shape[0] == t_obs:
                layer_type = 'temporal'
            elif original_shape[0] == num_agents:
                layer_type = 'social'
            elif original_shape[0] == t_fut:
                layer_type = 'decoder'
            
            if concatdim:
                if layer_type == 'temporal':
                    inp = inp.view(t_obs, -1, num_agents, self.d_model).flatten(-2, -1)
                if layer_type == 'social':
                    inp = inp.view(num_agents, -1, t_obs, self.d_model).flatten(-2, -1)
                import pdb; pdb.set_trace()

            if (first_agent or cls_token) and layer_type == 'temporal': # we're in temporal encoder
                inp = inp.view(t_obs, -1, num_agents, self.d_model).flatten(0,1)
            elif (curr_timestep or cls_token) and layer_type == 'social': # we're in social encoder
                inp = inp.view(num_agents, -1, t_obs, self.d_model).flatten(0,1)
            elif (final_timestep or cls_token) and layer_type == 'decoder': # we're in decoder
                inp = inp.permute(1, 0, 2)
            else:
                inp = inp.reshape(-1, self.d_model)
            
            # apply the MOE
            output, gate_score, top_k_idx = super().forward(inp, layer_type=layer_type)
            if (final_timestep or cls_token) and layer_type == 'decoder': # undo the permute
                output = output.reshape(inp.shape).permute(1, 0, 2)
            else:
                gate_score = gate_score.reshape(original_shape[0], -1, TOPK)
                top_k_idx = top_k_idx.reshape(original_shape[0], -1, TOPK)
            
            if ret is not None:
                ret['gate_score'] = gate_score
                ret['top_k_idx'] = top_k_idx
            return output.reshape(original_shape)

    if (concatdim or first_agent or curr_timestep or final_timestep or cls_token): moe_transformer_mlp, moe_gate = FMoETransformerMLPConcatDim, GSoftmaxGate
    elif harmonic: moe_transformer_mlp, moe_gate = FMoETransformerMLPHarmonic, GSoftmaxHarmonicGate
    elif viz: moe_transformer_mlp, moe_gate = FMoETransformerMLPViz, GSoftmaxGate
    elif diff_init: moe_transformer_mlp, moe_gate = FMoETransformerMLPDiffInit, GSoftmaxGate
    else: moe_transformer_mlp, moe_gate = FMoETransformerMLP, GSoftmaxGate

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, use_local_attn=False):
        super().__init__()
        self.use_local_attn = use_local_attn

        if self.use_local_attn:
            self.self_attn = MultiheadAttentionLocal(d_model, nhead, dropout=dropout)
        else:
            self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

        self.activation = _get_activation_fn(activation)
        self.scores = {} # for visualization
        if moe:
            self.dropout = nn.Dropout(dropout)
            self.activation_with_dropout = lambda x: self.dropout(self.activation(x))
            self.moe_ffn = moe_transformer_mlp(d_model=d_model, d_hidden=int(dim_feedforward / (TOPK + SHARED)), gate=moe_gate,
                        num_expert=NUMEXPERTS, top_k=TOPK, activation=self.activation_with_dropout)
        else:
            # Implementation of Feedforward model
            self.linear1 = nn.Linear(d_model, dim_feedforward)
            self.dropout = nn.Dropout(dropout)
            self.linear2 = nn.Linear(dim_feedforward, d_model)
        if SHARED > 0:
            self.linear1_shared = nn.Linear(d_model, int(dim_feedforward / (TOPK + SHARED)))
            self.dropout_shared = nn.Dropout(dropout)
            self.linear2_shared = nn.Linear(int(dim_feedforward / (TOPK + SHARED)), d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.normalize_before = normalize_before
        self.avg_expert_idx = 0 # for harmonic MoE

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     index_pair=None,
                     query_batch_cnt=None,
                     key_batch_cnt=None,
                     index_pair_batch=None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask,
                              index_pair=index_pair, query_batch_cnt=query_batch_cnt,
                              key_batch_cnt=key_batch_cnt, index_pair_batch=index_pair_batch)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        if moe:
            ret = {}
            if viz:
                src2 = self.moe_ffn(src, ret)
                self.scores['gate_score_e'] = ret.get("gate_score", None)
                self.scores['top_k_idx_e'] = ret.get("top_k_idx", None)
                ret = {}
            elif harmonic:
                src2 = self.moe_ffn(src, ret)
                self.avg_expert_idx += ret.get("avg_expert_idx", 0)
                ret = {}
            else:
                src2 = self.moe_ffn(src)
        else:
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src)))) # original code
        if SHARED > 0:
            src2_shared = self.linear2_shared(self.dropout_shared(self.activation(self.linear1_shared(src))))
            src2 = (TOPK * src2 + SHARED * src2_shared) / (TOPK + SHARED)
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    index_pair=None,
                    query_batch_cnt=None,
                    key_batch_cnt=None,
                    index_pair_batch=None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask,
                              index_pair=index_pair, query_batch_cnt=query_batch_cnt,
                              key_batch_cnt=key_batch_cnt, index_pair_batch=index_pair_batch)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                # for local-attn
                index_pair=None,
                query_batch_cnt=None,
                key_batch_cnt=None,
                index_pair_batch=None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos,
                                    index_pair=index_pair, query_batch_cnt=query_batch_cnt,
                                    key_batch_cnt=key_batch_cnt, index_pair_batch=index_pair_batch)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos,
                                 index_pair=index_pair, query_batch_cnt=query_batch_cnt,
                                 key_batch_cnt=key_batch_cnt, index_pair_batch=index_pair_batch)
