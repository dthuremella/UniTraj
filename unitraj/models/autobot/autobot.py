import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import special
from torch import optim
from torch.distributions import MultivariateNormal, Laplace
from torch.optim.lr_scheduler import MultiStepLR

from unitraj.models.base_model.base_model import BaseModel

from torch import Tensor
from unitraj.models.mtr.transformer.transformer_encoder_layer import viz, moe, SHARED, NUMEXPERTS, TOPK, upsample_hard, harmonic, cls_token, first_agent, curr_timestep, final_timestep, decoder2x
if moe:
    from unitraj.models.mtr.transformer.transformer_encoder_layer import moe_transformer_mlp, moe_gate
    from unitraj.models.mtr.transformer.transformer_encoder_layer import concatdim, T_OBS, NUM_AGENTS, moe_types
    div_by = 1 #TOPK + SHARED

    class MoETransformerEncoderLayer(nn.TransformerEncoderLayer):
        def __init__(self, d_model, nhead, dropout=0.1, dim_feedforward=2048, activation="relu", layer_norm_eps=1e-5,
                    batch_first=False, norm_first=False, bias=True, device=None, dtype=None, layer_type=None):
            super(MoETransformerEncoderLayer, self).__init__(d_model=d_model, nhead=nhead, dropout=dropout,
                                                            dim_feedforward=dim_feedforward, activation=activation,
                                                            layer_norm_eps=layer_norm_eps, 
                                                            batch_first=batch_first,
                                                            norm_first=norm_first, bias=bias, device=device, dtype=dtype)
            if moe:
                if concatdim:
                    if layer_type == 'temporal':
                        d_model *= NUM_AGENTS
                        dim_feedforward *= NUM_AGENTS
                    elif layer_type == 'social':
                        d_model *= T_OBS
                        dim_feedforward *= T_OBS
                self.activation_with_dropout = lambda x: self.dropout(self.activation(x))
                self.moe_ffn = moe_transformer_mlp(d_model=d_model, d_hidden=int(dim_feedforward / div_by), gate=moe_gate,
                            num_expert=NUMEXPERTS, top_k=TOPK, activation=self.activation_with_dropout)
                self.linear1, self.linear2 = None, None
            if SHARED > 0:
                self.linear1_shared = nn.Linear(d_model, int(dim_feedforward / div_by), bias=bias)
                self.dropout_shared = nn.Dropout(dropout)
                self.linear2_shared = nn.Linear(int(dim_feedforward / div_by), d_model, bias=bias)

            if harmonic:
                self.avg_expert_idx = 0 # for harmonic MoE

            if viz:
                self.scores = {}

        # feed forward block
        def _ff_block(self, x: Tensor) -> Tensor:
            if moe:
                ret = {}
                self.scores = {}
                if viz:
                    x = self.moe_ffn(x, ret)
                    self.scores['gate_score_e'] = ret.get("gate_score", None)
                    self.scores['top_k_idx_e'] = ret.get("top_k_idx", None)
                    ret = {}
                elif harmonic:
                    x = self.moe_ffn(x, ret)
                    self.avg_expert_idx += ret.get("avg_expert_idx", 0).item()
                else:
                    x = self.moe_ffn(x)
            else:
                x = self.linear2(self.dropout(self.activation(self.linear1(x))))
            if SHARED > 0:
                x_shared = self.linear2_shared(self.dropout_shared(self.activation(self.linear1_shared(x))))
                x = (TOPK * x + SHARED * x_shared) / (TOPK + SHARED)
            return self.dropout2(x)

    class MoETransformerDecoderLayer(nn.TransformerDecoderLayer):
        def __init__(self, d_model, nhead, dropout=0.1, dim_feedforward=2048, activation="relu", layer_norm_eps=1e-5,
                    batch_first=False, norm_first=False, bias=True, device=None, dtype=None):
            super(MoETransformerDecoderLayer, self).__init__(d_model=d_model, nhead=nhead, dropout=dropout,
                                                            dim_feedforward=dim_feedforward, activation=activation,
                                                            layer_norm_eps=layer_norm_eps, 
                                                            batch_first=batch_first,
                                                            norm_first=norm_first, bias=bias, device=device, dtype=dtype)
            if moe:
                self.activation_with_dropout = lambda x: self.dropout(self.activation(x))
                self.moe_ffn = moe_transformer_mlp(d_model=d_model, d_hidden=int(dim_feedforward / div_by), gate=moe_gate,
                            num_expert=NUMEXPERTS, top_k=TOPK, activation=self.activation_with_dropout)
                self.linear2, self.linear1 = None, None

            if SHARED > 0:
                self.linear1_shared = nn.Linear(d_model, int(dim_feedforward / div_by), bias=bias)
                self.dropout_shared = nn.Dropout(dropout)
                self.linear2_shared = nn.Linear(int(dim_feedforward / div_by), d_model, bias=bias)

            if harmonic:
                self.avg_expert_idx = 0 # for harmonic MoE

            if viz:
                self.scores = {}

        # feed forward block
        def _ff_block(self, x: Tensor) -> Tensor:
            if moe:
                ret = {}
                self.scores = {}
                if viz:
                    x = self.moe_ffn(x, ret)
                    self.scores['gate_score_d'] = ret.get("gate_score", None)
                    self.scores['top_k_idx_d'] = ret.get("top_k_idx", None)
                    ret = {}
                elif harmonic:
                    x = self.moe_ffn(x, ret)
                    self.avg_expert_idx += ret.get("avg_expert_idx", 0).item()
                else:
                    x = self.moe_ffn(x)
            else:
                x = self.linear2(self.dropout(self.activation(self.linear1(x))))
            if SHARED > 0:
                x_shared = self.linear2_shared(self.dropout_shared(self.activation(self.linear1_shared(x))))
                x = (TOPK * x + SHARED * x_shared) / (TOPK + SHARED)
            return self.dropout3(x)

class MapEncoderCNN(nn.Module):
    '''
    Regular CNN encoder for road image.
    '''

    def __init__(self, d_k=64, dropout=0.1, c=10):
        super(MapEncoderCNN, self).__init__()
        self.dropout = dropout
        self.c = c
        init_ = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        # MAP ENCODER
        fm_size = 7
        self.map_encoder = nn.Sequential(
            init_(nn.Conv2d(3, 32, kernel_size=4, stride=1)), nn.ReLU(),
            init_(nn.Conv2d(32, 32, kernel_size=4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(32, 32, kernel_size=3, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(32, 32, kernel_size=3, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(32, fm_size * self.c, kernel_size=2, stride=2)), nn.ReLU(),
            nn.Dropout2d(p=self.dropout)
        )
        self.map_feats = nn.Sequential(
            init_(nn.Linear(7 * 7 * fm_size, d_k)), nn.ReLU(),
            init_(nn.Linear(d_k, d_k)), nn.ReLU(),
        )
        self.fisher_information = None
        self.optimal_params = None

    def forward(self, roads):
        '''
        :param roads: road image with size (B, 128, 128, 3)
        :return: road features, with one for every mode (B, c, d_k)
        '''
        B = roads.size(0)  # batch size
        return self.map_feats(self.map_encoder(roads).view(B, self.c, -1))


class MapEncoderPts(nn.Module):
    '''
    This class operates on the road lanes provided as a tensor with shape
    (B, num_road_segs, num_pts_per_road_seg, k_attr+1)
    '''

    def __init__(self, d_k, map_attr=3, dropout=0.1):
        super(MapEncoderPts, self).__init__()
        self.dropout = dropout
        self.d_k = d_k
        self.map_attr = map_attr
        init_ = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        self.road_pts_lin = nn.Sequential(init_(nn.Linear(map_attr, self.d_k)))
        self.road_pts_attn_layer = nn.MultiheadAttention(self.d_k, num_heads=8, dropout=self.dropout)
        self.norm1 = nn.LayerNorm(self.d_k, eps=1e-5)
        self.norm2 = nn.LayerNorm(self.d_k, eps=1e-5)
        self.map_feats = nn.Sequential(
            init_(nn.Linear(self.d_k, self.d_k)), nn.ReLU(), nn.Dropout(self.dropout),
            init_(nn.Linear(self.d_k, self.d_k)),
        )

    def get_road_pts_mask(self, roads):
        road_segment_mask = torch.sum(roads[:, :, :, -1], dim=2) == 0
        road_pts_mask = (1.0 - roads[:, :, :, -1]).type(torch.BoolTensor).to(roads.device).view(-1, roads.shape[2])
        road_pts_mask = road_pts_mask.masked_fill((road_pts_mask.sum(-1) == roads.shape[2]).unsqueeze(-1), False) # Ensures no NaNs due to empty rows.
        return road_segment_mask, road_pts_mask

    def forward(self, roads, agents_emb):
        '''
        :param roads: (B, S, P, k_attr+1)  where B is batch size, S is num road segments, P is
        num pts per road segment.
        :param agents_emb: (T_obs, B, d_k) where T_obs is the observation horizon. THis tensor is obtained from
        AutoBot's encoder, and basically represents the observed socio-temporal context of agents.
        :return: embedded road segments with shape (S)
        '''
        B = roads.shape[0]
        S = roads.shape[1]
        P = roads.shape[2]
        road_segment_mask, road_pts_mask = self.get_road_pts_mask(roads)
        road_pts_feats = self.road_pts_lin(roads[:, :, :, :self.map_attr]).view(B * S, P, -1).permute(1, 0, 2)

        # Combining information from each road segment using attention with agent contextual embeddings as queries.
        agents_emb = agents_emb[-1].unsqueeze(2).repeat(1, 1, S, 1).view(-1, self.d_k).unsqueeze(0)
        road_seg_emb = self.road_pts_attn_layer(query=agents_emb, key=road_pts_feats, value=road_pts_feats,
                                                key_padding_mask=road_pts_mask)[0]
        road_seg_emb = self.norm1(road_seg_emb)
        road_seg_emb2 = road_seg_emb + self.map_feats(road_seg_emb)
        road_seg_emb2 = self.norm2(road_seg_emb2)
        road_seg_emb = road_seg_emb2.view(B, S, -1)

        return road_seg_emb.permute(1, 0, 2), road_segment_mask


def init(module, weight_init, bias_init, gain=1):
    '''
    This function provides weight and bias initializations for linear layers.
    '''
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_parameter('pe', nn.Parameter(pe, requires_grad=False))

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class OutputModel(nn.Module):
    '''
    This class operates on the output of AutoBot-Ego's decoder representation. It produces the parameters of a
    bivariate Gaussian distribution.
    '''

    def __init__(self, d_k=64):
        super(OutputModel, self).__init__()
        self.d_k = d_k
        init_ = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        self.observation_model = nn.Sequential(
            init_(nn.Linear(d_k, d_k)), nn.ReLU(),
            init_(nn.Linear(d_k, d_k)), nn.ReLU(),
            init_(nn.Linear(d_k, 5))
        )
        self.min_stdev = 0.01

    def forward(self, agent_decoder_state):
        T = agent_decoder_state.shape[0]
        BK = agent_decoder_state.shape[1]
        pred_obs = self.observation_model(agent_decoder_state.reshape(-1, self.d_k)).reshape(T, BK, -1)

        x_mean = pred_obs[:, :, 0]
        y_mean = pred_obs[:, :, 1]
        x_sigma = F.softplus(pred_obs[:, :, 2]) + self.min_stdev
        y_sigma = F.softplus(pred_obs[:, :, 3]) + self.min_stdev
        rho = torch.tanh(pred_obs[:, :, 4]) * 0.9  # for stability
        return torch.stack([x_mean, y_mean, x_sigma, y_sigma, rho], dim=2)


# class AutoBotEgo(nn.Module):
class AutoBotEgo(BaseModel):
    '''
    AutoBot-Ego Class.
    '''

    def __init__(self, config, k_attr=2, map_attr=2):

        super(AutoBotEgo, self).__init__(config)

        self.config = config
        init_ = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        self.T = config['future_len']
        self.past = config['past_len']
        self.fisher_information = None
        self.map_attr = map_attr
        self.k_attr = k_attr
        self.d_k = config['hidden_size']
        self.c = config['num_modes']

        self.L_enc = config['num_encoder_layers']
        self.dropout = config['dropout']
        self.num_heads = config['tx_num_heads']
        self.L_dec = config['num_decoder_layers']
        self.tx_hidden_size = config['tx_hidden_size']

        # INPUT ENCODERS
        self.agents_dynamic_encoder = nn.Sequential(init_(nn.Linear(self.k_attr, self.d_k)))

        self.scores = None
        transformer_encoder_layer = MoETransformerEncoderLayer if moe and ('social' in moe_types or 'temporal' in moe_types) else nn.TransformerEncoderLayer
        transformer_decoder_layer = MoETransformerDecoderLayer if moe and ('decoder' in moe_types) else nn.TransformerDecoderLayer
        if cls_token:
            self.cls_token_temporal = nn.Parameter(torch.Tensor(1, NUM_AGENTS, self.d_k), requires_grad=True)
            nn.init.trunc_normal_(self.cls_token_temporal, std=0.02)
            self.cls_token_social = nn.Parameter(torch.Tensor(T_OBS + 1, 1, self.d_k), requires_grad=True)
            nn.init.trunc_normal_(self.cls_token_social, std=0.02)
            self.cls_token_decoder = nn.Parameter(torch.Tensor(1, 1, self.d_k), requires_grad=True)
            nn.init.trunc_normal_(self.cls_token_decoder, std=0.02)

        # ============================== AutoBot-Ego ENCODER ==============================
        self.social_attn_layers = []
        self.temporal_attn_layers = []
        for _ in range(self.L_enc):
            if moe and ('social' in moe_types): tx_encoder_layer = transformer_encoder_layer(d_model=self.d_k, nhead=self.num_heads, dropout=self.dropout, dim_feedforward=self.tx_hidden_size, layer_type='social')
            else: tx_encoder_layer = transformer_encoder_layer(d_model=self.d_k, nhead=self.num_heads, dropout=self.dropout, dim_feedforward=self.tx_hidden_size)

            self.social_attn_layers.append(nn.TransformerEncoder(tx_encoder_layer, num_layers=1))

            if moe and ('temporal' in moe_types): tx_encoder_layer = transformer_encoder_layer(d_model=self.d_k, nhead=self.num_heads, dropout=self.dropout, dim_feedforward=self.tx_hidden_size, layer_type='temporal')
            else: tx_encoder_layer = transformer_encoder_layer(d_model=self.d_k, nhead=self.num_heads, dropout=self.dropout, dim_feedforward=self.tx_hidden_size)

            self.temporal_attn_layers.append(nn.TransformerEncoder(tx_encoder_layer, num_layers=1))

        self.temporal_attn_layers = nn.ModuleList(self.temporal_attn_layers)
        self.social_attn_layers = nn.ModuleList(self.social_attn_layers)

        # ============================== MAP ENCODER ==========================
        self.map_encoder = MapEncoderPts(d_k=self.d_k, map_attr=self.map_attr, dropout=self.dropout)
        self.map_attn_layers = nn.MultiheadAttention(self.d_k, num_heads=self.num_heads, dropout=0.3)

        # ============================== AutoBot-Ego DECODER ==============================
        self.Q = nn.Parameter(torch.Tensor(self.T, 1, self.c, self.d_k), requires_grad=True)
        nn.init.xavier_uniform_(self.Q)

        self.tx_decoder = []
        for _ in range(self.L_dec):
            self.tx_decoder.append(transformer_decoder_layer(d_model=self.d_k, nhead=self.num_heads,
                                                              dropout=self.dropout,
                                                              dim_feedforward=self.tx_hidden_size*2 if decoder2x else self.tx_hidden_size))
        self.tx_decoder = nn.ModuleList(self.tx_decoder)

        # ============================== Positional encoder ==============================
        self.pos_encoder = PositionalEncoding(self.d_k, dropout=0.0, max_len=self.past+1 if cls_token else self.past)

        # ============================== OUTPUT MODEL ==============================
        self.output_model = OutputModel(d_k=self.d_k)

        # ============================== Mode Prob prediction (P(z|X_1:t)) ==============================
        self.P = nn.Parameter(torch.Tensor(self.c, 1, self.d_k), requires_grad=True)  # Appendix C.2.
        nn.init.xavier_uniform_(self.P)

        self.mode_map_attn = nn.MultiheadAttention(self.d_k, num_heads=self.num_heads)

        self.prob_decoder = nn.MultiheadAttention(self.d_k, num_heads=self.num_heads, dropout=self.dropout)
        self.prob_predictor = init_(nn.Linear(self.d_k, 1))

        self.criterion = Criterion(self.config)

        self.fisher_information = None
        self.optimal_params = None

    def update_tau(self, epoch):
        if not moe:
            return
        num_epochs = self.config['max_epochs'] * 0.7 # should finish training in 70%
        tau_start = self.config['tau_start']
        tau_final = self.config['tau_final']
        tau = max(tau_final, tau_start * (tau_final / tau_start) ** (epoch / num_epochs))
        for transformer_type in [self.temporal_attn_layers, self.social_attn_layers, self.tx_decoder]:
            for layer in transformer_type:
                for sub_layer in layer.layers:
                    sub_layer.moe_ffn.gate.tau = tau

    def generate_decoder_mask(self, seq_len, device):
        ''' For masking out the subsequent info. '''
        subsequent_mask = (torch.triu(torch.ones((seq_len, seq_len), device=device), diagonal=1)).bool()
        return subsequent_mask

    def process_observations(self, ego, agents):
        '''
        :param observations: (B, T, N+2, A+1) where N+2 is [ego, other_agents, env]
        :return: a tensor of only the agent dynamic states, active_agent masks and env masks.
        '''
        # ego stuff
        ego_tensor = ego[:, :, :self.k_attr]
        env_masks_orig = ego[:, :, -1]
        env_masks = (1.0 - env_masks_orig).to(torch.bool)
        env_masks = env_masks.unsqueeze(1).repeat(1, self.c, 1).view(ego.shape[0] * self.c, -1)

        # Agents stuff
        temp_masks = torch.cat((torch.ones_like(env_masks_orig.unsqueeze(-1)), agents[:, :, :, -1]), dim=-1)
        opps_masks = (1.0 - temp_masks).to(torch.bool)  # only for agents.
        opps_tensor = agents[:, :, :, :self.k_attr]  # only opponent states

        return ego_tensor, opps_tensor, opps_masks, env_masks

    def temporal_attn_fn(self, agents_emb, agent_masks, layer):
        '''
        :param agents_emb: (T, B, N, H)
        :param agent_masks: (B, T, N)
        :return: (T, B, N, H)
        '''
        T_obs = agents_emb.size(0)
        B = agent_masks.size(0)
        num_agents = agent_masks.size(2)
        temp_masks = agent_masks.permute(0, 2, 1).reshape(-1, T_obs)
        temp_masks = temp_masks.masked_fill((temp_masks.sum(-1) == T_obs).unsqueeze(-1), False)
        agents_temp_emb = layer(self.pos_encoder(agents_emb.reshape(T_obs, B * (num_agents), -1)),
                                src_key_padding_mask=temp_masks)
        return agents_temp_emb.view(T_obs, B, num_agents, -1)

    def social_attn_fn(self, agents_emb, agent_masks, layer):
        '''
        :param agents_emb: (T, B, N, H)
        :param agent_masks: (B, T, N)
        :return: (T, B, N, H)
        '''
        T_obs, B, num_agents, dim = agents_emb.shape
        agents_emb = agents_emb.permute(2, 1, 0, 3).reshape(num_agents, B * T_obs, -1)
        agents_soc_emb = layer(agents_emb, src_key_padding_mask=agent_masks.view(-1, num_agents))
        agents_soc_emb = agents_soc_emb.view(num_agents, B, T_obs, -1).permute(2, 1, 0, 3)
        return agents_soc_emb

    def _forward(self, inputs):
        '''
        :param ego_in: [B, T_obs, k_attr+1] with last values being the existence mask.
        :param agents_in: [B, T_obs, M-1, k_attr+1] with last values being the existence mask.
        :param roads: [B, S, P, map_attr+1] representing the road network if self.use_map_lanes or
                      [B, 3, 128, 128] image representing the road network if self.use_map_img or
                      [B, 1, 1] if self.use_map_lanes and self.use_map_img are False.
        :return:
            pred_obs: shape [c, T, B, 5] c trajectories for the ego agents with every point being the params of
                                        Bivariate Gaussian distribution.
            mode_probs: shape [B, c] mode probability predictions P(z|X_{1:T_obs})
        '''
        ego_in, agents_in, roads = inputs['ego_in'], inputs['agents_in'], inputs['roads']

        B = ego_in.size(0)
        # Encode all input observations (k_attr --> d_k)
        ego_tensor, _agents_tensor, opps_masks, env_masks = self.process_observations(ego_in, agents_in)
        agents_tensor = torch.cat((ego_tensor.unsqueeze(2), _agents_tensor), dim=2)

        agents_emb = self.agents_dynamic_encoder(agents_tensor)
        if cls_token:
            agents_emb = torch.cat((self.cls_token_temporal.expand(B,-1,-1,-1), agents_emb),dim=1)
            opps_masks = torch.cat((torch.ones(B, 1, NUM_AGENTS).to(opps_masks.device).bool(), opps_masks), dim=1)
            agents_emb = torch.cat((self.cls_token_social.expand(B,-1,-1,-1), agents_emb),dim=2)
            opps_masks = torch.cat((torch.ones(B, T_OBS + 1, 1).to(opps_masks.device).bool(), opps_masks), dim=2)        
        agents_emb = agents_emb.permute(1, 0, 2, 3)
        # Process through AutoBot's encoder
        for i in range(self.L_enc):
            agents_emb = self.temporal_attn_fn(agents_emb, opps_masks, layer=self.temporal_attn_layers[i])
            agents_emb = self.social_attn_fn(agents_emb, opps_masks, layer=self.social_attn_layers[i])
        if cls_token:
            agents_emb = agents_emb[1:, :, 1:]

        ego_soctemp_emb = agents_emb[:, :, 0]  # take ego-agent encodings only.

        orig_map_features, orig_road_segs_masks = self.map_encoder(roads, ego_soctemp_emb)
        map_features = orig_map_features.unsqueeze(2).repeat(1, 1, self.c, 1).view(-1, B * self.c, self.d_k)
        road_segs_masks = orig_road_segs_masks.unsqueeze(1).repeat(1, self.c, 1).view(B * self.c, -1)

        # Repeat the tensors for the number of modes for efficient forward pass.
        context = ego_soctemp_emb.unsqueeze(2).repeat(1, 1, self.c, 1)
        context = context.view(-1, B * self.c, self.d_k)

        # AutoBot-Ego Decoding
        out_seq = self.Q.repeat(1, B, 1, 1).view(self.T, B * self.c, -1)
        time_masks = self.generate_decoder_mask(seq_len=self.T+1 if cls_token else self.T, device=ego_in.device)
        if cls_token:
            out_seq = torch.cat((self.cls_token_decoder.expand(-1, B * self.c, -1), out_seq), dim=0)

        for d in range(self.L_dec):
            ego_dec_emb_map = self.map_attn_layers(query=out_seq, key=map_features, value=map_features,
                                                   key_padding_mask=road_segs_masks)[0]
            out_seq = out_seq + ego_dec_emb_map
            out_seq = self.tx_decoder[d](out_seq, context, tgt_mask=time_masks, memory_key_padding_mask=env_masks)
        if cls_token: 
            out_seq = out_seq[1:]
        out_dists = self.output_model(out_seq).reshape(self.T, B, self.c, -1).permute(2, 0, 1, 3)

        # Mode prediction
        mode_params_emb = self.P.repeat(1, B, 1)
        mode_params_emb = self.prob_decoder(query=mode_params_emb, key=ego_soctemp_emb, value=ego_soctemp_emb)[0]

        mode_params_emb = self.mode_map_attn(query=mode_params_emb, key=orig_map_features, value=orig_map_features,
                                             key_padding_mask=orig_road_segs_masks)[0] + mode_params_emb
        mode_probs = F.softmax(self.prob_predictor(mode_params_emb).squeeze(-1), dim=0).transpose(0, 1)

        # return  [c, T, B, 5], [B, c]
        output = {}
        output['predicted_probability'] = mode_probs  # #[B, c]
        output['predicted_trajectory'] = out_dists.permute(2, 0, 1,
                                                           3)  # [c, T, B, 5] to [B, c, T, 5] to be able to parallelize code

        return output

    def forward(self, batch):
        self.scores = {'gate_score_e_temporal': [], 'top_k_idx_e_temporal': [], 'gate_score_e_social': [], 'top_k_idx_e_social': [], 'gate_score_d': [], 'top_k_idx_d': []} if moe and viz else None
        model_input = {}
        inputs = batch['input_dict']
        agents_in, agents_mask, roads = inputs['obj_trajs'], inputs['obj_trajs_mask'], inputs['map_polylines']
        ego_in = torch.gather(agents_in, 1, inputs['track_index_to_predict'].view(-1, 1, 1, 1).repeat(1, 1,
                                                                                                      *agents_in.shape[
                                                                                                       -2:])).squeeze(1)
        ego_mask = torch.gather(agents_mask, 1, inputs['track_index_to_predict'].view(-1, 1, 1).repeat(1, 1,
                                                                                                       agents_mask.shape[
                                                                                                           -1])).squeeze(
            1)
        agents_in = torch.cat([agents_in[..., :2], agents_mask.unsqueeze(-1)], dim=-1)
        agents_in = agents_in.transpose(1, 2)
        ego_in = torch.cat([ego_in[..., :2], ego_mask.unsqueeze(-1)], dim=-1)
        roads = torch.cat([inputs['map_polylines'][..., :2], inputs['map_polylines_mask'].unsqueeze(-1)], dim=-1)
        model_input['ego_in'] = ego_in
        model_input['agents_in'] = agents_in
        model_input['roads'] = roads
        output = self._forward(model_input)
        loss = self.get_loss(batch, output)
        # if self.training:
        #     loss = self.get_loss(batch, output)
        # else:
        #     loss = 0
        self.output = output
        if moe and viz:
            B, T_obs, num_agents, T_fut = agents_in.shape[0], agents_in.shape[1], agents_in.shape[2] + 1, self.T
            if cls_token: 
                    T_obs += 1
                    num_agents += 1
                    T_fut += 1
            if 'temporal' in moe_types:
                for layer in self.temporal_attn_layers:
                    for sub_layer in layer.layers:
                        if cls_token or first_agent:
                            self.scores['gate_score_e_temporal'].append(sub_layer.scores.get('gate_score_e', None).permute(1, 0, 2))
                            self.scores['top_k_idx_e_temporal'].append(sub_layer.scores.get('top_k_idx_e', None).permute(1, 0, 2))
                        else:  
                            self.scores['gate_score_e_temporal'].append(sub_layer.scores.get('gate_score_e', None).view(T_obs, B, num_agents, -1).permute(1, 0, 2, 3))
                            self.scores['top_k_idx_e_temporal'].append(sub_layer.scores.get('top_k_idx_e', None).view(T_obs, B, num_agents, -1).permute(1, 0, 2, 3))
            if 'social' in moe_types:
                for layer in self.social_attn_layers:
                    for sub_layer in layer.layers:
                        if cls_token or curr_timestep:
                            self.scores['gate_score_e_social'].append(sub_layer.scores.get('gate_score_e', None).permute(1, 0, 2))
                            self.scores['top_k_idx_e_social'].append(sub_layer.scores.get('top_k_idx_e', None).permute(1, 0, 2))
                        else:
                            self.scores['gate_score_e_social'].append(sub_layer.scores.get('gate_score_e', None).view(num_agents, B, T_obs, -1).permute(1, 2, 0, 3))
                            self.scores['top_k_idx_e_social'].append(sub_layer.scores.get('top_k_idx_e', None).view(num_agents, B, T_obs, -1).permute(1, 2, 0, 3))
            if 'decoder' in moe_types:
                for layer in self.tx_decoder:
                    if cls_token or final_timestep:
                        self.scores['gate_score_d'].append(layer.scores.get('gate_score_d', None).reshape(B, self.c, -1))
                        self.scores['top_k_idx_d'].append(layer.scores.get('top_k_idx_d', None).reshape(B, self.c, -1))

                    else:
                        self.scores['gate_score_d'].append(layer.scores.get('gate_score_d', None).reshape(T_fut, B, self.c, -1).permute(1, 0, 2, 3))
                        self.scores['top_k_idx_d'].append(layer.scores.get('top_k_idx_d', None).reshape(T_fut, B, self.c, -1).permute(1, 0, 2, 3))

        return output, loss

    def get_loss(self, batch, prediction):
        inputs = batch['input_dict']
        ground_truth = torch.cat([inputs['center_gt_trajs'][..., :2], inputs['center_gt_trajs_mask'].unsqueeze(-1)],
                                 dim=-1)
        loss = self.criterion(prediction, ground_truth, inputs['center_gt_final_valid_idx'])
        if harmonic and moe:
            avg_expert_idx = 0
            alpha = self.config['harmonic_alpha']
            for transformer_type in [self.temporal_attn_layers, self.social_attn_layers]:
                for layer in transformer_type:
                    for sub_layer in layer.layers:
                        avg_expert_idx += sub_layer.avg_expert_idx
            for layer in self.tx_decoder:
                avg_expert_idx += layer.avg_expert_idx
            loss = loss + alpha * avg_expert_idx
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config['learning_rate'], eps=0.0001)
        scheduler = MultiStepLR(optimizer, milestones=self.config['learning_rate_sched'], gamma=0.5)
        return [optimizer], [scheduler]


class Criterion(nn.Module):
    def __init__(self, config):
        super(Criterion, self).__init__()
        self.config = config

    def forward(self, out, gt, center_gt_final_valid_idx):

        return self.nll_loss_multimodes(out, gt, center_gt_final_valid_idx)

    def get_BVG_distributions(self, pred):
        B = pred.size(0)
        T = pred.size(1)
        mu_x = pred[:, :, 0].unsqueeze(2)
        mu_y = pred[:, :, 1].unsqueeze(2)
        sigma_x = pred[:, :, 2]
        sigma_y = pred[:, :, 3]
        rho = pred[:, :, 4]

        # Create the base covariance matrix for a single element
        cov = torch.stack([
            torch.stack([sigma_x ** 2, rho * sigma_x * sigma_y], dim=-1),
            torch.stack([rho * sigma_x * sigma_y, sigma_y ** 2], dim=-1)
        ], dim=-2)

        # Expand this base matrix to match the desired shape
        biv_gauss_dist = MultivariateNormal(loc=torch.cat((mu_x, mu_y), dim=-1), covariance_matrix=cov,validate_args=False)
        return biv_gauss_dist

    def get_Laplace_dist(self, pred):
        return Laplace(pred[:, :, :2], pred[:, :, 2:4],validate_args=False)

    def nll_pytorch_dist(self, pred, data, mask, rtn_loss=True):
        # biv_gauss_dist = get_BVG_distributions(pred)
        biv_gauss_dist = self.get_Laplace_dist(pred)
        num_active_per_timestep = mask.sum()
        data_reshaped = data[:, :, :2]
        if rtn_loss:
            # return (-biv_gauss_dist.log_prob(data)).sum(1)  # Gauss
            return ((-biv_gauss_dist.log_prob(data_reshaped)).sum(-1) * mask).sum(1)  # Laplace
        else:
            # return (-biv_gauss_dist.log_prob(data)).sum(-1)  # Gauss
            # need to multiply by masks
            # return (-biv_gauss_dist.log_prob(data_reshaped)).sum(dim=(1, 2))  # Laplace
            return ((-biv_gauss_dist.log_prob(data_reshaped)).sum(dim=2) * mask).sum(1)  # Laplace

    def nll_loss_multimodes(self, output, data, center_gt_final_valid_idx):
        """NLL loss multimodes for training. MFP Loss function
        Args:
          pred: [K, T, B, 5]
          data: [B, T, 5]
          modes_pred: [B, K], prior prob over modes
          noise is optional
        """
        modes_pred = output['predicted_probability']
        pred = output['predicted_trajectory'].permute(1, 2, 0, 3)
        mask = data[..., -1]

        entropy_weight = self.config['entropy_weight']
        kl_weight = self.config['kl_weight']
        use_FDEADE_aux_loss = self.config['use_FDEADE_aux_loss']

        modes = len(pred)
        nSteps, batch_sz, dim = pred[0].shape

        log_lik_list = []
        with torch.no_grad():
            for kk in range(modes):
                nll = self.nll_pytorch_dist(pred[kk].transpose(0, 1), data, mask, rtn_loss=False)
                log_lik_list.append(-nll.unsqueeze(1))  # Add a new dimension to concatenate later

        # Concatenate the list to form the log_lik tensor
            log_lik = torch.cat(log_lik_list, dim=1)

            priors = modes_pred
            log_priors = torch.log(priors)
            log_posterior_unnorm = log_lik + log_priors

            # Compute logsumexp for normalization, ensuring no in-place operations
            logsumexp = torch.logsumexp(log_posterior_unnorm, dim=-1, keepdim=True)
            log_posterior = log_posterior_unnorm - logsumexp

            # Compute the posterior probabilities without in-place operations
            post_pr = torch.exp(log_posterior)
            # Ensure post_pr is a tensor on the correct device
            post_pr = post_pr.to(data.device)

        # Compute loss.
        loss = 0.0
        for kk in range(modes):
            nll_k = self.nll_pytorch_dist(pred[kk].transpose(0, 1), data, mask, rtn_loss=True) * post_pr[:, kk]
            loss += nll_k.mean()

        # Adding entropy loss term to ensure that individual predictions do not try to cover multiple modes.
        entropy_vals = []
        for kk in range(modes):
            entropy_vals.append(self.get_BVG_distributions(pred[kk]).entropy())
        entropy_vals = torch.stack(entropy_vals).permute(2, 0, 1)
        entropy_loss = torch.mean((entropy_vals).sum(2).max(1)[0])
        loss += entropy_weight * entropy_loss

        # KL divergence between the prior and the posterior distributions.
        kl_loss_fn = torch.nn.KLDivLoss(reduction='batchmean')  # type: ignore
        kl_loss = kl_weight * kl_loss_fn(torch.log(modes_pred), post_pr)

        # compute ADE/FDE loss - L2 norms with between best predictions and GT.
        if use_FDEADE_aux_loss:
            adefde_loss = self.l2_loss_fde(pred, data, mask)
        else:
            adefde_loss = torch.tensor(0.0).to(data.device)

        # post_entropy
        final_loss = loss + kl_loss + adefde_loss

        if upsample_hard: 
            # --- ADD: per-sample loss (no gradient needed) ---
            with torch.no_grad():
                per_sample = torch.zeros(data.size(0), device=data.device)
                for kk in range(modes):
                    nll_k = self.nll_pytorch_dist(pred[kk].transpose(0, 1), data, mask, rtn_loss=True)
                    per_sample += nll_k * post_pr[:, kk]   # shape [B]
                self.per_sample  = per_sample
            # --- END ADD ---

        return final_loss

    def l2_loss_fde(self, pred, data, mask):

        fde_loss = (torch.norm((pred[:, -1, :, :2].transpose(0, 1) - data[:, -1, :2].unsqueeze(1)), 2, dim=-1) * mask[:,
                                                                                                                 -1:])
        ade_loss = (torch.norm((pred[:, :, :, :2].transpose(1, 2) - data[:, :, :2].unsqueeze(0)), 2,
                               dim=-1) * mask.unsqueeze(0)).mean(dim=2).transpose(0, 1)
        loss, min_inds = (fde_loss + ade_loss).min(dim=1)
        return 100.0 * loss.mean()


