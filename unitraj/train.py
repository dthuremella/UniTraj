import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from torch.utils.data import WeightedRandomSampler
import torch.nn.functional as F

torch.set_float32_matmul_precision('medium')
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from models import build_model
from datasets import build_dataset
from utils.utils import set_seed, find_latest_checkpoint
from pytorch_lightning.callbacks import ModelCheckpoint  # Import ModelCheckpoint
import hydra
from omegaconf import OmegaConf
import os

from models.mtr.transformer.transformer_encoder_layer import viz, moe, SHARED, NUMEXPERTS, TOPK, upsample_hard, harmonic, diff_init, concatdim, first_agent, curr_timestep, final_timestep, cls_token, moe_types, decoder2x, epithet, learnable_targets

if moe:
    from models.mtr.transformer.transformer_encoder_layer import GSoftmaxHarmonicGate
    moe_name = f'fmoe-{TOPK}k{SHARED}s{NUMEXPERTS}e{"S" if "social" in moe_types else ""}{"T" if "temporal" in moe_types else ""}{"D" if "decoder" in moe_types else ""}{"_upsample" if upsample_hard else ""}{epithet}{"_harmonic" if harmonic else ""}{"_diffinit" if diff_init else ""}{"_concatdim" if concatdim else ""}{"_agent0" if first_agent else ""}{"_currT" if curr_timestep else ""}{"_finalT" if final_timestep else ""}{"_cls" if cls_token else ""}-' 
else:
    moe_name = f'{"_upsample" if upsample_hard else ""}{"_decoder{}x".format(decoder2x) if decoder2x else ""}'

restore_best_weights = False  # Set to True to enable restoring best weights after each validation
print( f"RESTORE best weights after each validation: {restore_best_weights}")
taper_tau = False #doesnt seem to help
per_batch_learntarg = False

monitor_metric = 'val/brier_fde' ## tried 'val/minFDE6' to try to reproduce

class EMTargetCallback(Callback):
    def __init__(self, target_lr=1e-3):
        super().__init__()
        self.target_lr = target_lr
        self.optimizer_target = None

    def on_fit_start(self, trainer, pl_module):
        target_params = [p for n, p in pl_module.named_parameters() 
                         if 'target_logits' in n]
        if not target_params:
            print("[EMTargetCallback] No target_logits found — callback inactive.")
            return
        self.optimizer_target = torch.optim.Adam(target_params, lr=self.target_lr)
        self.optimizer_target.zero_grad()  # clean slate
        self._n_batches = 0
        print(f"[EMTargetCallback] Found {len(target_params)} target_logit tensors.")

    def _set_update_targets(self, pl_module, flag: bool):
        if 'temporal' in moe_types:
            for layer in pl_module.temporal_attn_layers:
                for sub_layer in layer.layers:
                    sub_layer.moe_ffn.gate.update_targets = flag
        if 'social' in moe_types:
            for layer in pl_module.social_attn_layers:
                for sub_layer in layer.layers:
                    sub_layer.moe_ffn.gate.update_targets = flag
        if 'decoder' in moe_types:
            for layer in pl_module.tx_decoder:
                layer.moe_ffn.gate.update_targets = flag

    def on_train_epoch_start(self, trainer, pl_module):
        # Reset accumulation counter each epoch
        self._n_batches = 0
        # Make sure target_logits grads are clean at epoch start
        if self.optimizer_target is not None:
            self.optimizer_target.zero_grad()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):

        if self.optimizer_target is None:
            return
        # Freeze model weights, unfreeze target_logits
        for n, p in pl_module.named_parameters():
            p.requires_grad_('target_logits' in n)
        self._set_update_targets(pl_module, True)

        # Second forward pass — need a fresh graph
        pl_module.train()  # ensure train mode
        with torch.enable_grad():
            # Call the underlying forward + loss directly, bypassing Lightning's graph management
            loss = pl_module.training_step(batch, batch_idx)
            loss.backward()
            if per_batch_learntarg:
                self.optimizer_target.step()       # update target_logits right away
                self.optimizer_target.zero_grad()  # clear grad for the next batch

        self._n_batches += 1

        # Restore normal state
        self._set_update_targets(pl_module, False)
        for n, p in pl_module.named_parameters():
            p.requires_grad_('target_logits' not in n)

    def on_train_epoch_end(self, trainer, pl_module):
        if self.optimizer_target is None or self._n_batches == 0:
            return

        # # Normalise accumulated gradients by number of batches
        # for n, p in pl_module.named_parameters():
        #     if 'target_logits' in n and p.grad is not None:
        #         p.grad /= self._n_batches

        if not per_batch_learntarg:
            # Single M-step per epoch
            self.optimizer_target.step()
            self.optimizer_target.zero_grad()

        # # Log learned targets
        # for n, p in pl_module.named_parameters():
        #     if 'target_logits' in n:
        #         learned = F.softmax(p.detach(), dim=0).tolist()
        #         print(f"[EMTargetCallback] {n}: {[f'{v:.3f}' for v in learned]}")

class HardExampleSamplerCallback(Callback):
    """Tracks per-sample loss and rebuilds the train DataLoader sampler each epoch."""

    def __init__(self, train_set, train_batch_size, cfg):
        self.train_set = train_set
        self.train_batch_size = train_batch_size
        self.cfg = cfg
        self.sample_weights = torch.ones(len(train_set))
        self._loss_accumulator = {}  # {global_idx: loss}

    def _build_loader(self):
        sampler = WeightedRandomSampler(
            weights=self.sample_weights,
            num_samples=len(self.sample_weights),
            replacement=True,
        )
        return DataLoader(
            self.train_set,
            batch_size=self.train_batch_size,
            sampler=sampler,
            num_workers=self.cfg.load_num_workers,
            drop_last=False,
            collate_fn=self.train_set.collate_fn,
        )
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        per_sample_loss = pl_module.criterion.per_sample
        indices = batch['input_dict'].get('_idx', None)
        if per_sample_loss is not None and indices is not None:
            for idx, loss_val in zip(indices.tolist(), per_sample_loss.tolist()):
                self._loss_accumulator[idx] = loss_val

    def on_train_epoch_end(self, trainer, pl_module):
        if not self._loss_accumulator:
            return

        n = len(self.train_set)
        # Default unseen samples to the mean of what we observed
        observed_losses = torch.tensor(list(self._loss_accumulator.values()))
        default_loss = observed_losses.mean().item()
        losses = torch.full((n,), default_loss)
        for idx, loss_val in self._loss_accumulator.items():
            losses[idx] = max(loss_val, 1e-6)

        MAX_WEIGHT = 10.0
        l_min, l_max = losses.min(), losses.max()
        if l_max > l_min:
            # Scale linearly: so that max loss → weight 100, min loss → weight 1
            # weights = 1.0 + 99.0 * (losses - l_min) / (l_max - l_min)
            # Scale by 1/x :
            # losses_norm = (losses - l_min) / (l_max - l_min)  # easiest=0, hardest=1
            # eps = 1 / MAX_WEIGHT
            # x = 1 - (1 - eps) * losses_norm       # x in [eps, 1]
            # weights = 1.0 / x                      # weights in [1, 1/eps=100]
            # scale by x^2 : 
            losses_norm = (losses - l_min) / (l_max - l_min)  # easiest=0, hardest=1
            weights = MAX_WEIGHT**2 * losses_norm**2
            weights[weights > MAX_WEIGHT] = MAX_WEIGHT # cap it to avoid putting too much focus on a few samples
        else:
            weights = torch.ones(n)  # all losses identical, sample uniformly

        self.sample_weights = weights
        self._loss_accumulator = {}

        print(f"↺ Rebuilt sampler — weight range [{weights.min():.1f}, {weights.max():.1f}] "
            f"(epoch {trainer.current_epoch})")
class IndexedDataset(torch.utils.data.Dataset): # needed for HardExampleSamplerCallback
    def __init__(self, base):
        self.base = base
        # expose collate_fn so the loader can find it
        self.collate_fn = base.collate_fn

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]
        # sample is probably a dict — just add the index into it
        sample['_idx'] = idx
        return sample

class TauScheduleCallback(Callback):
    """Schedule gate temperature (tau) based on epoch"""
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tau_start = cfg.get('tau_start', None)
    
    def on_epoch_start(self, trainer, pl_module):
        # Only apply if tau_start is configured
        if self.tau_start is not None:
            current_epoch = trainer.current_epoch
            # Call update_tau on the model
            if hasattr(pl_module, 'update_tau'):
                pl_module.update_tau(current_epoch)
                print(f"Epoch {current_epoch}: Updated tau")
            else:
                print(f"Warning: Model does not have update_tau method")

class RestoreBestWeights(Callback):
    def __init__(self):
        self.best_weights = None
        self.best_score = float('inf')
        self.monitor = monitor_metric
        self.mode = 'min'
    
    def on_validation_end(self, trainer, pl_module):
        # Get current validation metric
        current_score = trainer.callback_metrics.get(self.monitor)
        
        if current_score is None:
            return
        
        # Check if this is the best so far
        is_better = (self.mode == 'min' and current_score < self.best_score) or \
                    (self.mode == 'max' and current_score > self.best_score)
        
        if is_better:
            # Save best weights
            self.best_score = current_score
            self.best_weights = {k: v.cpu().clone() for k, v in pl_module.state_dict().items()}
            print(f"✓ New best model at epoch {trainer.current_epoch}: {self.monitor}={current_score:.4f}")
        else:
            # Restore best weights
            if self.best_weights is not None:
                pl_module.load_state_dict({k: v.to(pl_module.device) for k, v in self.best_weights.items()})
                print(f"↺ Restored best weights (epoch {trainer.current_epoch} was worse: {current_score:.4f} vs best {self.best_score:.4f})")

class MetricsPrinterCallback(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        if monitor_metric in metrics:
            print(f"\nEpoch {trainer.current_epoch}: {monitor_metric} = {metrics[monitor_metric]:.4f}")

@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg):
    set_seed(cfg.seed)
    OmegaConf.set_struct(cfg, False)  # Open the struct
    cfg = OmegaConf.merge(cfg, cfg.method)

    model = build_model(cfg)

    train_set = build_dataset(cfg)
    val_set = build_dataset(cfg, val=True)

    train_batch_size = max(cfg.method['train_batch_size'] // len(cfg.devices),  1)
    eval_batch_size = max(cfg.method['eval_batch_size'] // len(cfg.devices), 1)

    call_backs = []

    checkpoint_callback = ModelCheckpoint(
        dirpath=f'/home/x_divth/project/code/moe/UniTraj/checkpoints/{cfg.tx_hidden_size if not moe else ""}{moe_name}{cfg.harmonic_alpha if harmonic else ""}{cfg.method.model_name}{"perbatch" if per_batch_learntarg else ""}{monitor_metric[4:8]}/',
        monitor=monitor_metric,  # Replace with your validation metric
        filename='{epoch}-{val/minFDE6:.2f}',
        save_top_k=1, # save the top k checkpoints, set to -1 to save all checkpoints
        every_n_epochs=1,  # evaluate every n epochs
        save_last=True,  # Always keep the last checkpoint
        mode='min',  # 'min' for loss/error, 'max' for accuracy
    )

    call_backs.append(checkpoint_callback)
    call_backs.append(MetricsPrinterCallback())

    if restore_best_weights:
        call_backs.append(RestoreBestWeights())
    if taper_tau:
        call_backs.append(TauScheduleCallback(cfg))
    if upsample_hard:
        train_set = IndexedDataset(train_set)          # ← wrap to emit indices
        hard_sampler_cb = HardExampleSamplerCallback(train_set, train_batch_size, cfg)
        call_backs.append(hard_sampler_cb)
        model.train_dataloader = hard_sampler_cb._build_loader
    if harmonic and learnable_targets:
        call_backs.append(EMTargetCallback(target_lr=1e-3))  # can tune this lr separately

    train_loader = DataLoader(
        train_set, batch_size=train_batch_size, num_workers=cfg.load_num_workers, drop_last=False,
        collate_fn=train_set.collate_fn)

    val_loader = DataLoader(
        val_set, batch_size=eval_batch_size, num_workers=cfg.load_num_workers, shuffle=False, drop_last=False,
        collate_fn=train_set.collate_fn)

    trainer = pl.Trainer(
        max_epochs=cfg.method.max_epochs,
        logger=None if cfg.debug else WandbLogger(project="unitraj", name=cfg.exp_name, id=cfg.exp_name),
        devices=1 if cfg.debug else cfg.devices,
        gradient_clip_val=cfg.method.grad_clip_norm,
        # accumulate_grad_batches=cfg.method.Trainer.accumulate_grad_batches,
        accelerator="cpu" if cfg.debug else "gpu",
        profiler="simple",
        strategy="auto" if cfg.debug else "ddp_find_unused_parameters_false",
        reload_dataloaders_every_n_epochs=1 if upsample_hard else 0, 
        callbacks=call_backs
    )

    # automatically resume training
    if cfg.ckpt_path is None and not cfg.debug:
        # Pattern to match all .ckpt files in the base_path recursively
        search_pattern = os.path.join('./unitraj', cfg.exp_name, '**', '*.ckpt')
        cfg.ckpt_path = find_latest_checkpoint(search_pattern)

    if upsample_hard:
        # Give the model a way to find the callback's accumulator
        trainer._hard_sampler_callback = hard_sampler_cb
        trainer.fit(model=model, train_dataloaders=None, val_dataloaders=val_loader, ckpt_path=cfg.ckpt_path)

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=cfg.ckpt_path)

if __name__ == '__main__':
    train()