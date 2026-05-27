import pytorch_lightning as pl
import torch

torch.set_float32_matmul_precision('medium')
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from models import build_model
from datasets import build_dataset
from utils.utils import set_seed
import hydra
from omegaconf import OmegaConf

from pytorch_lightning.callbacks import Callback
from models.mtr.transformer.transformer_encoder_layer import viz, TOPK, moe
from train import moe_name
import pickle

@hydra.main(version_base=None, config_path="configs", config_name="config")
def evaluation(cfg):
    set_seed(cfg.seed)
    OmegaConf.set_struct(cfg, False)  # Open the struct
    cfg = OmegaConf.merge(cfg, cfg.method)
    cfg['eval'] = True

    model = build_model(cfg)

    val_set = build_dataset(cfg, val=True)

    eval_batch_size = cfg.method['eval_batch_size']

    val_loader = DataLoader(
        val_set, batch_size=eval_batch_size, num_workers=cfg.load_num_workers, shuffle=False, drop_last=False,
        collate_fn=val_set.collate_fn)

    if viz:
        class ScoreSaver(Callback):
            def __init__(self):
                self.scores = {}
                self.xs = []
                self.ys = []
                self.ypreds = []
                self.kalman = []
                self.traj_types = []
            
            def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
                if outputs is not None:
                    input_dict = batch['input_dict']
                    obj_trajs, obj_trajs_mask = input_dict['obj_trajs'], input_dict['obj_trajs_mask']
                    map_polylines, map_polylines_mask = input_dict['map_polylines'], input_dict['map_polylines_mask']
                    obj_valid_mask = (obj_trajs_mask.sum(dim=-1) > 0)  # (num_center_objects, num_objects)
                    num_objects = obj_valid_mask.shape[1]
                    map_valid_mask = (map_polylines_mask.sum(dim=-1) > 0)  # (num_center_objects, num_polylines)
                    global_token_mask = torch.cat((obj_valid_mask, map_valid_mask), dim=1)
                    batch_size, num_tok = global_token_mask.shape

                    if moe:
                        for key in model.scores:
                            if '_e_' not in key and model.config.method.model_name == 'MTR':
                                continue
                            if key not in self.scores:
                                self.scores[key] = {}
                            for i in range(len(model.scores[key])):
                                if i not in self.scores[key]:
                                    self.scores[key][i] = []

                                if model.config.method.model_name == 'autobot':
                                    self.scores[key][i].append(model.scores[key][i].cpu())

                                if model.config.method.model_name == 'MTR':    
                                    # Create output tensor filled with -1
                                    expert_choices_full = torch.full((batch_size, num_tok, TOPK), -1, dtype=model.scores[key][i].dtype)
                                    # Assign the expert choices back to valid positions
                                    expert_choices_full[global_token_mask] = model.scores[key][i].flatten(0,1).cpu()  # 71209 x 2 tensor -> 128 x 832 x 2 tensor

                                    self.scores[key][i].append(expert_choices_full[:,:num_objects].cpu())

                    # xs is (128, num_objects, n_future_timesteps, 6) where the 6 is px,py,vx,vy,ax,ay
                    self.xs.append(torch.cat((obj_trajs[:,:,:,:2], obj_trajs[:,:,:,35:]), dim=-1).cpu())
                    self.ys.append(input_dict['center_gt_trajs'].cpu())
                    self.ypreds.append(model.output['predicted_trajectory'].cpu())
                    self.kalman.append(input_dict['kalman_difficulty'].cpu())
                    self.traj_types.append(input_dict['trajectory_type'].cpu())
        score_saver = ScoreSaver()

    trainer = pl.Trainer(
        inference_mode=True,
        logger=None if cfg.debug else WandbLogger(project="unitraj", name=cfg.exp_name),
        devices=1,
        accelerator="cpu" if cfg.debug else "gpu",
        profiler="simple",
        callbacks=[score_saver] if viz else None,
    )

    trainer.validate(model=model, dataloaders=val_loader, ckpt_path=cfg.ckpt_path)

    if viz:
        for key in score_saver.scores:
            for i in range(len(score_saver.scores[key])):
                score_saver.scores[key][i] = torch.cat(score_saver.scores[key][i], dim=0)  # Concatenate along the batch dimension
        score_saver.xs = torch.cat(score_saver.xs, dim=0)
        score_saver.ys = torch.cat(score_saver.ys, dim=0)
        score_saver.ypreds = torch.cat(score_saver.ypreds, dim=0)
        score_saver.kalman = torch.cat(score_saver.kalman, dim=0)
        score_saver.traj_types = torch.cat(score_saver.traj_types, dim=0)
        # convert scores to a DataFrame and save as CSV for easier analysis
        data_dump = {'scores': score_saver.scores, 'x': score_saver.xs, 'y': score_saver.ys, 'ypred': score_saver.ypreds, 'kalman': score_saver.kalman, 'traj_types': score_saver.traj_types}
        pickle.dump(data_dump, open('viz_{}{}{}.pkl'.format(cfg.tx_hidden_size if not moe else "", moe_name, cfg.method.model_name), 'wb'))

if __name__ == '__main__':
    evaluation()
