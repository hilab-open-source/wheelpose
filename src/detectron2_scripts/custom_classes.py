from detectron2.engine import LRScheduler
from detectron2.engine import HookBase
from detectron2.engine import DefaultTrainer
from detectron2.data import build_detection_train_loader, build_detection_test_loader, DatasetMapper
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_every_n_seconds
from detectron2.evaluation.coco_evaluation import COCOEvaluator
import detectron2.utils.comm as comm

import logging
import time
import datetime
import torch
import os
import numpy as np

class CustomTrainer(DefaultTrainer):
#    @classmethod
#   def build_evaluator(cls, cfg, dataset_name, output_folder=None):
#        if output_folder is None:
#            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
#        COCOEvaluator(dataset_name, cfg, True, output_folder)
    
    def build_hooks(self):
        hooks = super().build_hooks()
        
        optimizer = self.optimizer
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        
        hooks.insert(-1, LossEvalHook(
            self.cfg.TEST.EVAL_PERIOD,
            self.model,
            optimizer,
            scheduler,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.VAL[0],
                DatasetMapper(self.cfg, True),
            )
        ))
        hooks = hooks[:-2] + hooks[-2:][::-1]
        return hooks
                
class LossEvalHook(LRScheduler):
    def __init__(self, eval_period, model, optimizer, scheduler, data_loader):
        super().__init__(optimizer=optimizer, scheduler=scheduler)
        self._model = model
        self._period = eval_period
        self._orig_data_loader = data_loader
        self._data_loader = iter(data_loader)
    
    def _do_loss_eval(self):
        # Copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
            
        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        for idx, inputs in enumerate(self._data_loader):            
            #if torch.cuda.is_available():
                #torch.cuda.synchronize()

            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
        mean_loss = np.mean(losses)
        self.trainer.storage.put_scalar('validation_loss', mean_loss)
        comm.synchronize()

        return mean_loss
            
    def _get_loss(self, data):
        # How loss is calculated on train_loop 
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced
        
    def _run_scheduler(self, total_val_loss):
        lr = self._optimizer.param_groups[self._best_param_group_id]["lr"]
        self.trainer.storage.put_scalar("lr", lr, smoothing_hint=False)
        self.scheduler.step(total_val_loss)
        
    def _TEST_calc_val_loss(self):
        try:
            data = next(self._data_loader)
        except:
            print("Problem loading next validation image, resetting dataloader.")
            self._data_loader = iter(self._orig_data_loader)
            data = next(self._data_loader)
        with torch.no_grad():
            loss_dict = self.trainer.model(data)
            
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {"val_" + k: v.item() for k, v in 
                                 comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            
            self.trainer.storage.put_scalars(total_val_loss=losses_reduced, 
                                                 **loss_dict_reduced)
       
            return losses_reduced
        
    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            mean_loss = self._TEST_calc_val_loss()
            self._run_scheduler(mean_loss)
        self.trainer.storage.put_scalars(timetest=12)
                
class ValLossScheduler(LRScheduler):
    def __init__(self, cfg, optimizer=None, scheduler=None):
        super().__init__()
        self.cfg = cfg.clone()
        self.cfg.DATASETS.TRAIN = cfg.DATASETS.VAL
        self._loader = iter(build_detection_train_loader(self.cfg))
        """
        Args:
            optimizer (torch.optim.Optimizer):
            scheduler (torch.optim.LRScheduler or fvcore.common.param_scheduler.ParamScheduler):
                if a :class:`ParamScheduler` object, it defines the multiplier over the base LR
                in the optimizer.

        If any argument is not given, will try to obtain it from the trainer.
        """
        self._optimizer = optimizer
        self._scheduler = scheduler
    
    def after_step(self):
        total_val_loss = self._calc_val_loss()
        
        lr = self._optimizer.param_groups[self._best_param_group_id]["lr"]
        self.trainer.storage.put_scalar("lr", lr, smoothing_hint=False)
        
        try:
            self.scheduler.step(total_val_loss)
        except:
            print('Error in Scheduler Step')
        
    def _calc_val_loss(self):
        data = next(self._loader)
        with torch.no_grad():
            loss_dict = self.trainer.model(data)
            
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {"val_" + k: v.item() for k, v in 
                                 comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            
            self.trainer.storage.put_scalars(total_val_loss=losses_reduced, 
                                                 **loss_dict_reduced)
       
            return losses_reduced
