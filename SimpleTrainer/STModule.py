from typing import Any

import torch

from .STConfig import Config
from .STLogger import BaseLogger


class Module:
    def __init__(self):
        self.model: torch.nn.Module = None
        self.optimizer: torch.optim.Optimizer = None
        self.scheduler: torch.optim.lr_scheduler.LRScheduler = None
        self.logger: BaseLogger = None
        self.config: Config = None

    def log(self, name: str, value: Any):
        self.logger.log({name: value})

    def log_dict(self, d: dict[str:Any]):
        self.logger.log(d)

    # ==================================================================================================================
    # ===== Train
    # ==================================================================================================================
    def before_train_epoch(self):
        pass

    def before_train_batch(self):
        pass

    def on_train_batch(self, batch, batch_idx) -> torch.Tensor:
        pass

    def after_train_batch(self):
        pass

    def after_train_epoch(self):
        pass

    # ==================================================================================================================
    # ===== Valid
    # ==================================================================================================================
    def before_valid_batch(self):
        pass

    def on_valid_batch(self, batch, batch_idx):
        pass

    def after_valid_batch(self):
        pass

    # ==================================================================================================================
    # ===== Test
    # ==================================================================================================================
    def before_test_epoch(self):
        pass

    def before_test_batch(self):
        pass

    def on_test_batch(self, batch, batch_idx):
        pass

    def after_test_batch(self):
        pass

    def after_test_epoch(self):
        pass
