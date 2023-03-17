from typing import Callable, Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange

from .STModule import Module


class Trainer:
    def __init__(
            self,
            module: Module,
            max_epochs: int = 10,
            device: str = "cuda",
            precision: int = 32
    ):
        self.module = module
        self.max_epochs = max_epochs
        self.device = device
        self.precision = precision

        self.dump_config()

    def train(
            self,
            train_dataloader: DataLoader,
            valid_dataloader: DataLoader,
    ):
        self.module.model.to(self.device)

        self.module.before_train_epoch()  # <====================================================================== Hook
        for _ in trange(self.max_epochs, desc="Train epoch"):
            self.module.model.train()  # <========================================================================= Hook
            self.module.before_train_batch()  # <================================================================== Hook
            for i, batch in tqdm(enumerate(train_dataloader), desc="Train batch"):
                loss = self.module.on_train_batch(batch, i)  # <=================================================== Hook
                loss.backward()
                self.module.optimizer.step()
                self.module.optimizer.zero_grad()
            self.module.scheduler.step()
            self.module.after_train_batch()  # <=================================================================== Hook

            self.module.model.eval()
            self._dev(
                name="Valid",
                dataloader=valid_dataloader,
                before_batch_fn=self.module.before_valid_batch,
                on_batch_fn=self.module.on_valid_batch,
                after_batch_fn=self.module.after_valid_batch,
            )
        self.module.after_train_epoch()  # <======================================================================= Hook

    @staticmethod
    def _dev(
            name: str,
            dataloader: DataLoader,
            before_batch_fn: Callable[[], None],
            on_batch_fn: Callable[[Any, int], None],
            after_batch_fn: Callable[[], None],
    ):
        before_batch_fn()  # <===================================================================================== Hook
        with torch.no_grad():
            for i, batch in tqdm(enumerate(dataloader), desc=f"{name} batch", leave=False):
                on_batch_fn(batch, i)  # <========================================================================= Hook
        after_batch_fn()  # <====================================================================================== Hook

    def test(
            self,
            test_dataloader: DataLoader,
    ):
        self.module.model.eval()
        self._dev(
            name="Test",
            dataloader=test_dataloader,
            before_batch_fn=self.module.before_test_batch,
            on_batch_fn=self.module.on_test_batch,
            after_batch_fn=self.module.after_test_batch,
        )

    def dump_config(self):
        self.module.config.dump_config()
