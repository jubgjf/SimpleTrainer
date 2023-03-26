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
            amp: bool = False
    ):
        self.module = module
        self.max_epochs = max_epochs
        self.device = device
        self.amp = amp

        self.dump_config()

    def train(
            self,
            train_dataloader: DataLoader,
            valid_dataloader: DataLoader,
    ):
        self.module.model.to(self.device)

        if self.amp:
            scaler = torch.cuda.amp.GradScaler()

        self.module.before_train_epoch()
        for _ in trange(self.max_epochs, desc="Train epoch"):
            self.module.model.train()
            self.module.before_train_batch()
            for i, batch in tqdm(enumerate(train_dataloader), desc="Train batch"):
                self.module.optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=self.amp):
                    loss = self.module.on_train_batch(batch, i)
                if self.amp:
                    scaler.scale(loss).backward()
                    scaler.step(self.module.optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    self.module.optimizer.step()
            self.module.scheduler.step()
            self.module.after_train_batch()

            self.module.model.eval()
            self._dev(
                name="Valid",
                dataloader=valid_dataloader,
                before_batch_fn=self.module.before_valid_batch,
                on_batch_fn=self.module.on_valid_batch,
                after_batch_fn=self.module.after_valid_batch,
            )
        self.module.after_train_epoch()

    @staticmethod
    def _dev(
            name: str,
            dataloader: DataLoader,
            before_batch_fn: Callable[[], None],
            on_batch_fn: Callable[[Any, int], None],
            after_batch_fn: Callable[[], None],
    ):
        before_batch_fn()
        with torch.no_grad():
            for i, batch in tqdm(enumerate(dataloader), desc=f"{name} batch", leave=False):
                on_batch_fn(batch, i)
        after_batch_fn()

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
