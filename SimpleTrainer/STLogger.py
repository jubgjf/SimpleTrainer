from abc import abstractmethod

import wandb

from .STConfig import Config


class BaseLogger:
    def __init__(self):
        pass

    @abstractmethod
    def log(self, d: dict):
        pass


class ConsoleLogger(BaseLogger):
    def __init__(self):
        super().__init__()

    def log(self, d: dict):
        log = ""
        for k, v in d.items():
            log += str(k) + ": " + str(v)


class WandbLogger(BaseLogger):
    def __init__(self, project: str, mode: str, config: Config, *args, **kwargs):
        super().__init__()
        wandb.init(project=project, mode=mode, config=config.__dict__, *args, **kwargs)

    def log(self, d: dict):
        wandb.log(d)
