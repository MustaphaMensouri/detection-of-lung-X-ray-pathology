import logging
import warnings
from typing import List, Sequence

import hydra
from lightning import Callback
from omegaconf import DictConfig
from lightning.pytorch.loggers import Logger

def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    callbacks: List[Callback] = []
    if not callbacks_cfg:
        return callbacks

    for _, cb_conf in callbacks_cfg.items():
        if "_target_" in cb_conf:
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks

def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    loggers: List[Logger] = []
    if not logger_cfg:
        return loggers

    for _, lg_conf in logger_cfg.items():
        if "_target_" in lg_conf:
            loggers.append(hydra.utils.instantiate(lg_conf))

    return loggers