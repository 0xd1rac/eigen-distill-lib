# base_offline_distiller.py
from abc import ABC, abstractmethod
from typing import List, Callable, Optional
import torch
from .base_distiller import BaseDistiller

class BaseOfflineDistiller(BaseDistiller, ABC):
    @abstractmethod
    def compute_teacher_outputs(self, device: str, temperature: float) -> list:
        """
        Compute (or precompute) teacher outputs for all batches.
        
        Returns:
            list: A list (or generator) of teacher outputs—typically one per batch.
        """
        pass
