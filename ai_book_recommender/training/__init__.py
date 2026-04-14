# -*- coding: utf-8 -*-
"""
🎓 Training Package
====================

Training pipelines and data loaders.
"""

from .train import Trainer, TrainingConfig
from .data_loader import RecommendationDataset, create_data_loaders

__all__ = [
    "Trainer",
    "TrainingConfig",
    "RecommendationDataset",
    "create_data_loaders",
]
