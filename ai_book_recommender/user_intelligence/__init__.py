# -*- coding: utf-8 -*-
"""
🧠 User Intelligence Package
=============================

User modeling, behavior analysis, and online learning.
"""

from .user_model import DynamicUserModel, UserProfiler
from .behavior_sequence import BehaviorSequenceModel, SessionEncoder
from .online_learning import OnlineLearner, FeedbackProcessor

__all__ = [
    "DynamicUserModel",
    "UserProfiler",
    "BehaviorSequenceModel",
    "SessionEncoder",
    "OnlineLearner",
    "FeedbackProcessor",
]
