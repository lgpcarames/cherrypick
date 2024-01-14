"""Top-level package for cherrypick."""

__author__ = """Lucas Carames"""
__email__ = 'lgpcarames@gmail.com'
__version__ = '0.1.0'

from .cherrypick import CherryPick
from .cherrypick import threshold_score
from .cherrypick import __get_features_threshold_score__
from .cherrypick import __best_threshold_classification__
from .cherrypick import __set_difficulty_group__
from .cherrypick import __generate_stats_success__
from .cherrypick import cherry_score


__all__ = [
            "CherryPick",
            "threshold_score",
            "cherry_score   "
]
