"""
Supported Algorithm:
    - train -> For training generators and discriminators
    - eval -> For evaluating generator networks
"""

from .algorithm import train
from .algorithm import eval_generator as eval
from .algorithm import fid_score

__version__ = '1.0.0'
__author__ = 'Ali Hedayatnia, M.Sc. Student of Artificial Intelligence @ University of Tehran'
