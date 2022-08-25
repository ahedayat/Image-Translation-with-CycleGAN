"""
Supported Architectures:
    - Generator -> for Generator Network
    - Discriminator -> for Discriminator Network
"""

from .generator import Generator
from .discriminator import Discriminator
from .nets_utils import ReplayBuffer
from .nets_utils import init_conv_layer_normal as init_layer
from .nets_utils import ReplayBuffer
from .nets_utils import load_net as load
from .nets_utils import save_net as save

__version__ = '1.0.0'
__author__ = 'Ali Hedayatnia, M.Sc. Student of Artificial Intelligence @ University of Tehran'
