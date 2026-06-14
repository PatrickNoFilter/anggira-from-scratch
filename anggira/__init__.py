"""
Anggira AI — From-Scratch Transformer with Curriculum Learning

Pure Python/NumPy implementation. No PyTorch, no TensorFlow.
"""

# Core modules
from .core import *
from .autodiff import *
from .probability import *
from .ml import *
from .nn import *

# Deep learning modules
from .transformer import *
from .gpt import *

# Specialized
from .nlp import *
from .generative import *
from .transformers import *

# Infrastructure
from .infra import *
from .llm_eng import *

__all__ = [
    'core', 'autodiff', 'probability', 'ml', 'nn',
    'transformer', 'gpt',
    'nlp', 'generative', 'transformers',
    'infra', 'llm_eng',
]
