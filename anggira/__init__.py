"""
Anggira AI - From-Scratch AI Engineering Project

Named after "messenger" - an AI built step by step from the ground up.

This project follows the ai-engineering-from-scratch curriculum (20 phases)
implemented in pure Python with minimal dependencies.

Modules available:
- Core AI modules: linear algebra, autodiff, probability, ML algorithms
- Deep learning: neural networks, transformers, GPT
- Specialized: NLP, audio, vision, multimodal, autonomous systems
- Advanced: agent engineering, swarm intelligence, infrastructure
- Capstone: agent harness, training pipeline, research agent, distributed training
- Ethics & Safety: alignment, red teaming, bias detection
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

# Specialized AI modules
from .nlp import *
from .audio import *
from .cv import *
from .multimodal import *
from .generative import *
from .rl import *
from .transformers import *

# Advanced systems
from .agent import *
from .autonomous import *
from .swarm import *
from .infra import *
from .llm_eng import *

# Ethics & Safety
from .safety import *

# Capstone projects
from .capstone import *
from .capstone_agent_harness import *
from .capstone_training import *
from .capstone_research import *
from .capstone_distributed import *

# Indonesian language modules (NEW)
from .indonesian import *
from .indonesian_tokenizer import *
from .indonesian_vocabulary import *
from .indonesian_gpt import *

__all__ = [
    # Core modules
    'core', 'autodiff', 'probability', 'ml', 'nn',
    'transformer', 'gpt',
    'nlp', 'audio', 'cv', 'multimodal', 'generative', 'rl', 'transformers',
    'agent', 'autonomous', 'swarm', 'infra', 'llm_eng',
    'safety',
    'capstone', 'capstone_agent_harness', 'capstone_training', 
    'capstone_research', 'capstone_distributed',
    # Indonesian modules
    'indonesian', 'indonesian_tokenizer', 'indonesian_vocabulary',
    'indonesian_gpt',
]