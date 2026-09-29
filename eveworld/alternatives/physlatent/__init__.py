"""Physics-aware latent adapter experiments for GigaWorld-0.

The baseline trainer is imported lazily; some eval environments have
diffusers/transformers version skew.
"""

from .modules import PhysicsLatentEncoder
from .transforms import PhysLatentGigaWorld0Transform

__all__ = [
    'PhysLatentGigaWorld0Transform',
    'PhysicsLatentEncoder',
    'PhysicsLatentGigaWorld0Trainer',
]


def __getattr__(name):
    if name == 'PhysicsLatentGigaWorld0Trainer':
        from .trainer import PhysicsLatentGigaWorld0Trainer

        return PhysicsLatentGigaWorld0Trainer
    raise AttributeError(name)
