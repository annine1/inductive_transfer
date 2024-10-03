
from neuralhydrology.augmentation.mixup import Mixup
from neuralhydrology.augmentation.noise import Noise
# from neuralhydrology.augmentation.windowwarp import WindowWarp

from neuralhydrology.utils.config import Config

import torch.nn as nn

SINGLE_FREQ_AUGMENTS = ["mixup", "noise", "windowwarp"]

def get_augment(cfg: Config ):

    """Get augment object, depending on the run configuration.
        
        Parameters
        ----------
        cfg : Config
            The run configuration.

        Returns
        -------
        nn.Module
            A new augment instance of the type specified in the config.
    """

    if cfg.augment.lower() in SINGLE_FREQ_AUGMENTS and len(cfg.use_frequencies) > 1:
        raise ValueError(f"AUGMENT {cfg.augment} does not support multiple frequencies.")


    #if cfg.augment.lower() != "mixu" and cfg.mass_inputs:
    #    raise ValueError(f"The use of 'mass_inputs' with {cfg.augment} is not supported.")

    if cfg.augment.lower() == "mixup":
        augment = Mixup(cfg=cfg)
    elif cfg.augment.lower() == "noise":
        augment = Noise(cfg=cfg)    
    # elif cfg.augment.lower() == "windowwarp":
        # augment = WindowWarp(cfg=cfg)
    else:
        raise NotImplementedError(f"{cfg.augment} not implemented or not linked in `get_augment()`")

    return augment    

        