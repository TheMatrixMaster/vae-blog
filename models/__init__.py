from .base import *
from .ae import *
from .vanilla_vae import *
from .gamma_vae import *
from .beta_vae import *

# Aliases
VAE = VanillaVAE
GaussianVAE = VanillaVAE

vae_models = {
    'AE':AE,
    'BetaVAE':BetaVAE,
    'GammaVAE':GammaVAE,
    'VanillaVAE':VanillaVAE,
}
