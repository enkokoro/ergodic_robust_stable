import numpy as np
from fourier_functions import Mu
from probability_distribution import mu_normalize, bounds

def renormalize_and_recompute_mu_k(mu, K, U_shape, ff):
    new_mu = mu_normalize(mu, U_shape)
    new_mu_k = Mu(new_mu, K, U_shape, ff)
    return new_mu, new_mu_k

def translate_target_dist(mu, K, U_shape, translation, ff):
    trans = lambda x: bounds(x, mu(x-translation), U_shape)
    return renormalize_and_recompute_mu_k(trans, K, U_shape, ff)

def uniform_noise(v, magnitude):
    return v + np.random.uniform(-magnitude, magnitude, v.shape)

def pointwise_noise_target_dist(mu, K, U_shape, magnitude, ff):
    noise = lambda x: uniform_noise(mu(x), magnitude)
    return renormalize_and_recompute_mu_k(noise, K, U_shape, ff)


    
    
