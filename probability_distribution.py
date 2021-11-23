import numpy as np
from scipy import integrate

def normalize_mu(mu, U_shape):
    int_bounds = [[0, U_shape[i]] for i in range(len(U_shape))]
    mu_total, _ = integrate.nquad(lambda *x: mu(np.array(x)), int_bounds)
    return lambda x: mu(x)/mu_total

def line(x, p1, p2, r):# currently only 2D
    p12 = p2-p1
    mid = (p1+p2)/2
    n = np.array([-p12[1], p12[0]])
    n = n / np.linalg.norm(n)
    dist_p12 = np.dot((x-p1), n)**2
    dist_p1 = sum((x-p1)**2)
    dist_p2 = sum((x-p2)**2)
    if np.dot((x-p1), p12) > 0 and np.dot((x-p2), -p12) > 0:
        dist_seg = dist_p12
    else:
        dist_seg = min(dist_p1, dist_p2)
    return np.exp(r * dist_seg)

def gaussian(x, center, width):
    return np.exp(-width * np.sum((x - center)**2))