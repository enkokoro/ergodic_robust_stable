import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

def mu_normalize(mu, U_shape):
    int_bounds = [[0, U_shape[i]] for i in range(len(U_shape))]
    mu_total, _ = integrate.nquad(lambda *x: mu(np.array(x)), int_bounds)
    return lambda x: mu(x)/mu_total

def gaussian(x, center, width):
    return np.exp(-width * np.sum((x - center)**2))

def mu_display2D(mu, U_shape):
    assert len(U_shape) == 2
    X,Y = np.meshgrid(*[np.linspace(0,U_shape[i]) for i in range(2)])
    _s = np.stack([X.ravel(), Y.ravel()]).T

    plt.contourf(X,Y, np.array(list(map(mu, _s))).reshape(X[0].shape))