import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

def mu_normalize(mu, U_shape):
    int_bounds = [[0, U_shape[i]] for i in range(len(U_shape))]
    mu_total, _ = integrate.nquad(lambda *x: mu(np.array(x)), int_bounds)
    return lambda x: mu(x)/mu_total

def gaussian(x, center, width):
    return np.exp(-1/width * np.sum((x - center)**2))

def mu_display2D(mu, U_shape, title=""):
    plt.figure()
    assert len(U_shape) == 2
    X,Y = np.meshgrid(*[np.linspace(0,U_shape[i]) for i in range(2)])
    _s = np.stack([X.ravel(), Y.ravel()]).T
    plt.title(title)
    plt.contourf(X,Y, np.array(list(map(mu, _s))).reshape(X.shape))
    plt.savefig(f"{title}.svg")
    plt.close("all")

def bounds(x, value, U_shape):
    return value if (0 <= x).all() and (x <= U_shape).all() else 0 

def uniform(U_shape):
    value = np.prod(U_shape)
    assert value > 0
    return lambda x: bounds(x, value, U_shape)

def mu_gaussians(g, U_shape):
    if len(g) == 0:
        return uniform(U_shape)
    else:
        s = lambda x: sum([gaussian(x, gg[0], gg[1]) for gg in g])
        return lambda x: bounds(x, s(x), U_shape)

def fourier_coefficient2distribution(ff, k_bands, c_k=None):
    def dist(x):
        res = 0
        for k in k_bands:
            if c_k is None:
                coeff_k = ff[k]['mu_k']
            else:
                coeff_k = c_k[k]
            res += coeff_k*ff[k]['f_k'](x)
        return res
    return dist