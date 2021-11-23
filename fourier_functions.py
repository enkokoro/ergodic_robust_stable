import numpy as np
from scipy import integrate
import torch 
import casadi

""" 
Interface 
    Definitions
        mu
        U_shape
        K

    Functions
        Fourier_Functions
            {
                new_k
                h_k
                lambda_k
                f_k
                df_k
                casadi_f_k
            }
        
        Constants(K, U_shape)
            {
                new_k
                h_k
                lambda_k
            }
        Mu(mu, K, U_shape, torch_fourier_functions)
            {
                mu_k
            }
        Torch_Fourier_Functions(K, U_shape)
            {
                f_k
                df_k
            }
        Casadi_Fourier_Functions(K, U_shape)
            {
                casadi_f_k
            }

"""

""" Grab All """
def Fourier_Functions(mu, U_shape, K, printProgress=False):
    if printProgress:
        print(f"Fourier Functions (mu, U_shape=",U_shape,", K=",K,")...")

    if printProgress:
        print("Computing constants (new_k, h_k, lambda_k)...")
    constants = Constants(K, U_shape)
    
    if printProgress:
        print("Computing torch fourier functions (f_k, df_k)...")
    torch_ff = Torch_Fourier_Functions(K, U_shape)

    if printProgress:
        print("Computing mu fourier coefficients (mu_k)... this will take a while because integration...")
    mu_fc = Mu(mu, K, U_shape, torch_ff)

    if printProgress:
        print("Computing casadi fourier function (casadi_f_k)...")
    casadi_ff = Casadi_Fourier_Functions(K, U_shape)

    if printProgress:
        print("Aggregating all constants and functions into one dict...")
    all = {}
    n = len(U_shape)
    for k in np.ndindex(*[K]*n):
        all[k] = {}
        all[k].update(constants[k])
        all[k].update(torch_ff[k])
        all[k].update(mu_fc[k])
        all[k].update(casadi_ff[k])
    return all

def Fourier_Functions_Visualize(U_shape, ff):
    print(ff)
    n = len(U_shape)
    #############
    X,Y = np.meshgrid(*[np.linspace(0,1)]*2)
_s = np.stack([X.ravel(), Y.ravel()]).T
    f_k = np.array(list(map(_ff['f_k'], _s))).reshape(X.shape)
    plt.contourf(X, Y, f_k)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, f_k, 100) #, 50, cmap='binary'


""" Constants """
def h_k(new_k, U_shape):
    U_bounds = [[0, U_bound] for U_bound in U_shape]
    integrand = lambda *x : np.prod(np.cos(np.array(x)*new_k)**2)
    integral_result, _ = integrate.nquad(integrand, U_bounds)
    return np.sqrt(integral_result)

def lambda_k(new_k, n):
    s = (n+1)/2
    lambda_k = 1 / (1 + np.linalg.norm(new_k)**2)**s
    return lambda_k

def new_k(k, U_shape):
    return np.array(k)*np.pi/np.array(U_shape)

def Constants(K, U_shape):
    n = len(U_shape)
    constants = {}
    for k in np.ndindex(*[K]*n):
        constants[k] = {}
        constants[k]['new_k'] = new_k(k, U_shape)
        constants[k]['h_k'] = h_k(constants[k]['new_k'], U_shape)
        constants[k]['lambda_k'] = lambda_k(constants[k]['new_k'], n)
    return constants

""" Mu """
def mu_k(mu, fourier_k, U_shape):
    U_bounds = [[0, U_bound] for U_bound in U_shape]
    # is mu defined everywhere in the bounds
    integrand = lambda *x: mu(np.array(x)) * fourier_k(x).numpy()
    integral_result, _ = integrate.nquad(integrand, U_bounds)
    return integral_result

def Mu(mu, K, U_shape, Torch_Functions):
    n = len(U_shape)
    _mu = {}
    for k in np.ndindex(*[K]*n):
        _mu[k] = {'mu_k': mu_k(mu, Torch_Functions[k]['f_k'], U_shape)}
    return _mu

""" Torch """
def torch_fourier_k(new_k, U_shape):
    return lambda x : (1/h_k(new_k, U_shape)) * torch.prod(torch.cos(torch.tensor(x)*torch.tensor(new_k)))

def torch_dfourier_k(new_k, U_shape):
    f_k = torch_fourier_k(new_k, U_shape)
    return lambda x : torch.autograd.functional.jacobian(f_k, x)

def torch_fourier_functions_k(k, U_shape):
    n = len(U_shape)
    new_k = np.array(k)*np.pi/np.array(U_shape)
    fourier_k = torch_fourier_k(new_k, U_shape)
    dfourier_k = torch_dfourier_k(new_k, U_shape) 
    return {'f_k': fourier_k, 
            'df_k': dfourier_k}

def Torch_Fourier_Functions(K, U_shape):
    n = len(U_shape)
    torch_ff = {}
    for k in np.ndindex(*[K]*n):
        torch_ff[k] = torch_fourier_functions_k(k, U_shape)
    return torch_ff

""" Casadi """
def casadi_prod(x, n):
    result = 1
    for i in range(n):
        result *= x[i] 
    return result

def casadi_fourier_k(new_k, U_shape):
    hk = h_k(new_k, U_shape)
    return lambda x : (1/hk) * casadi_prod(casadi.cos(x*casadi.MX(new_k)), len(U_shape))

def casadi_fourier_functions_k(k, U_shape):
    n = len(U_shape)
    new_k = np.array(k)*np.pi/np.array(U_shape)
    fourier_k = casadi_fourier_k(new_k, U_shape)
    return {'casadi_f_k': fourier_k}

def Casadi_Fourier_Functions(K, U_shape):
    n = len(U_shape)
    casadi_ff = {}
    for k in np.ndindex(*[K]*n):
        casadi_ff[k] = casadi_fourier_functions_k(k, U_shape)
    return casadi_ff