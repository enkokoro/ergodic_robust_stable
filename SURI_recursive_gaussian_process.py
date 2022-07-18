"""
FUNCTION REGRESSION SPECIFIC INPUTS
    technically any function regression model works as long as it obeys the following
    
    model.fit(data_tr, val_tr)
    val_te, std_te = model.predict(data_te, return_std=True)
"""
import numpy as np
import sklearn.gaussian_process as gp

class FunctionRegressionModel():
    def fit(self, data_tr, val_tr):
        raise NotImplementedError
    def predict(self, data_te, return_std=False):
        raise NotImplementedError

class RecursiveGP(FunctionRegressionModel):
    def __init__(self, mean, kernel, sigma, basis_vectors):
        self.m = mean
        self.k = kernel
        self.sigma = sigma
        # init
        self.X = basis_vectors
        self.mean_t = self.m(self.X) 
        self.cov_t = self.k(self.X, self.X) 

    def predict(self, data_te, return_std=False):
        X_t = data_te
        X = self.X
        J_t = self.k(X_t, X)@np.linalg.inv(self.k(X, X))
        mu_t_p = self.m(X_t) + J_t@(self.mean_t - self.m(X))
        C_t_p = self.k(X_t, X_t) + J_t@(self.cov_t - self.k(X,X))@J_t.T 

        if return_std:
            return mu_t_p, np.diag(C_t_p)
        else:
            return mu_t_p
        
    def fit(self, data_tr, val_tr):
        X_t = data_tr 
        y_t = val_tr
        X = self.X
        J_t = self.k(X_t, X)@np.linalg.inv(self.k(X, X))
        mu_t_p = self.m(X_t) + J_t@(self.mean_t - self.m(X))
        C_t_p = self.k(X_t, X_t) + J_t@(self.cov_t - self.k(X,X))@J_t.T 
        G_t = self.cov_t@J_t.T@np.linalg.inv(C_t_p + self.sigma**2)
        mu_t_g = self.mean_t + G_t@(y_t - mu_t_p)
        C_t_g = self.cov_t - G_t@J_t@self.cov_t

        self.mean_t = mu_t_g 
        self.cov_T = C_t_g


# testing
def test():
    basis_vectors = []
    for x in [0,1]:
        for y in [0,1]:
            for z in [0,1]:
                basis_vectors.append(np.array([x,y,z]))
    basis_vectors = np.array(basis_vectors)

    kernel = gp.kernels.ConstantKernel(1.0, (1e-1, 1e3)) * gp.kernels.RBF(10.0, (1e-3, 1e3))
    model = RecursiveGP(lambda x: np.zeros(x.shape[0]), kernel, 1.0, basis_vectors)
    def measurement(data):
        return np.max(data, axis=1)

    x_tr, y_tr, z_tr = [0, 0.25, 0.5, 0, 0, 0, 0, 1],[0, 0, 0, 0.25, 0.5, 0, 0, 1],[0, 0, 0, 0, 0, 0.25, 0.5, 1]# cube_point_cloud(5)
    data_tr = np.column_stack([x_tr, y_tr, z_tr]) # (n_points, 3)
    val_tr = measurement(data_tr)

    print(model.fit(data_tr, val_tr))
    print(model.predict(data_tr[:3], return_std=True))

# test()

