import numpy as np

from ergodic_agents import Agent 


class MMAgent1(Agent):
    """ first order agent """
    def control(self, t, delta_t, c_k_prev=None, x_prev=None):
        """ returns new control """
        if x_prev is None:
            x_prev = self.x_log[-1]

        if c_k_prev is None:
            if self.system_c_k is None:
                c_k_prev = self.c_k_log[-1]
            else:
                c_k_prev = self.system_c_k

        n = len(self.U_shape)
        B_j = np.zeros(n)

        for k in self.k_bands:
            lambda_k = self.ff[k]['lambda_k']
            grad_f_k = self.ff[k]['df_k']
            mu_k = self.ff[k]['mu_k']
            c_k = c_k_prev[k]
            s_k = c_k - mu_k # S_k = N*t*s_k = N*t*c_k - N*t*mu_k
            B_j += lambda_k * s_k * grad_f_k(x_prev)
            
            assert x_prev.shape == grad_f_k(x_prev).shape
            assert x_prev.shape == B_j.shape
        # print("B_j: ", B_j)

        # B_j = 0 or agent is too near boundary (because we are using approximations, 
        #   may end up slipping outside) 
        # # or every so often do some random movement
        if (np.linalg.norm(B_j) == 0 
            or (x_prev < self.eps).any() or (np.array(self.U_shape) - x_prev < self.eps).any()):
            # or random.randrange(10) < 1:
            print("oh no at time ", t)
            
            U_center = np.array(self.U_shape)/2
            if (U_center == x_prev).all():
                # move in random direction away from center 
                assert(x_prev.size == n)
                B_j = np.zeros(n)
                r_idx = 0
                # r_idx = random.randrange(n)
                B_j[r_idx] = -1
                
            else:
                # aim towards center if not already at center
                B_j = -1 * (U_center - x_prev)
        
        u_j = -1 * self.max_control * B_j / np.linalg.norm(B_j)
        return np.array([u_j]).reshape(-1,1)

    def move(self, u, t, delta_t, x_prev=None):
        """ returns new movement given u control"""
        if x_prev is None:
            x_prev = self.x_log[-1]
        new_x = x_prev + u*delta_t
        return new_x