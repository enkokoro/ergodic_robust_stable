import numpy as np

from ergodic_agents import Agent, calculate_ergodicity
import casadi

class CasadiAgent(Agent):
    """ first order agent """
    def control(self, t, delta_t, c_k_prev=None, x_prev=None, time_horizon=20):
        """ returns new control """
        if self.time_horizon is not None:
            time_horizon = self.time_horizon
        if c_k_prev is None:
            if self.system_c_k is None:
                c_k_prev = self.c_k_log[-1]
            else:
                c_k_prev = self.system_c_k
        c_opti = casadi.Opti()
        self.c_opti = c_opti
        u = c_opti.variable(self.n, time_horizon)

        c_opti.set_initial(u, np.zeros((self.n, time_horizon)))
        # check to make sure new move is within bounds
        x_prev = self.x_log[-1].reshape(-1, 1)
        c_k_prev = self.c_k_log[-1]
        _t = t
        for i in range(time_horizon):
            x_curr = self.move(u[:,i], _t, delta_t, x_prev=x_prev)
            c_opti.subject_to( x_curr >= 0 )
            c_opti.subject_to( x_curr <= np.array(self.U_shape) )

            c_opti.subject_to(casadi.sum1(u[:,i]*u[:,i]) <= self.max_control**2) # new
        
            c_k_curr = self.recalculate_c_k(_t, delta_t, c_k_prev=c_k_prev, x_prev=x_prev, x_curr=x_curr)
            x_prev = x_curr 
            c_k_prev = c_k_curr
            _t += delta_t
            
        c_opti.minimize(calculate_ergodicity(self.k_bands, c_k_curr, self.ff))

        p_opts = {}
        s_opts = {'print_level': 0}
        c_opti.solver('ipopt', p_opts, s_opts)
        sol = c_opti.solve() 
        return sol.value(u)

    def move(self, u, t, delta_t, x_prev=None):
        """ returns new movement given u control"""
        if x_prev is None:
            x_prev = self.x_log[-1]
        new_x = x_prev + u*delta_t
        return new_x