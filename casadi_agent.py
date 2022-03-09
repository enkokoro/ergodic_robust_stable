import numpy as np

from ergodic_agents import Agent, calculate_ergodicity
import casadi

class CasadiAgent(Agent):
    """ first order agent """
    def control(self, t, delta_t, c_k_prev=None, x_prev=None):
        """ returns new control """
        c_opti = casadi.Opti()
        self.c_opti = c_opti
        u = c_opti.variable(self.m)
        ## likely this issue, can prob easily be fixed component wise thing
        c_opti.subject_to( u <= self.umax )
        c_opti.subject_to( u >= -self.umax )
        # check to make sure new move is within bounds
        x_pred = self.move(u, t, delta_t)
        c_opti.subject_to( x_pred >= 0 )
        c_opti.subject_to( x_pred <= np.array(self.U_shape) )
        c_opti.set_initial(u, self.u_log[-1])
        
        c_k_pred = self.recalculate_c_k(t, delta_t, c_k_prev=self.c_k_log[-1], x_prev=self.x_log[-1], x_curr=x_pred):
        c_opti.minimize(calculate_ergodicity(self.k_bands, c_k_pred, self.ff))

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