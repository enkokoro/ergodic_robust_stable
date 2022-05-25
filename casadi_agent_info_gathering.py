import numpy as np

from ergodic_agents import Agent
import casadi

def dirac_delta_1D(x, a=1):
    return a*(1-np.tanh(a*x)**2)/2

def dirac_delta_2D(x, a=1):
    # return np.prod(dirac_delta_1D(x, a=a))
    return dirac_delta_1D(x[0], a=a)*dirac_delta_1D(x[1], a=a)

def info_gathering_metric(mu, trajectory, T, a):
    I = 1/T * sum([mu(casadi.reshape(x, (2,1))) for x in trajectory[:T]])
    C = 1/T**2 * sum([dirac_delta_2D(x-y, a=a) for y in trajectory[:T] for x in trajectory[:T]])
    print("I:", I.shape)
    print("C:", C.shape)
    return 2*I - C

class CasadiAgentInfoGathering(Agent):
    """ first order agent """
    def control(self, t, delta_t, c_k_prev=None, x_prev=None, mu=None, time_horizon=100):
        """ returns new control """
        c_opti = casadi.Opti()
        self.c_opti = c_opti
        u = c_opti.variable(self.n, time_horizon)
        # c_opti.set_initial(u, np.zeros((self.n, time_horizon)))
        # check to make sure new move is within bounds
        x_prev = self.x_log[-1].reshape(-1, 1)
        _t = t
        x = [x_prev]
        for i in range(time_horizon):
            x_curr = self.move(u[:,i], _t, delta_t, x_prev=x_prev)
            c_opti.subject_to( x_curr >= 0 )
            c_opti.subject_to( x_curr <= np.array(self.U_shape) )
            c_opti.subject_to( u[0,i]**2+u[1,i]**2 <= self.max_control**2 )

            x_prev = x_curr 
            x.append(x_prev)
            _t += delta_t
            
        c_opti.minimize(-info_gathering_metric(self.mu, x, time_horizon, 1/self.dx))

        p_opts = {}
        s_opts = {'print_level': 0}
        c_opti.solver('ipopt', p_opts, s_opts)
        sol = c_opti.solve() 
        result = sol.value(u)
        print(result)
        return result

    def move(self, u, t, delta_t, x_prev=None):
        """ returns new movement given u control"""
        if x_prev is None:
            x_prev = self.x_log[-1]
        new_x = x_prev + u*delta_t
        return new_x