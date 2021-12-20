import casadi
import torch 
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import time

from fourier_functions import Fourier_Functions, Mu



def calculate_ergodicity(k_bands, c_k, fourier_functions):
    e = 0
    for k in k_bands:
        lambda_k = fourier_functions[k]['lambda_k']
        mu_k = fourier_functions[k]['mu_k']
        e += lambda_k*(c_k[k]-mu_k)**2
    return e

def calculate_ergodicity1(k_bands, c_k, fourier_functions):
    e = 0
    for k in k_bands:
        lambda_k = fourier_functions[k]['lambda_k']
        mu_k = fourier_functions[k]['mu_k']
        e += lambda_k*abs(c_k[k]-mu_k)
    return e

def get_idx(l, idx):
    if l is None:
        return None 
    else:
        return l[idx]

class Agent:
    def __init__(self, agent_id, init_pos, max_control, k_bands, U_shape, fourier_functions, eps=1e-5): 
        """
        agent_id : number identifier, used in adjacency matrix
        init_pos : initial position, dim n, numpy
        U_shape : movement space
        max_control : max control
        k_bands : the bands it listens to
        eps : boundary distances
        mm_order : 1 for first order (default), 2 for second order
        """
        # fourier functions
        self.ff = fourier_functions

        # agent identifiers and system specifications
        self.agent_id = agent_id 

        # agent specificifications
        self.max_control = max_control 
        self.k_bands = k_bands

        # search area specifications
        self.U_shape = U_shape
        self.n = len(U_shape)
        self.eps = eps # boundary distance

        # agent position
        self.x_log = [init_pos] # position log
        self.u_log = [np.zeros(len(U_shape))] # control log
        # time averaged spatial distribution fourier coefficients
        self.c_k = {k : self.ff[k]['f_k'](init_pos) for k in k_bands} 
        self.c_k_log = [self.c_k]

        # system belief
        self.system_c_k = None
        self.system_c_k_log = []

        # agent ergodicity
        e_init = calculate_ergodicity(self.k_bands, self.c_k, self.ff)
        self.e_log = [e_init] 
        
    # COMMON FOR ALL AGENTS USING ERGODIC COVERAGE
    def recalculate_c_k(self, t, delta_t, c_k_prev=None, x_prev=None, x_curr=None):
        """ 
        if using for prediction, supply x_prev and x_curr as casadi.MX variables, will not change actual c_k
        """
        prediction_mode = False
        if c_k_prev is not None and x_prev is not None and x_curr is not None:
            # they must be casadi.MX variables
            prediction_mode = True
            c_k_curr = {}
        else:
            x_prev = self.x_log[-2]
            x_curr = self.x_log[-1]
            c_k_curr = self.c_k # actually updates self.c_k
            c_k_prev = self.c_k

        for k in self.k_bands:
            # trapezoidal rule
            if prediction_mode:
                fourier_fn = self.ff[k]['casadi_f_k']
            else:
                fourier_fn = self.ff[k]['f_k']
            average_f_k = (1/2)*(fourier_fn(x_prev) + fourier_fn(x_curr))
            c_k_curr[k] = (c_k_prev[k]*t + average_f_k*delta_t)/(t+delta_t)

        return c_k_curr
    
    def apply_dynamics(self, t, delta_t, u=None, c_k_prev=None, x_prev=None):
        """ 
        if using for prediction, supply prev_x and curr_x as casadi.MX variables, will not change actual c_k
        """
        prediction_mode = False
        if c_k_prev is not None and x_prev is not None:
            prediction_mode = True
        if u is None: # may or may not already supply control to be used
            u = self.control(t, delta_t, c_k_prev=c_k_prev, x_prev=x_prev)
        x_curr = self.move(u, t, delta_t, x_prev=x_prev)
        if not prediction_mode:
            self.u_log.append(u)
            self.x_log.append(x_curr)

        c_k_curr = self.recalculate_c_k(t, delta_t, c_k_prev=c_k_prev, x_prev=x_prev, x_curr=x_curr)
        e = calculate_ergodicity(self.k_bands, c_k_curr, self.ff)
        if not prediction_mode:
            self.e_log.append(e)
        return u, c_k_curr, x_curr, e
    
    # OBTAIN AGENT POSITION AND ERGODICITY INFORMATION
    def get_position_log(self):
        return np.stack(self.x_log)

    def get_ergodicity_log(self):
        return np.array(self.e_log)
    
    def get_c_k_log(self):
        return self.c_k_log

    def get_c_k(self, k):
        return self.c_k.get(k, 0)

    # TO DEFINE FOR VARIOUS TYPES OF AGENTS

    # def get_position(self, x): 
    #     # redefine if using "position" as something else (position, velocity)
    #     # not integrated yet
    #     return x 

    def control(self, t, delta_t, c_k_prev=None, x_prev=None):
        """ returns new control """
        pass 

    def move(self, u, t, delta_t, x_prev=None):
        """ returns new movement given u control"""
        pass


class AgentSystem:
    def __init__(self, agents, init_mu, U_shape, fourier_functions, K): 
        # fourier functions
        self.ff = fourier_functions
        self.K = K
        self.all_k_bands = list(np.ndindex(*[K]*len(U_shape)))

        # area details
        self.U_shape = U_shape
        self.n = len(self.U_shape)
        self.mu = init_mu

        # agents
        self.num_agents = len(agents)
        self.agents = agents

        # total ergodicity
        self.c_k = self.calculate_c_k()
        self.c_k_log = [self.c_k]
        self.e_log = [calculate_ergodicity(self.all_k_bands, self.c_k, self.ff)]
    
    def get_c_k_log(self):
        return self.c_k_log

    def get_ergodicity_log(self):
        return self.e_log

    def calculate_c_k(self):
        c_k = {}
        for k in self.all_k_bands:
            agents_c_k = [agent.get_c_k(k) for agent in self.agents]
            c_k[k] = (1/self.num_agents)*sum(agents_c_k)
        return c_k

    def evolve(self, t, delta_t, u_agents=None, c_k_agents_prev=None, x_agents_prev=None):
        print(self)
        print(t)
        print(delta_t)
        print(u_agents)
        print(c_k_agents_prev)
        print(x_agents_prev)
        self.communicate()
        prediction_mode = False 
        if c_k_agents_prev is not None and x_agents_prev is not None:
            prediction_mode = True
        u_agents_applied = []
        c_k_agents_curr = []
        x_agents_curr = []
        for agent in self.agents:
            id = agent.agent_id
            u = get_idx(u_agents, id)
            c_k_prev = get_idx(c_k_agents_prev, id)
            x_prev = get_idx(x_agents_prev, id)
            u, c_k_curr, x_curr, e = agent.apply_dynamics(t, delta_t, u=u, c_k_prev=c_k_prev, x_prev=x_prev)
            
            u_agents_applied.append(u)
            c_k_agents_curr.append(c_k_curr)
            x_agents_curr.append(x_curr)

        self.c_k = self.calculate_c_k()
        ergodicity_agents = calculate_ergodicity(self.all_k_bands, self.c_k, self.ff)
        if not prediction_mode:
            self.c_k_log.append(self.c_k)
            self.e_log.append(ergodicity_agents)

        return u_agents_applied, c_k_agents_curr, x_agents_curr, ergodicity_agents

    def c_k2distribution(self, c_k, k_bands):
        def dist(x):
            res = 0
            for k in k_bands:
                res += c_k[k]*self.ff[k]['f_k'](x)
            return res
        return dist
            
    # TO DEFINE FOR VARIOUS TYPES OF SYSTEMS
    def update_positions(self):
        pass 

    def update_mu(self, new_mu, new_mu_k):
        self.mu = new_mu
        for k in self.all_k_bands:
            self.ff[k].update(new_mu_k[k])

    def communicate(self):
        pass
    
    def visualize2d(self, filename="test", additional_title="TEST", plot_c_k=False): 
        date_and_time = time.strftime("_%Y_%m_%d-%H:%M")
        filename = filename + date_and_time

        # colors = ['r', 'm', 'c', 'y', 'g', 'b', 'k', 'w']
        colors = ['maroon', 'cyan', 'red', 'black', 'slateblue', 'orange', 'indigo', 'magenta', 'pink', 'white']
        assert self.num_agents <= len(colors), "does not support this many agents"

        if plot_c_k:
            fig, ((ax1, ax3, ax2)) = plt.subplots(1, 3, figsize=(6.4*2, 4.8))
        else:
            fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(6.4*2, 4.8))

        fig.suptitle("Ergodic Coverage " + additional_title)

        ax1.set_title('Search Space')
        ax1.set_aspect('equal')
        ax1.set_xlim(0, self.U_shape[0])
        ax1.set_ylim(0, self.U_shape[1])

        X,Y = np.meshgrid(np.linspace(0, self.U_shape[0]), np.linspace(0, self.U_shape[1]))
        _s = np.stack([X.ravel(), Y.ravel()]).T
        ax1.contourf(X, Y, np.array(list(map(self.mu, _s))).reshape(X.shape))

        ax2.set_title('Ergodicity')
        ax2.set(xlabel='Time')

        if plot_c_k:
            ax3.set_title('Coverage')
            ax3.set_aspect('equal')
            ax3.set_xlim(0, self.U_shape[0])
            ax3.set_xlim(0, self.U_shape[1])

        fig.tight_layout()

        pos_data = [([], []) for i in range(self.num_agents)]
        time_data = []
        ergodicity_data = []
        local_ergodicity_data = [[] for i in range(self.num_agents)]
        pos_lns = []
        local_erg_lns = []

        for i in range(self.num_agents):
            pos_ln, = ax1.plot(pos_data[i][0], pos_data[i][1], c=colors[i], label=i)
            pos_lns.append(pos_ln)

            local_erg, = ax2.plot(time_data, local_ergodicity_data[i], c=colors[i], label=i)
            local_erg_lns.append(local_erg)
        ergodicity_ln, = ax2.plot(time_data, ergodicity_data, c='b')

        def animate2d_init():
            phi2_max = max(self.e_log)
            time_max = len(self.e_log)
            ax2.set_xlim(0, time_max)
            ax2.set_ylim(0, phi2_max)
            init_spatial_dist = self.c_k2distribution(self.c_k_log[0], self.all_k_bands)
            if plot_c_k:
                cont = ax3.contourf(X, Y, np.array(list(map(init_spatial_dist, _s))).reshape(X.shape))
                return (*cont.collections, ergodicity_ln, *local_erg_lns, *pos_lns)
            else:
                return (ergodicity_ln, *local_erg_lns, *pos_lns)


        def animate2d_from_logs_update(frame):
            time_data.append(frame)
            for i in range(self.num_agents):
                pos_data[i][0].append(self.agents[i].x_log[frame][0])
                pos_data[i][1].append(self.agents[i].x_log[frame][1])
                pos_lns[i].set_data(pos_data[i][0], pos_data[i][1])

                local_ergodicity_data[i].append(self.agents[i].e_log[frame])
                local_erg_lns[i].set_data(time_data, local_ergodicity_data[i])
                
            
            ergodicity_data.append(self.e_log[frame])
            ergodicity_ln.set_data(time_data, ergodicity_data)

            spatial_dist = self.c_k2distribution(self.c_k_log[frame], self.all_k_bands)
            if plot_c_k:
                cont = ax3.contourf(X, Y, np.array(list(map(spatial_dist, _s))).reshape(X.shape))
                return (*cont.collections, ergodicity_ln, *local_erg_lns, *pos_lns)
            else:
                return (ergodicity_ln, *local_erg_lns, *pos_lns)

        update = animate2d_from_logs_update
        frames = len(self.e_log)


        FFwriter = animation.writers['ffmpeg']
        writer = FFwriter(fps=30, metadata=dict(artist='Me'), bitrate=1800)
        anime = animation.FuncAnimation(fig, animate2d_from_logs_update, init_func=animate2d_init, 
                                    frames=frames, interval=20, blit=True)  
        plt.show()
        if filename is not None:
            anime.save(filename+".mp4", writer=writer) 
