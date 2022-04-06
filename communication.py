from ergodic_agents import AgentSystem
import numpy as np 

class GlobalBroadcastingSystem(AgentSystem):
    def __init__(self, *args):
        super().__init__(*args)
        self.system_type = "Global Broadcasting"

    def communicate(self, t, delta_t):
        system_c_k = {}
        for k in self.all_k_bands:
            system_c_k[k] = 0
            for agent in self.agents:
                system_c_k[k] += agent.c_k[k]
            system_c_k[k] = system_c_k[k]/self.num_agents
        
        for agent in self.agents:
            agent.system_c_k = system_c_k
            agent.system_c_k_log.append(system_c_k)

class LocalCommunicationSystem(AgentSystem):
    def set_communication_range(self, comm_range):
        assert comm_range > 0
        self.comm_range = comm_range 
    
    def set_communication_timestep(self, comm_timestep):
        assert comm_timestep > 0
        self.comm_timestep = comm_timestep

class LocalAveragingSystem(LocalCommunicationSystem):
    def __init__(self, *args):
        super().__init__(*args)
        self.system_type = "Local Averaging"

    def communicate(self, t, delta_t):
        for agent in self.agents:
            agent.system_c_k = {}
            for k in self.all_k_bands:
                agent.system_c_k[k] = 0
                num_neighbors = 0
                for neighbor in self.agents:
                    if np.linalg.norm(agent.x_log[-1] - neighbor.x_log[-1]) < self.comm_range:
                        agent.system_c_k[k] += neighbor.c_k[k]
                        num_neighbors += 1
                agent.system_c_k[k] /= num_neighbors
            agent.system_c_k_log.append(agent.system_c_k)
                

class LocalStaticConsensusSystem(LocalCommunicationSystem):
    def __init__(self, *args):
        super().__init__(*args)
        self.system_type = "Local Static Consensus"

    def communicate(self, t, delta_t):
        for agent in self.agents:
            agent.system_c_k = agent.system_c_k_log[-1].copy()
            for k in self.all_k_bands:
                for neighbor in self.agents:
                    if np.linalg.norm(agent.x_log[-1] - neighbor.x_log[-1]) < self.comm_range:
                        agent.system_c_k[k] -= self.comm_timestep*(agent.system_c_k_log[-1][k] - neighbor.c_k[k])
            agent.system_c_k_log.append(agent.system_c_k)

class LocalDynamicConsensusSystem(LocalCommunicationSystem):
    def __init__(self, *args):
        super().__init__(*args)
        self.system_type = "Local Dynamic Consensus"

    def communicate(self, t, delta_t):
        for agent in self.agents:
            agent.system_c_k = agent.system_c_k_log[-1].copy()
            for k in self.all_k_bands:
                for neighbor in self.agents:
                    if np.linalg.norm(agent.x_log[-1] - neighbor.x_log[-1]) < self.comm_range:
                        du_i_dt = (self.ff[k](agent.x_log[-1])*t - self.agent_c_k[-1]*t)/t**2
                        agent.system_c_k[k] -= self.comm_timestep*(agent.system_c_k_log[-1][k] - neighbor.c_k[k]) + du_i_dt
            agent.system_c_k_log.append(agent.system_c_k)