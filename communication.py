from ergodic_agents import AgentSystem

class GlobalBroadcastingSystem(AgentSystem):
    def communicate(self):
        system_c_k = {}
        for k in self.all_k_bands:
            system_c_k[k] = 0
            for agent in self.agents:
                system_c_k[k] += agent.c_k[k]
            system_c_k[k] = system_c_k[k]/self.num_agents
        
        for agent in self.agents:
            agent.system_c_k = system_c_k
            agent.system_c_k_log.append(system_c_k)


class LocalAveragingSystem(AgentSystem):
    def communicate(self):
        pass 

class LocalStaticConsensusSystem(AgentSystem):
    def communicate(self):
        pass 

class LocalDynamicConsensusSystem(AgentSystem):
    def communicate(self):
        pass 