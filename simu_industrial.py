import numpy as np
from tqdm import tqdm

from agents.round_robin import RoundRobin
from agents.ssr import SSR
from associations.industrial import IndustrialAssociation
from channels.quadriga import QuadrigaChannels
from mobilities.simple import SimpleMobility
from sixg_radio_mgmt import CommunicationEnv
from traffics.industrial import IndustrialTraffic

scenarios = ["industrial"]
agents = ["ssr"]
seed = 10

for scenario in scenarios:
    for agent_name in agents:
        rng = (
            np.random.default_rng(seed)
            if seed != -1
            else np.random.default_rng()
        )
        comm_env = CommunicationEnv(
            QuadrigaChannels,
            IndustrialTraffic,
            SimpleMobility,
            IndustrialAssociation,
            scenario,
            agent_name,
            rng=rng,
        )

        match agent_name:
            case "round_robin":
                AgentClass = RoundRobin
            case "ssr":
                AgentClass = SSR
            case _:
                raise Exception("Agent not implemented")

        agent = AgentClass(
            comm_env,
            comm_env.max_number_ues,
            comm_env.max_number_basestations,
            comm_env.num_available_rbs,
        )
        comm_env.set_agent_functions(
            agent.obs_space_format,
            agent.action_format,
            agent.calculate_reward,
        )
        obs = comm_env.reset()
        number_steps = 1000
        for step_number in tqdm(np.arange(comm_env.max_number_steps)):
            sched_decision = agent.step(obs)
            obs, _, end_ep, _ = comm_env.step(sched_decision)
            if end_ep and (step_number + 1) < number_steps:
                comm_env.reset()
