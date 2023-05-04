import numpy as np
from stable_baselines3.common.env_checker import check_env
from tqdm import tqdm

from agents.round_robin import RoundRobin
from agents.ssr_protect import SSRProtect
from agents.ssr_rl import SSRRL
from associations.industrial import IndustrialAssociation
from channels.quadriga import QuadrigaChannels
from mobilities.simple import SimpleMobility
from sixg_radio_mgmt import CommunicationEnv
from traffics.industrial import IndustrialTraffic

scenarios = ["industrial"]
agents = ["ssr_protect"]  # ["ssr", "ssr_protect"]
agents_rl = ["ssr_project"]  # ["ssr", "ssr_protect"]
seed = 10
for scenario in scenarios:
    for agent_name in agents:
        comm_env = CommunicationEnv(
            QuadrigaChannels,
            IndustrialTraffic,
            SimpleMobility,
            IndustrialAssociation,
            scenario,
            agent_name,
            obs_space=SSRRL.get_obs_space if agent_name in agents else None,
            action_space=SSRRL.get_action_space,
        )

        match agent_name:
            case "ssr":
                AgentClass = SSRRL
            case "ssr_protect":
                AgentClass = SSRProtect
            case _:
                raise Exception("Agent not implemented")

        agent = AgentClass(
            comm_env,
            comm_env.max_number_ues,
            comm_env.max_number_basestations,
            comm_env.num_available_rbs,
            seed=seed,
        )
        comm_env.set_agent_functions(
            agent.obs_space_format,
            agent.action_format,
            agent.calculate_reward,
        )

        # check_env(comm_env)
        train_episodes = 99
        steps_per_episode = 1000
        train_runs = 1
        total_number_steps = train_episodes * steps_per_episode * train_runs
        print("########### TRAIN ###########")
        agent.train(total_number_steps)

        # Test
        print("########### TEST ###########")
        comm_env.max_number_episodes = 100
        obs = comm_env.reset(seed=seed, options={"initial_episode": 99})[0]
        for step_number in tqdm(np.arange(comm_env.max_number_steps)):
            sched_decision = agent.step(obs)
            obs, _, end_ep, _, _ = comm_env.step(sched_decision)
            if end_ep and (step_number + 1) < total_number_steps:
                comm_env.reset()
