from typing import Union
from itertools import product

import numpy as np
from gym import spaces
from stable_baselines3.sac.sac import SAC

from sixg_radio_mgmt import Agent, CommunicationEnv


class SSRRL(Agent):
    def __init__(
        self,
        env: CommunicationEnv,
        max_number_ues: int,
        max_number_basestations: int,
        num_available_rbs: np.ndarray,
        hyperparameters: dict = {},
        seed: int = np.random.randint(1000),
    ) -> None:
        super().__init__(
            env,
            max_number_ues,
            max_number_basestations,
            num_available_rbs,
            seed,
        )
        self.agent = SAC(
            "MlpPolicy",
            env,
            verbose=0,
            tensorboard_log="./tensorboard-logs/",
            seed=self.seed,
        )

        # Variables for round-robin scheduling
        self.current_ues = np.array([])
        self.rbs_per_ue = np.zeros(
            (self.env.max_number_slices, max_number_ues)
        )
        self.allocation_rbs = []

        # Creating possible combinations of actions
        self.action_space_options = self.create_combinations(
            self.num_available_rbs[0], self.env.max_number_slices
        )

    def step(self, obs_space: Union[np.ndarray, dict]) -> np.ndarray:
        return self.agent.predict(np.asarray(obs_space), deterministic=True)[0]

    def train(self, total_timesteps: int) -> None:
        self.agent.learn(total_timesteps=int(total_timesteps), callback=[])

    def save(self, filename: str) -> None:
        self.agent.save(filename)

    def load(self, filename: str, env: CommunicationEnv) -> None:
        self.agent = SAC.load(filename, env=env)

    @staticmethod
    def obs_space_format(obs_space: dict) -> np.ndarray:
        formatted_obs_space = np.array([])
        hist_labels = [
            # "pkt_incoming",
            "dropped_pkts",
            # "pkt_effective_thr",
            "buffer_occupancies",
            # "spectral_efficiencies",
        ]
        for hist_label in hist_labels:
            if hist_label == "spectral_efficiencies":
                formatted_obs_space = np.append(
                    formatted_obs_space,
                    np.squeeze(np.sum(obs_space[hist_label], axis=2)),
                    axis=0,
                )
            else:
                formatted_obs_space = np.append(
                    formatted_obs_space, obs_space[hist_label], axis=0
                )

        return formatted_obs_space

    @staticmethod
    def calculate_reward(obs_space: dict) -> float:
        reward = -np.sum(obs_space["dropped_pkts"], dtype=float)
        return reward

    @staticmethod
    def get_action_space() -> spaces.Box:
        return spaces.Box(low=-1, high=1, shape=(3,))

    @staticmethod
    def get_obs_space() -> spaces.Box:
        return spaces.Box(low=0, high=np.inf, shape=(2 * 2,), dtype=np.float32)

    def action_format(
        self,
        action: np.ndarray,
    ) -> np.ndarray:
        action_rbs = self.nn_output_to_rbs(action)
        sched_decision = self.round_robin(action_rbs, self.env.slices.ue_assoc)

        return sched_decision

    def round_robin(
        self,
        rbs_per_slice: np.ndarray,
        slice_ue_assoc: np.ndarray,
    ) -> np.ndarray:
        number_slices = len(rbs_per_slice)
        if np.array_equal(self.allocation_rbs, np.array([])):
            initial_rb = 0
            self.allocation_rbs = [
                np.zeros(
                    (
                        self.max_number_ues,
                        self.num_available_rbs[basestation],
                    )
                )
                for basestation in np.arange(self.max_number_basestations)
            ]
            for slice_idx in np.arange(number_slices):
                idx_active_ues = slice_ue_assoc[slice_idx].nonzero()[0]
                num_active_ues = np.sum(slice_ue_assoc[slice_idx]).astype(int)
                num_rbs_per_ue = int(
                    (
                        np.floor(rbs_per_slice[slice_idx] / num_active_ues)
                        if num_active_ues > 0
                        else 0
                    )
                )
                remaining_rbs = (
                    rbs_per_slice[slice_idx] - num_rbs_per_ue * num_active_ues
                )
                self.rbs_per_ue = np.ones(num_active_ues) * num_rbs_per_ue
                self.rbs_per_ue[:remaining_rbs] += 1

                for idx, ue_idx in enumerate(idx_active_ues):
                    self.allocation_rbs[0][
                        ue_idx,
                        initial_rb : initial_rb + int(self.rbs_per_ue[idx]),
                    ] = 1
                    initial_rb += int(self.rbs_per_ue[idx])

        return np.array(self.allocation_rbs)

    def nn_output_to_rbs(self, action) -> np.ndarray:
        rbs_allocation = (
            ((action + 1) / np.sum(action + 1)) * self.num_available_rbs[0]
            if np.sum(action + 1) != 0
            else np.ones(action.shape[0])
            * (1 / action.shape[0])
            * self.num_available_rbs[0]
        )
        action_idx = np.argmin(
            np.sum(np.abs(self.action_space_options - rbs_allocation), axis=1)
        )

        return self.action_space_options[action_idx]

    def create_combinations(self, number_slices: int, full=False):
        """
        Create the combinations of possible arrays with RBs allocation for each
        slice. For instance, let's assume 3 slices and 17 RBs available in the
        basestation, a valid array should be [1, 13, 3] since its summation is
        equal to 17 RBs. Moreover, it indicates that the first slice received 1
        RB, the second received 13 RBs, and the third received 3 RBs. A valid
        array always has a summation equal to the total number of RBs in a
        basestation and has its array-length equal to the number of slices. An
        action taken by RL agent is a discrete number that represents the index
        of the option into the array with all possible RBs allocations for
        these slices.
        """

        combinations = []
        combs = product(
            range(0, self.num_available_rbs[0] + 1), repeat=number_slices
        )
        for comb in combs:
            if np.sum(comb) == self.num_available_rbs[0]:
                combinations.append(comb)
        return np.asarray(combinations)
