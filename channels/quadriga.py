from typing import Optional, Tuple

import numpy as np
import scipy.io as sio

from sixg_radio_mgmt import Channel


class QuadrigaChannels(Channel):
    def __init__(
        self,
        max_number_ues: int,
        max_number_basestations: int,
        num_available_rbs: np.ndarray,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> None:
        super().__init__(
            max_number_ues, max_number_basestations, num_available_rbs, rng
        )
        self.thermal_noise_power = 10e-14
        self.episode_number = -1
        self.episode_channels = np.empty([])

    def step(
        self,
        step_number: int,
        episode_number: int,
        mobilities: np.ndarray,
        sched_decision: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if self.episode_number != episode_number:
            self.episode_number = episode_number
            self.episode_channels = self.read_mat_files(episode_number)

        if sched_decision is not None:
            return self.calc_spectral_eff(sched_decision, step_number)
        else:
            spectral_efficiencies = [
                np.ones((self.max_number_ues, self.num_available_rbs[i]))
                for i in np.arange(self.max_number_basestations)
            ]
        return np.array(spectral_efficiencies)

    def calc_spectral_eff(
        self, sched_decision: np.ndarray, step_number: int
    ) -> np.ndarray:
        # Sum transmitter and receiver antennas, and changing dimensions to
        # B x U x R to match sched_decision dimensions. After multiplying
        # with sched_decision, we obtain the channels only in the allocated RB
        allocated_rbs_channels = (
            np.expand_dims(
                np.sum(
                    np.sum(self.episode_channels[:, :, :, :, step_number], 0),
                    0,
                ),
                0,
            )
            * sched_decision
        )

        allocated_rbs_rsrp = np.power(np.abs(allocated_rbs_channels), 2)
        spectral_efficiencies = np.log2(
            1 + (allocated_rbs_rsrp / self.thermal_noise_power)
        )
        return spectral_efficiencies

    def read_mat_files(self, episode: int) -> np.ndarray:
        channels = sio.loadmat(f"channels/quadriga_channels/sim_{episode}.mat")

        return channels["H"]
