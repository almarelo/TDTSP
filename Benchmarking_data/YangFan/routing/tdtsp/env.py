from collections import deque
from typing import Optional, Iterable
import copy

import torch
import numpy as np
import math

from tensordict.tensordict import TensorDict
from torchrl.data import Bounded, Composite, Unbounded

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.utils.ops import gather_by_index, get_distance, get_tour_length
from rl4co.utils.pylogger import get_pylogger

from .matrix import TimeDependentAdjacentMatrix
from .generator import TDTSPGenerator
from .render import render

log = get_pylogger(__name__)


class TDTSPEnv(RL4COEnvBase):
    """time-dependent Travelling Salesman Problem (TDTSP) environment.
    At each step, the agent chooses a city to visit. The reward is 0 unless the agent visits all the cities.
    In that case, the reward is (-)length of the path: maximizing the reward is equivalent to minimizing the path length.

    Observations:
        - time adjacent matrix.
        - the current location of the vehicle.
        - the current time.

    Constraints:
        - the tour must return to the starting customer.
        - each customer must be visited exactly once.

    Finish condition:
        - the agent has visited all customers and returned to the starting customer.

    Reward:
        - (minus) The time of the tour.

    Args:
        generator: TSPGenerator instance as the data generator
    """


    def __init__(
        self,
        matrix: TimeDependentAdjacentMatrix = None,
        generator: TDTSPGenerator = None,
        generator_params: dict = {},
        time_matrix_params: dict = {},
        name: str = "tdtsp",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.matrix = matrix if matrix is not None else TimeDependentAdjacentMatrix(**time_matrix_params)
        if generator is None:
            self.generator = TDTSPGenerator(self.matrix, **generator_params)
        self._make_spec(self.generator)
        self.name = name
        self.pomo = kwargs.get('pomo', False)

    def _step(self, td: TensorDict) -> TensorDict:
        reward = -self.matrix.get_distance(td['current_node'], td['action'], td['time'])

        current_time = td["time"] + self.get_distance(td['current_node'], td['action'], td['time'])

        current_node = td["action"]
        first_node = current_node if 'first_node' not in td else td["first_node"]
        assert torch.max(current_node) <= td["locs"].shape[1] - 1, "The action must be in the range of locs"

        available = td['action_mask'].scatter(
            -1, current_node.unsqueeze(-1).expand_as(td['action_mask']), 0
        )

        done = torch.sum(available, dim=-1) == 0

        td.update(
            {
                "first_node": first_node,
                "current_node": current_node,
                "action_mask": available,
                "time": current_time,
                "reward": reward,
                "done": done,
                "i": td["i"] + 1,
            }
        )
        return td

    def _reset(self, td: Optional[TensorDict] = None, batch_size=None) -> TensorDict:
        init_locs = td["locs"] if td is not None and 'locs' in td else None
        if batch_size is None:
            batch_size = self.batch_size if init_locs is None else init_locs.shape[:-2]
        device = init_locs.device if init_locs is not None else self.device
        self.to(device)
        self.matrix.to(device)
        if init_locs is None:
            init_locs = self.generator._generate(batch_size=batch_size)["locs"]
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size

        num_loc = init_locs.shape[1]
        self.matrices = self.matrix.select(init_locs)  # shape: [batch_size, num_loc, num_loc, horizon]
        init_locs.to(device)  # shape: [batch_size, num_loc, 1]
        batch_indices = torch.arange(batch_size[0], device=device)
        current_node = torch.zeros(batch_size, dtype=torch.int64, device=device)
        available = torch.ones((*batch_size, num_loc), dtype=torch.bool, device=device)
        if not self.pomo:
            available[batch_indices, current_node] = False  # current node is visited
        start_time = (td['start_time'] if td is not None and 'start_time' in td else
                      torch.ones(*batch_size, dtype=torch.float32, device=device) * 0)
        i = torch.zeros((*batch_size, 1), dtype=torch.float32, device=device)

        return TensorDict(
            {
                "adj": self.matrices,
                "locs": init_locs,
                "first_node": current_node.clone(),
                "current_node": current_node,
                "start_time": start_time,
                "time": start_time,
                "i": i,
                "action_mask": available,
                "reward": torch.zeros((*batch_size, 1), dtype=torch.float32),
                "done": torch.sum(available, dim=-1) == 0,
            },
            batch_size=batch_size,
        )

    def _make_spec(self, generator: TDTSPGenerator):
        self.observation_spec = Composite(
            adj=Unbounded(
                shape=(generator.num_loc, generator.num_loc, generator.horizon),
                dtype=torch.float32,
            ),
            locs=Unbounded(
                shape=(generator.num_loc, 1),
                dtype=torch.int64,
            ),
            first_node=Unbounded(
                shape=(1),
                dtype=torch.int64,
            ),
            current_node=Unbounded(
                shape=(1),
                dtype=torch.int64,
            ),
            i=Unbounded(
                shape=(1),
                dtype=torch.float32,
            ),
            start_time=Unbounded(
                shape=(1),
                dtype=torch.float32,
            ),
            time=Unbounded(
                shape=(1),
                dtype=torch.float32,
            ),
            action_mask=Unbounded(
                shape=(generator.num_loc),
                dtype=torch.bool,
            ),
            shape=(),
        )
        self.action_spec = Bounded(
            shape=(1),
            dtype=torch.int64,
            low=0,
            high=generator.num_loc,
        )
        self.reward_spec = Unbounded(shape=(1))
        self.done_spec = Unbounded(shape=(1), dtype=torch.bool)

    def _get_reward(self, td: TensorDict, actions: torch.Tensor) -> TensorDict:
        if self.check_solution:
            self.check_solution_validity(td, actions)
        if actions.shape[-1] < td['adj'].shape[1]:
            actions = torch.cat([td["first_node"].unsqueeze(1), actions], dim=1)
        locs_ordered = gather_by_index(td["locs"], actions)
        return -self.matrix.get_tour_length(td['start_time'], locs_ordered)

    def get_tour_length(self, td, tours):
        locs = gather_by_index(td["locs"], tours)
        return self.matrix.get_tour_length(td['start_time'], locs)

    def get_path_length(self, td, paths):
        locs = gather_by_index(td["locs"], paths)
        return self.matrix.get_path_length(td['start_time'], locs)

    @staticmethod
    def check_solution_validity(td: TensorDict, actions: torch.Tensor) -> None:
        if actions.shape[-1] == td['adj'].shape[1]:
            assert (
                    torch.arange(actions.size(1), out=actions.data.new())
                    .view(1, -1)
                    .expand_as(actions)
                    == actions.data.sort(1)[0]
            ).all(), "Invalid tour"
        elif actions.shape[-1] == td['adj'].shape[1] - 1:
            assert (
                    torch.arange(1, actions.size(1) + 1, out=actions.data.new())
                    .view(1, -1)
                    .expand_as(actions)
                    == actions.data.sort(1)[0]
            ).all(), "Invalid tour"
        else:
            raise ValueError(f"Invalid action shape: {actions.shape}, expected {td['adj'].shape[1]} or {td['adj'].shape[1] - 1}")

    def replace_selected_actions(
        self,
        cur_actions: torch.Tensor,
        new_actions: torch.Tensor,
        selection_mask: torch.Tensor,
    ) -> torch.Tensor:
        cur_actions[selection_mask] = new_actions[selection_mask]
        return cur_actions

    def render(self, td: TensorDict, actions: torch.Tensor = None, ax=None):
        return render(td, actions, ax)

    def get_distances(self, locs, time):  # get all pair distances at a specific time: (batch_size, num_loc, num_loc)
        return self.matrix.get_distances(locs, locs, time)

    def get_distance(self, loc1, loc2, time):  # get pairwise distance at a specific time: (batch_size,)
        return self.matrix.get_distance(loc1, loc2, time)


class TDTSPEnvForPomo(TDTSPEnv):
    def __init__(
        self,
        matrix: TimeDependentAdjacentMatrix = None,
        generator: TDTSPGenerator = None,
        generator_params: dict = {},
        time_matrix_params: dict = {},
        name: str = "tdtsp",
        **kwargs
    ):
        time_matrix_params['interpolate'] = "constant"
        super().__init__(matrix, generator, generator_params, time_matrix_params, name, **kwargs)

    def copy(self, env: TDTSPEnv):
        # copy the matrix from TDTSPEnv
        self.matrix = copy.deepcopy(env.matrix)
        self.matrix.to_constant()

    def _reset(self, td: Optional[TensorDict] = None, batch_size=None) -> TensorDict:
        init_locs = td["locs"] if td is not None and 'locs' in td else None
        if batch_size is None:
            batch_size = self.batch_size if init_locs is None else init_locs.shape[:-2]
        device = init_locs.device if init_locs is not None else self.device
        self.to(device)
        self.matrix.to(device)
        if init_locs is None:
            init_locs = self.generator._generate(batch_size=batch_size)["locs"]
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size

        num_loc = init_locs.shape[1]
        self.matrices = self.matrix.select(init_locs)  # shape: [batch_size, num_loc, num_loc, horizon]
        init_locs.to(device)  # shape: [batch_size, num_loc, 1]
        current_node = torch.zeros(batch_size, dtype=torch.int64, device=device)
        available = torch.ones((*batch_size, num_loc), dtype=torch.bool, device=device)
        start_time = (td['start_time'] if td is not None and 'start_time' in td else
                      torch.ones(*batch_size, dtype=torch.float32, device=device) * 0)
        i = torch.zeros((*batch_size, 1), dtype=torch.float32, device=device)

        return TensorDict(
            {
                "adj": self.matrices,
                "locs": init_locs,
                "current_node": current_node,
                "start_time": start_time,
                "time": start_time,
                "i": i,
                "action_mask": available,
                "reward": torch.zeros((*batch_size, 1), dtype=torch.float32),
                "done": torch.sum(available, dim=-1) == 0,
            },
            batch_size=batch_size,
        )

    def _step(self, td: TensorDict) -> TensorDict:
        reward = -self.matrix.get_distance(td['current_node'], td['action'], td['time'])

        current_time = td["time"] + self.get_distance(td['current_node'], td['action'], td['time'])

        current_node = td["action"]
        first_node = current_node if td["i"].all() == 0 else td["first_node"]
        assert torch.max(current_node) <= td["locs"].shape[1] - 1, "The action must be in the range of locs"

        available = td['action_mask'].scatter(
            -1, current_node.unsqueeze(-1).expand_as(td['action_mask']), 0
        )

        done = torch.sum(available, dim=-1) == 0

        td.update(
            {
                "first_node": first_node,
                "current_node": current_node,
                "action_mask": available,
                "time": current_time,
                "reward": reward,
                "done": done,
                "i": td["i"] + 1,
            }
        )
        return td

    @staticmethod
    def check_solution_validity(td: TensorDict, actions: torch.Tensor) -> None:
        assert (
                torch.arange(actions.size(1), out=actions.data.new())
                .view(1, -1)
                .expand_as(actions)
                == actions.data.sort(1)[0]
        ).all(), "Invalid tour"