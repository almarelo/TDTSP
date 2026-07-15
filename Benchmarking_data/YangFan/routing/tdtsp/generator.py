from typing import Callable, Union

import torch

from tensordict.tensordict import TensorDict
from torch.distributions import Uniform

from rl4co.envs.common.utils import Generator, get_sampler
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class TDTSPGenerator(Generator):
    """Data generator for the Time-Dependent Travelling Salesman Problem (TDTSP).

        Args:
            num_loc: number of locations (customers) in the TSP
            min_loc: minimum value for the location coordinates
            max_loc: maximum value for the location coordinates
            init_sol_type: the method type used for generating initial solutions (random or greedy)
            loc_distribution: distribution for the location coordinates

        Returns:
            A TensorDict with the following keys:
                locs [batch_size, num_loc, 2]: locations of each customer
        """

    def __init__(
        self,
        matrix,
        num_loc: int=20,
        init_sol_type: str="random",
        **kwargs,
    ):
        self.matrix = matrix  # shape: [num_nodes, num_nodes, horizon]
        self.num_nodes = matrix.num_nodes
        self.horizon = matrix.horizon
        self.num_loc = num_loc
        self.init_sol_type = init_sol_type

    def _generate(self, batch_size) -> TensorDict:
        # torch.manual_seed(0)
        locs = torch.concatenate([torch.randperm(self.num_nodes)[:self.num_loc].unsqueeze(0) for _ in range(*batch_size)],
                                    dim=0).unsqueeze(-1)  # shape: [batch_size, num_loc, 1]
        return TensorDict(
            {
                "locs": locs,
            },
            batch_size=batch_size,
        )

    def _get_initial_solutions(self, locs):
        batch_size = locs.size(0)

        if self.init_sol_type == "random":
            set = torch.rand(batch_size, self.num_loc).argsort().long()
            rec = torch.zeros(batch_size, self.num_loc).long()
            index = torch.zeros(batch_size, 1).long()

            for i in range(self.num_loc - 1):
                rec.scatter_(1, set.gather(1, index + i), set.gather(1, index + i + 1))

            rec.scatter_(1, set[:, -1].view(-1, 1), set.gather(1, index))
        else:
            raise NotImplementedError()

        return rec.expand(batch_size, self.num_loc).clone()
