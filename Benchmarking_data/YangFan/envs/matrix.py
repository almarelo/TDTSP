import numpy as np
import torch
from scipy.interpolate import CubicSpline
import math


class TimeDependentAdjacentMatrix:
    def __init__(self, data: str = 'beijing', interpolate: str = 'linear', scale: float = 1.0, downsample: int = 12):
        self.matrix = self.parse(data, scale)
        self.num_nodes = self.matrix.shape[0]
        self.horizon = self.matrix.shape[-1]
        assert self.matrix.shape[1] == self.num_nodes, "The matrix must be square"
        if downsample < self.horizon:
            self._downsample(downsample)
            self.horizon = downsample
        self.coefficients = self._interpolate(interpolate)  # shape: [num_nodes, num_nodes, horizon, 4]
        self.constant = self._interpolate(method='constant')  # shape: [num_nodes, num_nodes, horizon, 2]
        self.static = False

    def _downsample(self, downsample: int):
        # downsample the matrix to reduce the number of time steps
        step = self.horizon // downsample
        self.matrix = self.matrix[:, :, ::step][:, :, :downsample] / step

    def to_constant(self):
        self.coefficients = self._interpolate(method='constant')

    def _interpolate(self, method: str = 'linear'):
        # coefficients = np.zeros((*self.matrix.shape, 4)) if self.interpolate == 'spline' else np.zeros((*self.matrix.shape, 2))
        coefficients = np.zeros((*self.matrix.shape, 2))  # shape: [num_nodes, num_nodes, horizon, 2]
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                line = self.matrix[i, j].numpy()
                if method == 'spline':
                    coefficients[i, j] = CubicSpline(
                        np.arange(self.horizon + 1),
                        np.concatenate((line, [line[0]]), axis=0),
                        bc_type='periodic'
                    ).c.T
                elif method == 'linear':
                    coefficients[i, j] = np.stack([np.concatenate((line[1:], [line[0]]), axis=0) - line, line], axis=-1)
                elif method == 'piecewise':
                    coefficients[i, j] = np.stack([np.zeros_like(line), line], axis=-1)
                elif method == 'constant':
                    # self.matrix[i, j] = torch.tensor(np.ones_like(line) * line[0], dtype=torch.float32)
                    coefficients[i, j] = np.stack([np.zeros_like(line), np.ones_like(line) * line[0]], axis=-1)
                else:
                    raise ValueError(f"Unknown interpolation method: {method}")
        return torch.tensor(coefficients, dtype=torch.float32)  # TODO: device

    def get_tour_length(self, cur_time: torch.Tensor, ordered_locs: torch.Tensor) -> torch.Tensor:
        time = torch.zeros(ordered_locs.shape[0], ordered_locs.shape[1] + 1, dtype=torch.float32, device=ordered_locs.device)
        time[:, 0] = cur_time
        ordered_locs_next = torch.roll(ordered_locs, -1, dims=1)
        for i in range(ordered_locs.shape[1]):
            time[:, i + 1] = time[:, i] + self.get_distance(ordered_locs[:, i], ordered_locs_next[:, i], time[:, i])
        return time[:, -1] - time[:, 0]

    def get_path_length(self, cur_time: torch.Tensor, ordered_locs: torch.Tensor) -> torch.Tensor:
        time = torch.zeros(ordered_locs.shape[0], ordered_locs.shape[1], dtype=torch.float32, device=ordered_locs.device)
        time[:, 0] = cur_time
        for i in range(ordered_locs.shape[1] - 1):
            time[:, i + 1] = time[:, i] + self.get_distance(ordered_locs[:, i], ordered_locs[:, i + 1], time[:, i])
        return time[:, -1] - time[:, 0]

    def get_distance(self, loc1: torch.Tensor, loc2: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        # loc1, loc2: [batch_size, 1]
        # time: [batch_size,]
        batch_idx = torch.arange(loc1.shape[0], device=loc1.device)
        batched_coef = self.coefficients[loc1, loc2].squeeze(1) if not self.static else self.constant[loc1, loc2].squeeze(1) # shape: [batch_size, horizon, 4]

        time_idx = torch.floor(time)
        surplus = time - time_idx
        batched_coef = batched_coef[batch_idx, time_idx.long() % self.horizon]  # shape: [batch_size, 4]
        # if self.interpolate == 'spline':
        #     time_poly = torch.stack([surplus ** 3, surplus ** 2, surplus, torch.ones_like(surplus)], dim=-1)
        # else:
        #     time_poly = torch.stack([surplus, torch.ones_like(surplus)], dim=-1)
        if batched_coef.shape[-1] == 4:
            time_poly = torch.stack([surplus ** 3, surplus ** 2, surplus, torch.ones_like(surplus)], dim=-1)  # shape: [batch_size, 2]
        else:
            time_poly = torch.stack([surplus, torch.ones_like(surplus)], dim=-1)
        return torch.sum(batched_coef * time_poly, dim=-1)

    def get_distances(self, locs1: torch.Tensor, locs2: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        # locs1, locs2: [batch_size, num_loc, 1]
        # time: [batch_size,]
        locs1 = locs1.squeeze(-1)
        locs2 = locs2.squeeze(-1)

        time_idx = torch.floor(time)
        surplus = time - time_idx

        batched_coef = self.coefficients[locs1[:, :, None], locs2[:, None, :]] if not self.static else self.constant[locs1[:, :, None], locs2[:, None, :]] # shape: [batch_size, num_loc, num_loc, horizon, 4]
        batch_idx = torch.arange(locs1.shape[0], device=locs1.device)
        batched_coef = batched_coef[batch_idx, :, :, time_idx.long() % self.horizon]  # shape: [batch_size, num_loc, num_loc, 4]
        # if self.interpolate == 'spline':
        #     time_coef = torch.stack([surplus ** 3, surplus ** 2, surplus, torch.ones_like(surplus)], dim=-1)
        # else:
        #     time_coef = torch.stack([surplus, torch.ones_like(surplus)], dim=-1)  # shape: [batch_size, 2]
        time_coef = torch.stack([surplus, torch.ones_like(surplus)], dim=-1)  # shape: [batch_size, 2]
        time_coef = time_coef[:, None, None, :]  # expand the shape to: [batch_size, 1, 1, 4]
        return torch.sum(batched_coef * time_coef, dim=-1)  # shape: [batch_size, num_loc, num_loc], all pairwise distances at a specific time

    def amplify_variants(self, matrix, factor=0.1):
        # amplify the variants of the matrix to enhance the temporal dependency
        mean_value = matrix.mean(dim=-1)
        deviation = matrix - mean_value[:, :, None]
        enhanced_matrix = mean_value[:, :, None] + deviation * (1 + factor)
        assert (enhanced_matrix >= 0).all(), "The enhanced matrix must be non-negative"
        return enhanced_matrix

    def amplify_congestion(self, matrix: torch.Tensor, k=0.1, factor=0.1):
        # amplify the congestion of top congested edges to enhance the temporal dependency
        mean_value = matrix.mean(dim=-1)
        congestion = matrix - mean_value[:, :, None]
        # find the top k congested edges
        k = int(matrix.numel() * k)
        congestion = congestion.flatten()
        _, indices = congestion.topk(k)
        congestion[indices] = congestion[indices] * (1 + factor)
        congestion = congestion.view_as(matrix)
        enhanced_matrix = mean_value[:, :, None] + congestion
        return enhanced_matrix

    def parse(self, filename, scale=1.0):
        matrix = np.load('~/rl4co/data/tdtsp/' + filename +'.npy')
        matrix = torch.tensor(matrix, dtype=torch.float32) * scale
        if filename == 'beijing':
            matrix = matrix.roll(2, dims=-1)
        elif filename == 'lyon':
            matrix = matrix[:, :, :40]
        else:
            self.amplify_variants(matrix, factor=0.2)
            matrix = self.select_congestion(matrix, 50)
        return matrix

    def select_congestion(self, matrix: torch.Tensor, num: int):
        variants = torch.var(matrix, dim=-1)
        congestion = torch.mean(variants, dim=-1)
        indices = torch.topk(congestion, num, largest=True)[1]
        print(indices.shape)
        row_selected = matrix[indices, :, :]
        column_selcted = row_selected[:, indices, :]
        print(column_selcted.shape)
        return column_selcted

    def select(self, indices):
        num_locs = indices.shape[1]
        idx_i = indices.expand(-1, -1, num_locs)  # shape: [batch_size, num_loc, num_loc]
        idx_j = idx_i.transpose(1, 2)
        return self.matrix[idx_i, idx_j]  # shape: [batch_size, num_loc, num_loc, horizon]

    def to(self, device):
        self.coefficients = self.coefficients.to(device)
        self.matrix = self.matrix.to(device)
        self.constant = self.constant.to(device)
        return self


