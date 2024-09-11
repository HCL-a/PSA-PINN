import numpy as np
import torch.cuda
from pyDOE import lhs


def trainingData(num_bc, num_f):
    np.random.seed(1234)
    lb = np.array([0, 0])
    ub = np.array([1, 1])
    bc_left_xy = np.vstack((np.zeros(num_bc), np.random.rand(num_bc))).T
    bc_right_xy = np.vstack((np.ones(num_bc), np.random.rand(num_bc))).T
    bc_top_xy = np.vstack((np.random.rand(num_bc), np.ones(num_bc))).T
    bc_bottom_xy = np.vstack((np.random.rand(num_bc), np.zeros(num_bc))).T
    bc_left_p = np.ones(num_bc)[:, None]
    bc_right_p = np.zeros(num_bc)[:, None]
    bc_D_xy = np.vstack([bc_left_xy, bc_right_xy])
    bc_D_p = np.vstack([bc_left_p, bc_right_p])
    bc_N_xy = np.vstack([bc_top_xy, bc_bottom_xy])
    bc_xy_train = np.vstack([bc_left_xy, bc_right_xy, bc_bottom_xy, bc_top_xy])

    xy_train = lb + (ub - lb) * lhs(2, num_f)
    all_xy_train = np.vstack((bc_xy_train, xy_train))

    bc_D_xy = torch.tensor(bc_D_xy, dtype=torch.float32)
    bc_D_p = torch.tensor(bc_D_p, dtype=torch.float32)
    bc_N_xy = torch.tensor(bc_N_xy, dtype=torch.float32)
    all_xy_train = torch.tensor(all_xy_train, dtype=torch.float32)

    return bc_D_xy, bc_D_p, bc_N_xy, all_xy_train
