import numpy as np
from typing import Tuple, List, Union

import numpy.random as npr
import random
from numpy.linalg import inv

def convert_vec_to_two_dim(index_vec, modulus) -> np.array(Tuple[int, int]):
    return np.array([convert_to_two_dim(index, modulus) for index in index_vec])

def convert_to_two_dim(index, modulus) -> Tuple[int, int]:
    return int(index / modulus), int(index % modulus)

def create_obs_by_group(group_vec, modulus, treat_dummy, control_outcome, sd, te, e_vec):
    group_loc = np.where(group_vec == treat_dummy)[0]
    group_mat_loc = convert_vec_to_two_dim(group_loc, modulus)

    obs_group = []
    for i in range(len(group_loc)):
        obs_group.append((group_mat_loc[i][0], group_mat_loc[i][1],
                          control_outcome[group_mat_loc[i][0]][group_mat_loc[i][1]] + treat_dummy*te + npr.normal(scale=sd) + e_vec[i] ))
    return obs_group