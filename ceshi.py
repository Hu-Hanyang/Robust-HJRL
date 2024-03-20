import numpy as np
from gym_pybullet_drones.envs.HoverDistb import HoverDistbEnv
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from stable_baselines3.common.env_checker import check_env
import os 
import json
import pandas
import torch
import torch.nn as nn

# distb_type = "aixed"
# if distb_type not in [None, 'fixed', 'boltzmann', 'random', 'rarl', 'rarl-population']:
#     print("Invalid distb_type")

# distb_leve = 2.0
# if distb_leve not in np.arange(0.0, 2.1, 0.1):
#     print("Invalid distb_leve")


# distb_level = np.round(np.random.uniform(0.0, 2.1), 1)
# print(distb_level)


# num_drones = 3

# pos = np.zeros((num_drones, 3))

# pos_lim = 0.25

# for i in range(num_drones):
# #     pos[i] += np.random.uniform(-pos_lim, pos_lim, 3)
# # print(pos)
#     pos[i] += np.random.uniform(-pos_lim, pos_lim, 3)
#     pos[i][2] = np.random.uniform(0.5, 1.5)
# print(pos)



# print(init_obs.shape)

# standard_env = HoverAviary()
# check_env(standard_env)

# initial_xyzs = np.array([0, 0, 1], dtype=np.float32)
# print(initial_xyzs.shape)

# changed = initial_xyzs.reshape(1,3)
# print(changed.shape)
# L = 0.0397
# COLLISION_H = 0.1
# COLLISION_Z_OFFSET = 0.1
# INIT_XYZS = np.vstack([np.array([x*4*L for x in range(1)]), \
#                                         np.array([y*4*L for y in range(1)]), \
#                                         np.ones(1) * (COLLISION_H/2-COLLISION_Z_OFFSET+.1)]).transpose().reshape(1, 3)

# print(INIT_XYZS.shape)
# initial_xyzs = np.array([[0, 0, 1]], dtype=np.float32)
# print(initial_xyzs.shape)

# initial_rpys=np.zeros((1, 3))
# print(initial_rpys.shape)


env = HoverDistbEnv(disturbance_type='fixed', distb_level=1.0)
check_env(env)
# print(env.ACTION_BUFFER_SIZE)
# print(env.distb_level)
# print(env.observation_space)
# print(env.action_space)
# init_obs, init_info = env.reset()
# print(env.pos)


# env = HoverAviary()
# check_env(env)

