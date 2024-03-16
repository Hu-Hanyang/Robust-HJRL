import numpy as np
from gym_pybullet_drones.envs.HoverDistb import HoverDistbEnv
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from stable_baselines3.common.env_checker import check_env

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

env = HoverDistbEnv(disturbance_type='fixed', distb_level=1.0)
check_env(env)
# print(env.ACTION_BUFFER_SIZE)
# print(env.distb_level)
# print(env.observation_space)
# init_obs, init_info = env.reset()

# print(init_obs.shape)

# standard_env = HoverAviary()
# check_env(standard_env)
