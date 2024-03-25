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


# env = HoverDistbEnv(disturbance_type='fixed', distb_level=0.0)
# print(env.OUTPUT_FOLDER)

# print(env.ACTION_BUFFER_SIZE)
# print(env.distb_level)
# # print(env.observation_space)
# # print(env.action_space)
# init_obs, init_info = env.reset()
# print(init_obs)
# # print(env.pos)

# obs = env.reset()
# env.render()
# for i in range(100):
#     # action, _ = model.predict(obs, deterministic=False)
#     action = env.action_space.sample()
#     obs, reward, terminated, truncated, info = env.step(action)
#     # for index in range(len(info["render_image"])):
#         # writer.append_data(cv2.rotate(info["render_image"][index], cv2.ROTATE_180))
#     if terminated or truncated:
#         obs = env.reset()
# env.close()


import numpy as np
import imageio
import cv2

# Assuming your list of numpy arrays is called image_list
# image_list should contain RGBA images of shape (h, w, 4)
# For the sake of this example, let's create a dummy image_list
# Replace this with your actual list
image_list = [np.random.rand(100, 100, 4) for _ in range(50)]

# Initialize video writer
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (image_list[0].shape[1], image_list[0].shape[0]))

# Write each frame to the video
for frame in image_list:
    # Convert RGBA to BGR
    bgr_frame = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGBA2BGR)
    out.write(bgr_frame)

# Release video writer
out.release()

