import numpy as np
from gym_pybullet_drones.envs.HoverDistb import HoverFixedDistbEnv, HoverBoltzmannDistbEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
import os 
import json
import pandas
import torch
import torch.nn as nn
import imageio


# Function to create GIF
def create_gif(image_list, filename, duration=0.1):
    images = []
    for img in image_list:
        images.append(img.astype(np.uint8))  # Convert to uint8 for imageio
    imageio.mimsave(f'{filename}', images, duration=duration)


# Check the env
env = HoverFixedDistbEnv()
print(env.observation_space)
print(env.action_space)
# print(env.randomization_reset)
check_env(env)  #TODO: Not pass
# state.shape = (20,) state = [pos, quat, rpy, vel, ang_vel, last_clipped_action]
# init_obs, init_info = env.reset()
# print(f"The self.laset_clipped_action is {env.last_clipped_action} and its shape is {env.last_clipped_action.shape}")
# print(f"The init_obs shape is {init_obs.shape} in line 28")
# print(f"The initial xyz are {init_obs[0][0:3]} and its shape is {init_obs[0][0:3].shape}")
# print(f"The initial quaternion are {init_obs[0][3:7]} and its shape is {init_obs[0][3:7].shape}")
# print(f"The initial xyz-velocities are {init_obs[0][10:13]} and its shape is {init_obs[0][10:13].shape}")
# print(f"The initial rpy are {init_obs[0][7:10]}")
# print(f"The initial rpy-velocities are {init_obs[0][10:13]}")
# print(f"The initial clipped action is {init_obs[0][13:17]} and its shape is {init_obs[0][13:17].shape} in line 34")
# print(f"The initial init_obs is {init_obs[0]}")
# model = PPO.load("training_results_sb3/fixed-distb_level_1.0/seed_40226/save-2024.03.26_12:13/train_logs/PPO_6720000_steps.zip")

# check performances
print(f"The current env's disturbance level is {env.distb_level}")
num_gifs = 1
frames = [[] for _ in range(num_gifs)]


# state = env._getDroneStateVector(0)
# print(f"The state is {state}.")


# # Generate GIFs
# num=0

# while num < num_gifs:
#     terminated, truncated = False, False
#     rewards = 0.0
#     steps = 0
#     max_steps=100
#     init_obs, init_info = env.reset()
#     print(f"The init_obs shape is {init_obs.shape}")
#     print(f"The initial position is {init_obs[0][0:3]}")
#     print(f"The intial obs is {init_obs}.")
#     frames[num].append(env.get_CurrentImage())  # the return frame is np.reshape(rgb, (h, w, 4))
    
#     for _ in range(max_steps):
#         if _ == 0:
#             obs = init_obs
            
#         # manual control
#         # motor = -0.01 + 0.001*_
#         motor = 0.01
#         action = np.array([[motor, motor*10, motor, motor]])
#         # print(f"The current step{_} action is {action}.")
#         # print(f"The last step action is {env.last_action}.")
#         # random control
#         # action = env.action_space.sample()
#         # load the model to control
#         # action, _ = model.predict(obs, deterministic=False)

#         obs, reward, terminated, truncated, info = env.step(action)
#         # print(f"After the step, the last step action is {env.last_action}.")
#         # print(f"The current obs of the step{_} is {obs}.")
        
#         # print(f"The current last action is {obs[0][13:17]}")
#         print(f"The current reward of the step{_} is {reward} and this leads to {terminated} and {truncated}")
#         # print(f"The current penalty of the step{_} is {info['current_penalty']} and the current distance is {info['current_dist']}")
#         frames[num].append(env.get_CurrentImage())
#         rewards += reward
#         steps += 1
        
#         if terminated or truncated or steps>=max_steps:
#             print(f"[INFO] Test {num} is terminated or truncated with rewards = {rewards} and {steps} steps.")
#             create_gif(frames[num], f'check_env_gif{num}-motor{motor}-distb{env.distb_level}-{steps}.gif', duration=0.1)
#             print(f"The final position is {obs[0][0:3]}.")
#             num += 1
#             break
# env.close()
    

# # # Test the shape of the action
# # action = np.array([[0.0, 0.0, 0.0, 0.0]])
# # print(f"The shape of action is {action.shape}.")

# # pwms = np.zeros((2, 4))
# # for n in range(2):
# #     pwms[0, :] = 30000 + np.clip(action, -1, +1) * 30000  # PWM in [0, 60000]
# #     print(f"The shape of pwms[{n}] is {pwms[n].shape}.")

# # print(f"The shape of pwms is {pwms.shape}.")