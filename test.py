"""Test the CrazyFlie in environments with heterogeneous disturbances based on the PPO algorithm.

The original code repository is https://github.com/utiasDSL/gym-pybullet-drones. 

Example
-------
In a terminal, run as:

    $ python test.py 

Notes
-----
This is a minimal working example integrating `gym-pybullet-drones` with 
reinforcement learning library `stable-baselines3`.

"""
import os
import time
import json
from datetime import datetime
import argparse
import gymnasium as gym
import numpy as np
import shutil
import torch
import cv2
import imageio
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, CallbackList, CheckpointCallback, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import obs_as_tensor, safe_mean, set_random_seed
from stable_baselines3.common.monitor import Monitor


from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.HoverDistb import HoverFixedDistbEnv
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType


def test(train_distb_type, train_distb_level, train_seed, randomization_reset, 
         test_distb_type, test_distb_level,  max_test_steps,  num_videos):

    #### Load the trained model ###################################
    if train_distb_type == 'fixed' or None:
        trained_model = f"training_results_sb3/fixed-distb_level_{train_distb_level}/seed_{train_seed}/save-initial_random_{randomization_reset}/final_model.zip"
    else:  # 'boltzmann', 'random', 'rarl', 'rarl-population'
        trained_model = f"training_results_sb3/{train_distb_type}/seed_{train_seed}/save-initial_random_{randomization_reset}/final_model.zip" 
    assert os.path.exists(trained_model), f"[ERROR] The trained model {trained_model} does not exist, please check the loading path or train one first."
    
    model = PPO.load(trained_model)
    
    #### Create the environment ################################
    env = HoverFixedDistbEnv(disturbance_type=test_distb_type, distb_level=test_distb_level, record=True, randomization_reset=randomization_reset)
    print(f"[INFO] The test environment is with {test_distb_type} distb type and {test_distb_level} distb level.")
    
    #### Make save path ###################################
    if test_distb_type == 'fixed' or None:
        filename = os.path.join('test_results_sb3/' + 'fixed'+'-'+f'distb_level_{test_distb_level}', f'using-{train_distb_type}-distb_level_{train_distb_level}_model', f'initial_random_{randomization_reset}') 
    else:  # 'boltzmann', 'random', 'rarl', 'rarl-population'
        filename = os.path.join('test_results_sb3/' + test_distb_type, f'using-{train_distb_type}_model', f'initial_random_{randomization_reset}')
    if not os.path.exists(filename):
        os.makedirs(filename+'/')
    print(f"[INFO] Save the test videos (GIFs) at: {filename}")

    frames = [[] for _ in range(num_videos)]
    num = 0
    
    while num < num_videos:
        terminated, truncated = False, False
        rewards = 0.0
        steps = 0
        obs, info = env.reset()
        frames[num].append(env.get_CurrentImage())  # the return frame is np.reshape(rgb, (h, w, 4))
        
        for step in range(max_test_steps):
            action, _ = model.predict(obs, deterministic=False)
            # print(f"The current action {step} is {action}")
            obs, reward, terminated, truncated, info = env.step(action)
            frames[num].append(env.get_CurrentImage())
            rewards += reward
            steps += 1
            
            if terminated or truncated or steps>=max_test_steps:
                print(f"[INFO] Test {num+1} is terminated or truncated with rewards = {rewards} and {steps} steps.")
                # generate_videos(frames[num], filename, num, fps)
                generate_gifs(frames[num], filename, num)
                num += 1
                break


def generate_videos(frames, save_path, idx, fps):
    # Initialize video writer
    video_path = f'{save_path}/output{idx}.mp4'
    # out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frames[0].shape[1], frames[0].shape[0]))
    # # Write each frame to the video
    # for frame in frames:
    #     # Convert RGBA to BGR
    #     bgr_frame = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGBA2BGR)
    #     out.write(bgr_frame)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use other codecs as well (e.g., 'XVID')
    out = cv2.VideoWriter(video_path, fourcc, fps, (frames[0].shape[1], frames[0].shape[0]))

    # Write frames to the video file
    for image in frames:
        image = np.asarray(image, dtype=np.uint8)
        out.write(image)
    # Release video writer
    out.release()
    print(f"[INFO] The video {idx} is saved as: {video_path}.")


# Function to create GIF
def generate_gifs(frames, save_path, idx, duration=0.1):
    images = []
    for frame in frames:
        images.append(frame.astype(np.uint8))  # Convert to uint8 for imageio
    imageio.mimsave(f'{save_path}/gif{idx+1}.gif', images, duration=duration)



if __name__ == '__main__':
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script')

    parser.add_argument('--train_distb_type',         default="boltzmann",      type=str,           help='Type of disturbance to be applied to the drones [None, "fixed", "boltzmann", "random", "rarl", "rarl-population"] (default: "fixed")', metavar='')
    parser.add_argument('--train_distb_level',        default=0.0,          type=float,         help='Level of disturbance to be applied to the drones (default: 0.0)', metavar='')
    parser.add_argument('--train_seed',               default=42,        type=int,           help='Seed for the random number generator (default: 40226)', metavar='')
    parser.add_argument('--multiagent',         default=False,        type=str2bool,      help='Whether to use example LeaderFollower instead of Hover (default: False)', metavar='')
    parser.add_argument('--randomization_reset',         default=True,        type=str2bool,      help='Whether to use example LeaderFollower instead of Hover (default: False)', metavar='')
    parser.add_argument('--test_distb_type',    default="fixed",      type=str,           help='Type of disturbance in the test environment', metavar='')
    parser.add_argument('--test_distb_level',   default=1.0,          type=float,         help='Level of disturbance in the test environment', metavar='')
    parser.add_argument('--max_test_steps',     default=500,          type=int,           help='Maximum number of steps in the test environment', metavar='')
    parser.add_argument('--num_videos',         default=3,            type=int,           help='Number of videos to generate in the test environment', metavar='')
    parser.add_argument('--fps',                default=50,           type=int,           help='Frames per second in the generated videos', metavar='')
    
    args = parser.parse_args()

    test(train_distb_type=args.train_distb_type, train_distb_level=args.train_distb_level, train_seed=args.train_seed, 
         randomization_reset=args.randomization_reset, test_distb_type=args.test_distb_type, test_distb_level=args.test_distb_level, 
         max_test_steps=args.max_test_steps, num_videos=args.num_videos)

    # python test.py --train_distb_type boltzmann --train_seed 42 --test_distb_type fixed --test_distb_level 1.0 --max_test_steps 500 --num_videos 3