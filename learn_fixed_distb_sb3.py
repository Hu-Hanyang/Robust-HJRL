"""Training the CrazyFlie in environments with heterogeneous disturbances based on the PPO algorithm.

The original code repository is https://github.com/utiasDSL/gym-pybullet-drones. 

Example
-------
In a terminal, run as:

    $ python learn.py --multiagent false

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

DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = True
# DEFAULT_OUTPUT_FOLDER = '_results'
DEFAULT_COLAB = False

DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'
DEFAULT_ACT = ActionType('pwm') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid' 'pwm' #TODO: check here!
DEFAULT_AGENTS = 2
DEFAULT_MA = False

DEFAULT_DISTURBANCE_TYPE = 'fixed'
assert DEFAULT_DISTURBANCE_TYPE in ['fixed', 'boltzmann', 'random', 'rarl', 'rarl-population']
DEFAULT_DISTURBANCE_LEVEL = 0.0


class ValidateCheckpointCallback(BaseCallback):
    """
    Callback for saving a model every ``save_freq`` calls
    to ``env.step()`` and validating the model immediately.
    By default, it only saves model checkpoints,
    you need to pass ``save_replay_buffer=True``,
    and ``save_vecnormalize=True`` to also save replay buffer checkpoints
    and normalization statistics checkpoints.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``save_freq = max(save_freq // n_envs, 1)``

    :param save_freq: Save checkpoints every ``save_freq`` call of the callback.
    :param save_path: Path to the folder where the model will be saved.
    :param name_prefix: Common prefix to the saved models
    :param save_replay_buffer: Save the model replay buffer
    :param save_vecnormalize: Save the ``VecNormalize`` statistics
    :param verbose: Verbosity level: 0 for no output, 2 for indicating when saving model checkpoint
    :param 
    """

    def __init__(
        self,
        save_freq: int,
        save_path: str,
        name_prefix: str = "rl_model",
        save_replay_buffer: bool = False,
        save_vecnormalize: bool = False,
        verbose: int = 0,
        distb_type: str = 'fixed',
        distb_level: float = 0.0,
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.save_replay_buffer = save_replay_buffer
        self.save_vecnormalize = save_vecnormalize
        self.distb_type = distb_type
        self.distb_level = distb_level

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _checkpoint_path(self, checkpoint_type: str = "", extension: str = "") -> str:
        """
        Helper to get checkpoint path for each type of checkpoint.

        :param checkpoint_type: empty for the model, "replay_buffer_"
            or "vecnormalize_" for the other checkpoints.
        :param extension: Checkpoint file extension (zip for model, pkl for others)
        :return: Path to the checkpoint
        """
        return os.path.join(self.save_path, f"{self.name_prefix}_{checkpoint_type}{self.num_timesteps}_steps.{extension}")

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            model_path = self._checkpoint_path(extension="zip")
            self.model.save(model_path)
            # Hanyang: validate the model immediately
            self._validate(model_path)

            if self.verbose >= 2:
                print(f"Saving model checkpoint to {model_path}")

            if self.save_replay_buffer and hasattr(self.model, "replay_buffer") and self.model.replay_buffer is not None:
                # If model has a replay buffer, save it too
                replay_buffer_path = self._checkpoint_path("replay_buffer_", extension="pkl")
                self.model.save_replay_buffer(replay_buffer_path)
                if self.verbose > 1:
                    print(f"Saving model replay buffer checkpoint to {replay_buffer_path}")

            if self.save_vecnormalize and self.model.get_vec_normalize_env() is not None:
                # Save the VecNormalize statistics
                vec_normalize_path = self._checkpoint_path("vecnormalize_", extension="pkl")
                self.model.get_vec_normalize_env().save(vec_normalize_path)
                if self.verbose >= 2:
                    print(f"Saving model VecNormalize to {vec_normalize_path}")

        return True
    
    def _validate(self, model_path):
        validate_model = PPO.load(model_path)

        #### Make save path ###################################
        validate_result = os.path.dirname(model_path) + f'/validations/'
        if not os.path.exists(validate_result):
            os.makedirs(validate_result+'/')
        
        #### Create the environment ################################
        validate_env = HoverFixedDistbEnv(disturbance_type=self.distb_type, distb_level=self.distb_level, record=True)
        num_gifs = 5
        frames = [[] for _ in range(num_gifs)]
        num = 0
        
        while num < num_gifs:
            terminated, truncated = False, False
            rewards = 0.0
            steps = 0
            obs, info = validate_env.reset()
            frames[num].append(validate_env.get_CurrentImage())  # the return frame is np.reshape(rgb, (h, w, 4))
            
            for _ in range(500):
                action, _ = validate_model.predict(obs, deterministic=False)
                obs, reward, terminated, truncated, info = validate_env.step(action)
                frames[num].append(validate_env.get_CurrentImage())
                rewards += reward
                steps += 1
                
                if terminated or truncated or steps>=500:
                    print(f"[INFO] Test {num} is terminated or truncated with rewards = {rewards} and {steps} steps.")
                    self._generate_gifs(frames[num], validate_result, num)
                    num += 1
                    break
        validate_env.close()
    
    def _generate_gifs(self, frames, save_path, idx):
        # Initialize video writer
        images = []
        for frame in frames:
            images.append(frame.astype(np.uint8))

        imageio.mimsave(f'{save_path}-gif{idx}.gif', images, duration=0.1)


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self):
        # Log scalar value (here a random variable)
        # success_rate = self.training_env.get_success_rate(window_size=100)
        if len(self.model.ep_info_buffer) > 0 and len(self.model.ep_info_buffer[0]) > 0:
            # success_rate = safe_mean([ep_info["s"] for ep_info in self.model.ep_info_buffer])
            # self.logger.record("rollout/ep_success_rate", success_rate)
            # action = safe_mean([ep_info["action"][3] for ep_info in self.model.ep_info_buffer])
            # self.logger.record("rollout/push_velocity", action)
            return True


class CustomMonitor(Monitor):
    #Hanayng: not using this class now
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.episode_successes = []

    def step(self, action):
        """
        Step the environment with the given action

        :param action: the action
        :return: observation, reward, done, information
        """
        observation, reward, terminated, truncated, info = super().step(action)

        if terminated or truncated:
            # info["episode"]["s"] = info.get('success')
            info["episode"]["action"] = action
        return observation, reward, terminated, truncated, info
    
    

def train(settings="training_fixed.json", multiagent=False):
    
    #### Load the training settings ###################################
    with open(settings) as f:
        training_data = json.load(f)
        distb_type = training_data['distb_type']
        n_env = training_data['n_envs']
        env_seed = training_data['env_seed']
        train_seed = training_data['train_seed']
        batch_size = training_data['batch_size']
        n_epochs = training_data['n_epochs']
        n_steps = training_data['n_steps']
        target_kl = training_data['target_kl']
        total_timesteps = training_data['total_timesteps']
    
    env = HoverFixedDistbEnv()
    randomization_reset = env.randomization_reset
    distb_level = env.distb_level
    
    #### Make save path ###################################
    if distb_type == 'fixed' or None:
        filename = os.path.join('training_results_sb3/' + 'fixed'+'-'+f'distb_level_{distb_level}', 'seed_'+f"{train_seed}", 'save-'+f'intial_random_{randomization_reset}') 
    else:  # 'boltzmann', 'random', 'rarl', 'rarl-population'
        filename = os.path.join('training_results_sb3/' + distb_type, 'seed_'+f"{train_seed}", 'save-'+f'intial_random_{randomization_reset}') 
    if not os.path.exists(filename):
        os.makedirs(filename+'/')

    shutil.copy(settings, filename)
    # Read the existing data from the copied file
    with open(f"{filename}/{settings}", 'r') as json_file:
        data = json.load(json_file)
    data.update({"randomization_reset": randomization_reset, "distb_level": distb_level})
    # Write the updated dictionary back to the JSON file
    with open(f"{filename}/{settings}", 'w') as json_file:
        json.dump(data, json_file, indent=4)
    print(f"[INFO] Save the training settings at: {filename}/{settings}")


    #### Create the environment ################################
    if not multiagent:
        train_env = make_vec_env(HoverFixedDistbEnv,
                                 env_kwargs=dict(obs=DEFAULT_OBS, act=DEFAULT_ACT),
                                 n_envs=n_env,
                                 seed=env_seed
                                 )
        # eval_env = HoverDistbEnv(disturbance_type=distb_type, distb_level=distb_level,obs=DEFAULT_OBS, act=DEFAULT_ACT)
    else:
        train_env = make_vec_env(MultiHoverAviary,
                                 env_kwargs=dict(num_drones=DEFAULT_AGENTS, obs=DEFAULT_OBS, act=DEFAULT_ACT),
                                 n_envs=1,
                                 seed=0
                                 )
        # eval_env = MultiHoverAviary(num_drones=DEFAULT_AGENTS, obs=DEFAULT_OBS, act=DEFAULT_ACT)

    #### Check the environment's spaces ########################
    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)
    #### Train the model #######################################
    model = PPO('MlpPolicy',
                train_env,
                batch_size=batch_size,
                n_epochs=n_epochs,
                n_steps=n_steps,
                seed=train_seed,
                target_kl=target_kl, 
                tensorboard_log=filename+'/tb/',
                verbose=1)

    # #### Target cumulative rewards (problem-dependent) ##########
    # if DEFAULT_ACT == ActionType.ONE_D_RPM:
    #     target_reward = 474.15 if not multiagent else 949.5
    # else:
    #     target_reward = 467. if not multiagent else 920.
    # callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=target_reward,
    #                                                  verbose=1)

    #TODO: test the customized wrapper
    # checkpoint_callback = CheckpointCallback(
    #                         save_freq=1e4,
    #                         save_path=f"{filename}/train_logs/",
    #                         name_prefix="PPO",
    #                         save_replay_buffer=True,
    #                         save_vecnormalize=True,
    #                         )
    
    checkpoint_callback = ValidateCheckpointCallback(
                            save_freq=1e5,
                            save_path=f"{filename}/train_logs/",
                            name_prefix="PPO",
                            save_replay_buffer=True,
                            save_vecnormalize=True,
                            distb_type=distb_type,
                            distb_level=distb_level,
                            )

    
    tensorboard_callback = TensorboardCallback()
    train_callback = CallbackList([checkpoint_callback, tensorboard_callback])
    
    # eval_callback = EvalCallback(eval_env,
    #                              callback_on_new_best=callback_on_best,
    #                              verbose=1,
    #                              best_model_save_path=filename+'/',
    #                              log_path=filename+'/',
    #                              eval_freq=int(1000),
    #                              deterministic=True,
    #                              render=False)
    
    #### Train the model #######################################
    print("Start training")
    start_time = time.perf_counter()
    model.learn(total_timesteps=int(total_timesteps), callback=train_callback)

    #### Save the model ########################################
    model.save(filename+'/final_model.zip')
    print(filename)
    duration = time.perf_counter() - start_time
    print(f"The time of training is {duration//3600}hours-{(duration%3600)//60}minutes-{(duration%3600)%60}seconds. \n")
    

    # #### Print training progression ############################
    # with np.load(filename+'/evaluations.npz') as data:
    #     for j in range(data['timesteps'].shape[0]):
    #         print(str(data['timesteps'][j])+","+str(data['results'][j][0]))

    ############################################################
    ############################################################
    ############################################################
    ############################################################
    ############################################################

    # # if local:
    # #     input("Press Enter to continue...")

    # # if os.path.isfile(filename+'/final_model.zip'):
    # #     path = filename+'/final_model.zip'
    # if os.path.isfile(filename+'/best_model.zip'):
    #     path = filename+'/best_model.zip'
    # else:
    #     print("[ERROR]: no model under the specified path", filename)
    # model = PPO.load(path)


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

    parser.add_argument('--train_distb_type',         default="fixed",      type=str,           help='Type of disturbance to be applied to the drones [None, "fixed", "boltzmann", "random", "rarl", "rarl-population"] (default: "fixed")', metavar='')
    parser.add_argument('--train_distb_level',        default=0.0,          type=float,         help='Level of disturbance to be applied to the drones (default: 0.0)', metavar='')
    parser.add_argument('--train_seed',               default=42,        type=int,           help='Seed for the random number generator (default: 40226)', metavar='')
    parser.add_argument('--multiagent',         default=False,        type=str2bool,      help='Whether to use example LeaderFollower instead of Hover (default: False)', metavar='')
    parser.add_argument('--randomization_reset',         default=True,        type=str2bool,      help='Whether to use example LeaderFollower instead of Hover (default: False)', metavar='')
    parser.add_argument('--settings',           default="training_fixed.json",        type=str,           help='The path to the training settings file (default: None)', metavar='')
    parser.add_argument('--test_distb_type',    default="fixed",      type=str,           help='Type of disturbance in the test environment', metavar='')
    parser.add_argument('--test_distb_level',   default=0.0,          type=float,         help='Level of disturbance in the test environment', metavar='')
    parser.add_argument('--max_test_steps',     default=500,          type=int,           help='Maximum number of steps in the test environment', metavar='')
    parser.add_argument('--num_videos',         default=3,            type=int,           help='Number of videos to generate in the test environment', metavar='')
    parser.add_argument('--fps',                default=50,           type=int,           help='Frames per second in the generated videos', metavar='')
    
    args = parser.parse_args()

    train(settings=args.settings, multiagent=args.multiagent)