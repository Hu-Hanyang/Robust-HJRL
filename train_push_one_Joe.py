import gym
import numpy as np
import imageio
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.utils import obs_as_tensor, safe_mean, set_random_seed
from robosuite import load_controller_config
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, BaseCallback
import cv2
import argparse
import pickle
from env_push_one import PusherOneSingleAction
from env_push_box import Push_Box


class NormalizeActionSpaceWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Store both the high and low arrays in their original forms
        self.action_space_low = self.action_space.low
        self.action_space_high = self.action_space.high

        # We normalize action space to a range [-1, 1]
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=self.action_space.shape, dtype=np.float32)

    def action(self, action):
        # convert action from [-1,1] to original range
        action = self.denormalize_action(action)
        return action

    def reverse_action(self, action):
        # convert action from original range to [-1,1]
        action = self.normalize_action(action)
        return action

    def normalize_action(self, action):
        action = 2 * ((action - self.action_space_low) / (self.action_space_high - self.action_space_low)) - 1
        return action

    def denormalize_action(self, action):
        action = (action + 1) / 2 * (self.action_space_high - self.action_space_low) + self.action_space_low
        return action


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        # success_rate = self.training_env.get_success_rate(window_size=100)
        if len(self.model.ep_info_buffer) > 0 and len(self.model.ep_info_buffer[0]) > 0:
            success_rate = safe_mean([ep_info["s"] for ep_info in self.model.ep_info_buffer])
            self.logger.record("rollout/ep_success_rate", success_rate)
            action = safe_mean([ep_info["action"][3] for ep_info in self.model.ep_info_buffer])
            self.logger.record("rollout/push_velocity", action)
            return True



def init_env(render, n_envs=8, context=None):

    def make_env():
        def _make_env():
            env=Push_Box(render=render)
            if context is not None:
                env.set_context(context)
                # np.random.seed(0)
            return NormalizeActionSpaceWrapper(env)
        
        if n_envs == -1:
            return _make_env()
        else:
            return CustomMonitor(_make_env())
            # return _make_env()

    if n_envs == -1:
        return make_env()
    if n_envs == 1:
        return DummyVecEnv([make_env for _ in range(n_envs)])
    else:
        return SubprocVecEnv([make_env for _ in range(n_envs)])

class CustomMonitor(Monitor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.episode_successes = []

    def step(self, action):
        """
        Step the environment with the given action

        :param action: the action
        :return: observation, reward, done, information
        """
        observation, reward, done, info = super().step(action)

        if done:
            info["episode"]["s"] = info.get('success')
            info["episode"]["action"] = action
        return observation, reward, done, info



def train(total_timesteps=2000000, algorithm='ppo', folder='', context = None, learning_rate=5e-4):
    
    env = init_env(render=False, n_envs=64, context=context)
    
    # Save a checkpoint every 1000 steps
    checkpoint_callback = CheckpointCallback(
                            save_freq=60,
                            save_path=f"train_logs/push_one/{algorithm}/{folder}",
                            name_prefix=algorithm,
                            save_replay_buffer=True,
                            save_vecnormalize=True,
                            )

    tensorboard_callback = TensorboardCallback()
    train_callback = CallbackList([checkpoint_callback, tensorboard_callback])
    
    if algorithm == 'ppo':
    
        ########## PPO Training ##########
        model = PPO(
            "MlpPolicy",
            env,
            # ent_coef=0.001,
            target_kl=0.03,
            verbose=0,
            batch_size = 64, # 64
            n_steps = 10, # 512
            n_epochs = 5,
            learning_rate = learning_rate,
            tensorboard_log="./PPO_Pusher_Tensorboard/push_one/",
            policy_kwargs=dict(
                net_arch=[64, 64],
                log_std_init=-1.5,
                ),

        )
        model.learn(total_timesteps=total_timesteps, callback=train_callback)

        model.save("ppo_pusher")
        ###################################
    elif algorithm == 'sac':
        ######### SAC Training ###########
        
        model = SAC("MlpPolicy",    # policy type
                    env,            # environment
                    verbose=1,      # print progressbar
                    learning_starts=100,
                    gradient_steps=32,
                    batch_size=64,
                    train_freq=8,
                    ent_coef=0.1,
                    # action_noise = NormalActionNoise(mean=np.zeros(4), sigma=0.5 * np.ones(4)),
                    policy_kwargs=dict(net_arch=[32, 32]),
                    tensorboard_log="./SAC_Pusher_Tensorboard/push_one/"
                    )
        model.learn(total_timesteps=total_timesteps, callback=train_callback)
        model.save("sac_pusher")
        
    else: 
        raise NotImplementedError

# Hanyang: render is to show the video of the trained model
def render(steps,folder,algorithm, context=None):
    ## Save rendering video
    writer = imageio.get_writer('train_logs/push_one/train_video.mp4', fps=20)
    env = init_env(render=True, n_envs=-1, context=context)

    video_length = 5
    ## If you are trying to reproduce the results, please modify the loading path
    if algorithm == 'sac':
        model = SAC.load(f"train_logs/push_one/sac/{folder}/sac_{steps}_steps.zip")
    elif algorithm == 'ppo':
        model = PPO.load(f"train_logs/push_one/ppo/{folder}/ppo_{steps}_steps.zip")
    obs = env.reset()
    for i in range(video_length):
        action, _ = model.predict(obs, deterministic=False)
        obs, rewards, dones, info = env.step(action)
        for index in range(len(info["render_image"])):
            writer.append_data(cv2.rotate(info["render_image"][index], cv2.ROTATE_180))
        if dones:
            obs = env.reset()
    writer.close()
    env.close()
    
def policy_training(context):
    # If the context is provided then load the env_params
    # this is based on the how we loaded the env_params before. we have to select "env_params" first and then we have access 
    # to the env_params.
    if context is not None:
        training_env_params = context['env_params']
    train (total_timesteps=10000, algorithm='sac', folder = 'COMPASS', context=training_env_params)
    print ("----------------Algorithm training finished------------------")
    agent = SAC.load(f"logs/pusher/sac/COMPASS/sac_38400_steps.zip")
    return agent

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description ='Training Pusher')
    
    parser.add_argument('--task', type=str, default="render", help="Select whether to train or render.")
    parser.add_argument('--steps', type=str, default="3840", help="render steps")
    parser.add_argument('--folder', type=str, default='', help="Log directory")
    parser.add_argument('--algo', type=str, default='ppo', help="training algorithm")
    parser.add_argument('--load_context', type=str, default=None, help="context")
    parser.add_argument('--seed', type=int, default=63, help="set seed")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="lr")
    args = parser.parse_args()
   
    # If the context is provided then load the env_params
    if args.load_context is not None:
        with open (f'{args.load_context}','rb') as f:    
            data = pickle.load(f)
        # context = data
        context = data['e_dr_sim']
        context['init@target@position'] = np.array([0.03, 0, 0.96])
    else: 
        context = None
    
    set_random_seed(args.seed, using_cuda=True)
    
    if args.task == "train":
        train(total_timesteps=150000, algorithm=args.algo, folder = args.folder, context = context, learning_rate=args.learning_rate)
    elif args.task == "render":
        render(args.steps, args.folder, args.algo, context=context)