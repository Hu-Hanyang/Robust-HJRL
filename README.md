# Robust Reinforcement Learning with Hamilton-Jacobi Reachability Analysis

This repository is deeply-based on the [`gym-pygbullet-drones`](https://github.com/utiasDSL/gym-pybullet-drones) repository and the [`optimized_dp`](https://github.com/SFU-MARS/optimized_dp). Please refer to them for more information.

## Installation
Tested on Intel x64/Ubuntu 18.04.



```sh
git clone git@github.com:Hu-Hanyang/gym-pybullet-drones.git
cd gym-pybullet-drones/

conda env create -f environment.yml  # now the packages in the environment.yml have conflicts, please install odp env first and then install other required packegs in the drone env.
conda activate drones

pip3 install --upgrade pip
pip3 install -e .  

cd gym_pybullet_drones/hj_distbs
pip3 install -e .
```

## Use

### Environments with Disturbances
There are 3 files used for the training of the quadrotor in the environment with disturbances. They are `BaseDistb.py`, `BaseDistbRL.py` and `HoverDistb.py` (all in `gym-pybullet-drones/envs/`). 

To train the policy using PPO, please use:
```
python learn_distb.py
```

The default one is 0-disturbance.


### Working Logs
#### Older changes
MICHAEL MODIFICATION for using the CleanRL library
`action = np.atleast_2d(action)`
in the file `BaseDistbRL.py`.

#### 3.27 
I suspect there is something wrong with the env because training still has no effect.

#### 3.28
Somethings to do:
1. Check the thrust and torque calculation in the current env compared with the original one: check the magnitude;
2. Transfer the PWMs to the current env to replace the rmps;
3. As for the original env, remove all the noisy things and train the policy again;
4. Implement different method: use Boltzmann distribution with one fixed disturbance level;
5. Revise the draft: point out the former drawbacks and corresponding our method to fix those problems.

#### 4.2
Implement PWM control signal.

#### 4.3
Train new reward function (add rpy penalty) in env with distb=0.0 and no initial randomization: training_results_sb3/fixed-distb_level_0.0/seed_42/save-2024.04.03_15:54
training_results_sb3/boltzmann/seed_42/save-2024.04.03_16:25

#### 4.4 
All failed.

#### 4.10
Bug solved: reset the distb level in the def step() function.

The reason why the performance is not comparable with the one used in the disturbance-CrazyFlie-simulation is that:
1. The lack of observation_noise
2. The lack of motor_thrust_noise