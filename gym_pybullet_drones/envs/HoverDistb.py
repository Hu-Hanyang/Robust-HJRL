import numpy as np

from gym_pybullet_drones.envs.BaseDistbRL import BaseDistbRLEnv
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class HoverDistbEnv(BaseDistbRLEnv):
    """Single agent RL problem: hover at position."""

    ################################################################################
    
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 disturbance_type = None,
                 distb_level: float=0.0,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 200,
                 ctrl_freq: int = 100,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM
                 ):
        """Initialization of a single agent RL environment.

        Using the generic single agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        disturbance_type : str, optional
            The type of disturbance to be applied to the drones [None, 'fixed', 'boltzmann', 'random', 'rarl', 'rarl-population'].
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """
        self.TARGET_POS = np.array([0,0,1])
        self.EPISODE_LEN_SEC = 8
        # Set the limits for states
        self.rp_limit = 60 * self.DEG2RAD  # rad
        self.rpy_dot_limit = 300 * self.DEG2RAD  # rad/s
        self.z_lim = 0.05  # m

        # Set the penalties
        self.penalty_action =1e-4
        self.penalty_angle_rate = 1e-3
        self.penalty_terminal = 1e2

        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         disturbance_type=disturbance_type,
                         distb_level=distb_level,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act
                         )

    ################################################################################
    
    def _computeReward(self, clipped_action):
        # TODO: transfer our former _computeReward method here, done
        """Computes the current reward value.

        Parameters
        ----------
        clipped_action : ndarray | dict[..]
            The input action for one or more drones.

        Returns
        -------
        float
            The reward.

        """
        state = self._getDroneStateVector(0)  # state.shape = (20,) state = [pos, quat, rpy, vel, ang_vel, last_clipped_action]
        
        normed_clipped_a = 0.5 * (np.clip(clipped_action, -1, 1) + 1)

        penalty_action = self.penalty_action * np.linalg.norm(normed_clipped_a)
        penalty_rpy_dot = self.penalty_angle_rate * np.linalg.norm(self.drone.rpy_dot)
        penalty_terminal = self.penalty_terminal if self._computeTerminated() else 0.  # Hanyang: try larger crash penalty

        penalties = np.sum([penalty_action, penalty_rpy_dot, penalty_terminal])
        dist = np.linalg.norm(state[0:3] - self.TARGET_POS)
        reward = -dist - penalties

        return reward

    ################################################################################
    
    def _computeTerminated(self):
        # TODO: transfer our former _computeTerminated method here, done
        """Computes the current done value.
        done = True if either the rp or rpy_dot or z hits the limits

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        state = self._getDroneStateVector(0)  # state.shape = (20,) state = [pos, quat, rpy, vel, ang_vel, last_clipped_action]
        position_limit = abs(state[0]) > 1.5 or abs(state[1]) > 1.5 or state[2] > 2.0
        rp = state[7:9]  # rad
        rp_limit = rp[np.abs(rp) > self.rp_limit].any()
        rpy_dot = state[13:16]  # rad/s
        rpy_dot_limit = rpy_dot[np.abs(rpy_dot) > self.rpy_dot_limit].any()
        z = state[2]  # m
        z_limit = z <= self.z_lim

        done = True if position_limit or rp_limit or rpy_dot_limit or z_limit else False

        return done
        
    ################################################################################
    
    def _computeTruncated(self):
        # TODO: not sure correct or not
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out.

        """
        state = self._getDroneStateVector(0)
        # if (abs(state[0]) > 1.5 or abs(state[1]) > 1.5 or state[2] > 2.0 # Truncate when the drone is too far away
        #      or abs(state[7]) > .4 or abs(state[8]) > .4 # Truncate when the drone is too tilted
        # ):
        #     return True
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False

    ################################################################################
    
    def _computeInfo(self):
        # TODO: transfer our former _computeInfo method here, done
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        info = {}
        # info['disturbance_level'] = self.distb_level
        info["answer"] = 42
        return info #### Calculated by the Deep Thought supercomputer in 7.5M years
