import os
import zmq
import pybullet as p
import pybullet_data
import numpy as np
from gym import spaces
from util.color_print import printGreen, printBlue, printRed, printYellow
import gym
from environments.inmoov import inmoov
from environments.srl_env import SRLGymEnv
GRAVITY = -9.8
URDF_PATH = "/urdf_robot/"
RENDER_WIDTH, RENDER_HEIGHT = 224, 224


# python -m rl_baselines.train --env InmoovGymEnv-v0 --srl-model ground_truth --algo ppo2 --log-dir logs/
# --num-timesteps 200000

def getGlobals():
    """
    :return: (dict)
    """
    return globals()


class InmoovGymEnv(SRLGymEnv):
    def __init__(self, urdf_path=URDF_PATH, max_steps=1000,
                 env_rank=0,
                 srl_pipe=None,
                 action_repeat=1, srl_model="ground_truth",
                 seed=0, debug_mode=False, render=False,
                 positional_control=True,
                 discrete=True,
                 **kwargs):
        """

        :param urdf_path:
        :param max_steps:
        :param env_rank:
        :param srl_pipe:
        :param action_repeat:
        :param srl_model:
        :param seed: (int) the random seed for the GymEnv
        :param debug_mode: (bool) if True, the GUI will show up during the training (or for dug purpose)
        :param kwargs:
        """
        super(InmoovGymEnv, self).__init__(srl_model=srl_model,
                                           relative_pos=True,
                                           env_rank=env_rank,
                                           srl_pipe=srl_pipe)

        self.seed(seed)
        self.urdf_path = urdf_path

        self._observation = None
        self.debug_mode = debug_mode
        self._inmoov = None
        self._observation = None
        self.action_repeat = action_repeat
        self._inmoov_id = -1
        self._tomato_id = -1
        self.max_steps = max_steps
        self._step_counter = 0
        self._render = render
        # for more information, please refer to the function _get_tomato_pos
        self._tomato_link_id = 3

        self.positional_control = positional_control
        self.srl_model = srl_model
        self.camera_target_pos = (0.0, 0.0, 1.0)
        self._width = RENDER_WIDTH
        self._height = RENDER_HEIGHT
        self.terminated = False
        self.n_contacts = 0
        self.state_dim = self.getGroundTruthDim()
        self._first_reset_flag = False

        if debug_mode:
            client_id = p.connect(p.SHARED_MEMORY)
            if client_id < 0:
                p.connect(p.GUI)
            p.resetDebugVisualizerCamera(5., 180, -41, [0.52, -0.2, -0.33])
        else:
            p.connect(p.DIRECT)

        # TODO: here, we only use, for the moment, discrete action

        if self.srl_model == "raw_pixels":
            self.observation_space = spaces.Box(low=0, high=255, shape=(self._height, self._width, 3), dtype=np.uint8)
        else:  # Todo: the only possible observation for srl_model is ground truth
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        self.reset()
        # if positional control, then let the control to be discrete, continue otherwise
        if self.positional_control:
            self.action_space = spaces.Discrete(6) if discrete else spaces.Box(np.array([-1,-1, 0]), np.array([1,1,2]))
        else:
            low = np.array(self._inmoov.joint_lower_limits)
            high = np.array(self._inmoov.joint_upper_limits)
            self.action_space = spaces.Box(low, high, dtype=np.float32)

    def reset(self):
        self._step_counter = 0
        self.terminated = False
        self.n_contacts = 0
        if not self._first_reset_flag:
            # print('first reset,loading urdf...')
            p.resetSimulation()
            self._first_reset_flag = True
            p.setPhysicsEngineParameter(numSolverIterations=150)
            p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"), [0, 0, 0])
            p.setGravity(0, 0, GRAVITY)

            self._inmoov = inmoov.Inmoov(urdf_path=self.urdf_path, positional_control=self.positional_control)
            self._inmoov_id = self._inmoov.inmoov_id

            self._tomato_id = p.loadURDF(os.path.join(self.urdf_path, "tomato_plant.urdf"), [0.4, 0.4, 0.5])
        print('fast reset,resetting robot joints')
        self._inmoov.reset_joints()
        # p.resetSimulation()
        # p.setPhysicsEngineParameter(numSolverIterations=150)
        # p.setGravity(0., 0., GRAVITY)
        return self.get_observation()

    def _get_effector_pos(self):
        return self._inmoov.get_effector_pos()

    def _get_tomato_pos(self):
        # 0: the tomato at the bottom
        # 1,2 : the two little tomatoes on top
        # 3: the only lonely tomato on the top
        tomato_pos = p.getLinkState(self._tomato_id, self._tomato_link_id)[0]
        return np.array(tomato_pos)

    def ground_truth(self):
        # a relative position
        if self.positional_control:
            return self._get_effector_pos() - self._get_tomato_pos()
        else:
            # control by joints state
            return self._inmoov.getGroundTruth()

    def getSRLState(self, observation=None):
        # TODO: raw pixels
        if self.srl_model == "ground_truth":
            return self.ground_truth()

    def getGroundTruth(self):
        return self.ground_truth()

    def getTargetPos(self):
        return self._get_tomato_pos()

    @staticmethod
    def getGroundTruthDim():
        return 3

    def _reward(self):
        distance = np.linalg.norm(self._get_effector_pos() - self._get_tomato_pos(), 2)
        # printYellow("The distance between target and effector: {:.2f}".format(distance))
        return - distance

    def _termination(self):
        """
        :return: (bool) whether an episode is terminated
        """
        if self.terminated or self._step_counter > self.max_steps:
            return True
        return False

    def effector_position(self):
        return self._inmoov.getGroundTruth()

    def guided_step(self):
        """
        Effect a guided step towards the target
        :return:
        """
        tomato_pos = self._get_tomato_pos()
        eff_pos = self._get_effector_pos()
        action = tomato_pos - eff_pos
        self._inmoov.apply_action_pos(action)

    def server_step(self, action):
        # directly control by the position, whether the joint poses or the end of the effector
        if self.positional_control:
            self._inmoov.apply_position_pos(action)
            p.stepSimulation()
            self._step_counter += 1
        else:  # control by joint, continual control
            assert len(action) == self.action_space.shape[0]
            self._inmoov.apply_action_joints(action)
            self._step_counter += 1
        reward = self._reward()
        obs = self.get_observation()
        done = self._termination()
        infos = {}
        robot_view = self._inmoov.robot_render()
        left_px, right_px = robot_view[0][...,:3], robot_view[1][...,:3]
        px = np.array([left_px, right_px])
        return np.array(obs), reward, done, infos, px, self._get_effector_pos(),

    def step(self, action):
        # TODO: bug might be here
        # if action == 5:
        #     self.guided_step()
        if self.positional_control:
            if action is None:
                action = np.array([0, 0, 0])
            dv = 1.2
            dx = [-dv, dv, 0, 0, 0, 0][action]
            dy = [0, 0, -dv, dv, 0, 0][action]
            dz = [0, 0, 0, 0, -dv, dv][action]
            action = [dx, dy, dz]
            for i in range(self.action_repeat):
                self._inmoov.apply_action_pos(action)
                p.stepSimulation()
            self._step_counter += 1
        else:  # control by joint, continual control
            assert len(action) == self.action_space.shape[0]
            for i in range(self.action_repeat):
                self._inmoov.apply_action_joints(action)
        reward = self._reward()
        obs = self.get_observation()
        done = self._termination()
        infos = {}
        if self._render:
            self._inmoov.robot_render()

        return np.array(obs), reward, done, infos
        # printGreen(action)
        # printYellow(self._inmoov.getGroundTruth())
        # tt()
        # self._observation = self.render(mode='rgb')
        # reward = self._reward()
        # done = self._termination()
        # self.obs[:], rewards, self.dones, infos
        # return np.array(self._observation), reward, done, {}

    def get_observation(self):
        if self.srl_model == "raw_pixels":
            self._observation = self.render(mode="rgb")
            return self._observation
        elif self.srl_model == "ground_truth":
            self._observation = self.ground_truth()
            return self.ground_truth()
        else:
            raise NotImplementedError


    def render(self, mode='rgb'):
        """
        return the RBG image
        :param mode: 'robot' to get the robot view
        :return:
        """
        camera_target_position = self.camera_target_pos
        view_matrix1 = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=camera_target_position,
            distance=2.,
            yaw=145,  # 145 degree
            pitch=-36,  # -36 degree
            roll=0,
            upAxisIndex=2
        )
        proj_matrix1 = p.computeProjectionMatrixFOV(
            fov=60, aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
            nearVal=0.1, farVal=100.0)
        (_, _, px, depth, mask) = p.getCameraImage(
            width=RENDER_WIDTH, height=RENDER_HEIGHT, viewMatrix=view_matrix1,
            projectionMatrix=proj_matrix1, renderer=p.ER_TINY_RENDERER)
        px, depth, mask = np.array(px), np.array(depth), np.array(mask)


        return px

    def close(self):
        return
