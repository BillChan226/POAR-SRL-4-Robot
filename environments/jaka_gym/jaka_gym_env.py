import os
import pybullet as p
import pybullet_data
import numpy as np
from gym import spaces
from util.color_print import printGreen, printBlue, printRed, printYellow
import gym
from environments.jaka_gym.jaka import Jaka
from environments.srl_env import SRLGymEnv

from state_representation.episode_saver import EpisodeSaver
from srl_zoo.preprocessing import getNChannels

import cv2
import math
GRAVITY = -9.8
RENDER_WIDTH, RENDER_HEIGHT = 224, 224
URDF_PATH = "/urdf_robot/jaka_urdf/jaka_local.urdf"
RELATIVE_POS = True  # Use relative position for ground truth

def getGlobals():
    """
    :return: (dict)
    """
    return globals()
# ssh -N -f -L localhost:6006:localhost:8097  tete@283a60820s.wicp.vip -p 17253
# python -m rl_baselines.train --env JakaButtonGymEnv-v0 --srl-model ground_truth --algo ppo2 --log-dir logs/ --num-cpu 16

class JakaButtonGymEnv(SRLGymEnv):
    def __init__(self, urdf_path=URDF_PATH, max_steps=1000,
                 env_rank=0, random_target=True,
                 srl_pipe=None, is_discrete=True,
                 action_repeat=1, srl_model="raw_pixels", multi_view = False,
                 control_mode = "position", learn_states=False, save_path='srl_zoo/data/',
                 seed=2, debug_mode=True, name="Jaka_button_obs_gym", max_distance=0.8, record_data=False, state_dim=-1, **_):
        super(JakaButtonGymEnv, self).__init__(srl_model=srl_model,
                                           relative_pos=True,
                                           env_rank=env_rank,
                                           srl_pipe=srl_pipe)

        self.seed(seed)
        self.urdf_path = urdf_path
        self._random_target = random_target
        self._observation = None
        self.debug_mode = debug_mode
        self._observation = None
        self.action_repeat = action_repeat
        self._jaka_id = -1
        self._button_id= -1
        self.max_steps = max_steps
        self._step_counter = 0
        self._render = True
        self.position_control = (control_mode == "position")
        self.discrete_action = is_discrete
        self.srl_model = srl_model
        self.camera_target_pos = (0.316, -0.2, -0.1)
        self._width = RENDER_WIDTH
        self._height = RENDER_HEIGHT
        self.terminated = False
        self.n_contacts = 0
        self.state_dim = state_dim
        if self.srl_model == "ground_truth":
            self.state_dim = self.getGroundTruthDim()
        self._first_reset_flag = False
        self.saver = None
        self.multi_view = multi_view

        if record_data:
            self.saver = EpisodeSaver(name, max_distance, state_dim, globals_=getGlobals(), relative_pos=RELATIVE_POS,
                                      learn_states=learn_states, path=save_path)

        if debug_mode:
            self.debug_joints = []
            client_id = p.connect(p.SHARED_MEMORY)
            if client_id < 0:
                p.connect(p.GUI)
            p.resetDebugVisualizerCamera(5., 180, -41, [0.52, -0.2, -0.33])
        else:
            p.connect(p.DIRECT)
        self.action_space = spaces.Discrete(6)
        if self.srl_model == "raw_pixels":
            self.observation_space = spaces.Box(low=0, high=255, shape=(self._height, self._width, 3), dtype=np.uint8)
        else:  # Todo: the only possible observation for srl_model is ground truth
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)

        self.reset()

    def reset(self, generated_observation=None, state_override=None):
        self._step_counter = 0
        self.terminated = False
        self.n_contacts = 0
        self.distance = 0
        if not self._first_reset_flag:
            p.resetSimulation()
            self._first_reset_flag = True
            p.setPhysicsEngineParameter(numSolverIterations=150)
            p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"), [0, 0, 0])
            p.setGravity(0, 0, GRAVITY)

            self._jaka = Jaka(urdf_path=self.urdf_path, positional_control=self.position_control)
            self._jaka_id = self._jaka.jaka_id
            x_pos = 0.5
            y_pos = 0
            
            if self._random_target:
                x_pos += 0.15 * self.np_random.uniform(-1, 1)
                y_pos += 0.3 * self.np_random.uniform(-1, 1)

            self.button_uid = p.loadURDF("/urdf/simple_button.urdf", [x_pos, y_pos, 0])
            self.button_pos = np.array([x_pos, y_pos, 0])

        self._jaka.reset_joints()
        # p.resetSimulation()
        # p.setPhysicsEngineParameter(numSolverIterations=150)
        # p.setGravity(0., 0., GRAVITY)
        self._observation = self.get_observation()
        # if self.srl_model == "raw_pixels":
        #     self._observation = self._observation[0]
        if self.debug_mode:
            keys = list(self._jaka.joint_name.keys())
            keys.sort()
            for i, k in enumerate(keys):
                self.debug_joints.append(p.addUserDebugParameter(self._jaka.joint_name[k].decode(),
                                                                 self._jaka.joint_lower_limits[i],
                                                                 self._jaka.joint_upper_limits[i], 0.))
        self._observation = self.get_observation()
        if self.saver is not None:
            self.saver.reset(self._observation, self.getTargetPos(), self.getGroundTruth())
        #if self.srl_model != "raw_pixels":
        #    return self.getSRLState(self._observation)

        return self._observation

    def getTargetPos(self):
        #self.button_pos = np.around(p.getBasePositionAndOrientation(self.button_uid)[0],decimals=3)
        return self.button_pos

    def getGroundTruth(self):
        return self._jaka.getGroundTruth()

    def _reward(self):
        # TODO: reward bad definition
        #contact_points = p.getContactPoints(self._jaka_id, self.button_uid,1)
        #reward = int(len(contact_points) > 0)-0.01
        #if np.linalg.norm(self.getGroundTruth() - self.getTargetPos()) > 0.8 :
            #reward = -np.linalg.norm(self.getGroundTruth() - self.getTargetPos())
        #if contact_with_button:
            #self.terminated = True
        distance = np.linalg.norm(self.getGroundTruth() - self.getTargetPos(),2)
        #re = math.exp(-distance*10)
        re = - distance
        #if distance < 0.12:
        	#re += 1
        	#print('contact!')
        #distance2= np.linalg.norm(np.array([0,0,0]) - self.getTargetPos(),2)
 
        return re
        #return - np.linalg.norm(self.getGroundTruth() - self.getTargetPos())
        

    def _termination(self):
        if self.terminated or self._step_counter > self.max_steps:
            return True
        return False

    def getExtendedObservation(self):
        if getNChannels() > 3:
            self.multi_view = True
        observation = self.render("rgb_array")
        return observation

    def get_observation(self):
        if self.srl_model == "ground_truth":
            if self.relative_pos:
                return self.getGroundTruth() - self.getTargetPos()
            return self.getGroundTruth()
        elif self.srl_model == "raw_pixels":
            return self.getExtendedObservation()

        else:
            return self.getSRLState(self.getExtendedObservation())
            #return NotImplementedError()

    @staticmethod
    def getGroundTruthDim():
        return 3

    def step(self, action, generated_observation=None, action_proba=None, action_grid_walker=None):
        #np.random.rand(self.seed)
        old_pos = self.getGroundTruth()
        self.action = action  # For saver
        if self.position_control and self.discrete_action:
            dv = 0.2
            dx = [-dv, dv, 0, 0, 0, 0][action]
            dy = [0, 0, -dv, dv, 0, 0][action]
            dz = [0, 0, 0, 0, -dv, dv][action]
            action = [dx, dy, dz] + np.random.normal(size=3, scale=1/30, loc=0)
            #action = [dx, dy, dz]
            for _ in range(self.action_repeat):
                self._jaka.apply_action_pos(action)
                p.stepSimulation()
            new_pos = self.getGroundTruth()
            self.distance += np.linalg.norm(new_pos - old_pos,2)
            # to draw the line for trajectory
            p.addUserDebugLine(old_pos,new_pos,[1, 0, 0], 4.0)
            #print(self.distance)
            self._step_counter += 1
            reward = self._reward()
            obs = self.get_observation()
            done = self._termination()
            infos = {}
            self._observation = self.get_observation()
            if self.saver is not None:
                self.saver.step(np.array(obs), self.action, reward, done, self.getGroundTruth())
            return np.array(obs), reward, done, infos

    # def render(self, mode='channel_last', img_name=None):
    #     """
    #     You could change the Yaw Pitch Roll distance to change the view of the robot
    #     :param mode:
    #     :return:
    #     """
    #     camera_matrix = p.computeViewMatrixFromYawPitchRoll(
    #         cameraTargetPosition=self._jaka.robot_base_pos,
    #         distance=2.,
    #         yaw=145,  # 145 degree
    #         pitch=-40,  # -45 degree
    #         roll=0,
    #         upAxisIndex=2
    #     )
    #     proj_matrix1 = p.computeProjectionMatrixFOV(
    #         fov=60, aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
    #         nearVal=0.15, farVal=100.0)
    #     # depth: the depth camera, mask: mask on different body ID
    #     (_, _, px, depth, mask) = p.getCameraImage(
    #         width=RENDER_WIDTH, height=RENDER_HEIGHT, viewMatrix=camera_matrix,
    #         projectionMatrix=proj_matrix1, renderer=p.ER_TINY_RENDERER)
    #
    #     px, depth, mask = np.array(px), np.array(depth), np.array(mask)
    #
    #     px = px.reshape(RENDER_HEIGHT, RENDER_WIDTH, -1)
    #     depth = depth.reshape(RENDER_HEIGHT, RENDER_WIDTH, -1)
    #     mask = mask.reshape(RENDER_HEIGHT, RENDER_WIDTH, -1)
    #     px = px[..., :-1]
    #     if img_name is not None:
    #         cv2.imwrite(img_name, px[..., [2, 1, 0]])
    #     if mode != 'channel_last':
    #         # channel first
    #         px = np.transpose(px, [2, 0, 1])
    #     return px, depth, mask

    def render(self, mode='human', close=False):
        if mode != "rgb_array":
                return np.array([])
        camera_target_pos = self.camera_target_pos

        # if self.debug_mode:
        #     self._cam_dist = p.readUserDebugParameter(self.dist_slider)
        #     self._cam_yaw = p.readUserDebugParameter(self.yaw_slider)
        #     self._cam_pitch = p.readUserDebugParameter(self.pitch_slider)
        #     x = p.readUserDebugParameter(self.x_slider)
        #     y = p.readUserDebugParameter(self.y_slider)
        #     z = p.readUserDebugParameter(self.z_slider)
        #     camera_target_pos = (x, y, z)
        #     # self._cam_roll = p.readUserDebugParameter(self.roll_slider)

        # TODO: recompute view_matrix and proj_matrix only in debug mode
        view_matrix1 = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=camera_target_pos,
            distance=1.6,
            yaw=155,
            pitch=-36,
            roll=0,
            upAxisIndex=2)
        proj_matrix1 = p.computeProjectionMatrixFOV(
            fov=60, aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
            nearVal=0.1, farVal=100.0)
        (_, _, px1, _, _) = p.getCameraImage(
            width=RENDER_WIDTH, height=RENDER_HEIGHT, viewMatrix=view_matrix1,
            projectionMatrix=proj_matrix1, renderer=p.ER_TINY_RENDERER)
        rgb_array1 = np.array(px1)

        if self.multi_view:
            # adding a second camera on the other side of the robot
            view_matrix2 = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=(0.316, 0.316, -0.105),
                distance=1.05,
                yaw=32,
                pitch=-13,
                roll=0,
                upAxisIndex=2)
            proj_matrix2 = p.computeProjectionMatrixFOV(
                fov=60, aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
                nearVal=0.1, farVal=100.0)
            (_, _, px2, _, _) = p.getCameraImage(
                width=RENDER_WIDTH, height=RENDER_HEIGHT, viewMatrix=view_matrix2,
                projectionMatrix=proj_matrix2, renderer=self.renderer)
            rgb_array2 = np.array(px2)
            rgb_array_res = np.concatenate((rgb_array1[:, :, :3], rgb_array2[:, :, :3]), axis=2)
        else:
            rgb_array1 = rgb_array1.reshape(RENDER_WIDTH, RENDER_WIDTH, -1)
            rgb_array_res = rgb_array1[:, :, :3]
        return rgb_array_res

    def debug_step(self):
        current_joints = []
        for i in self.debug_joints:
            tmp_joint_control = p.readUserDebugParameter(i)
            current_joints.append(tmp_joint_control)
        for joint_state, joint_key in zip(current_joints, self._jaka.joints_key):
            p.resetJointState(self._jaka_id, joint_key, targetValue=joint_state)
        p.stepSimulation()


if __name__ == '__main__':
    jaka = JakaButtonGymEnv(debug_mode=True)
    i = 0
    import time
    # while i < 100:
    #     i += 1
    #     jaka.step(1)
    #     time.sleep(0.2)
    #     printYellow(jaka._reward())
    #     # jaka.render(img_name='trash/out{}.png'.format(i))
    #     # jaka._render(img_name='trash/out{}.png'.format(i))
    goal_conf = jaka.getTargetPos()
    effector_id = 5
    rest_poses = jaka._jaka.initial_joints_state  
    joint_ranges = [4] * len(jaka._jaka.joints_key)
    target_velocity = [0] * len(jaka._jaka.joints_key)
    pos = [-0.1,0.4,0]
    joint_poses = p.calculateInverseKinematics(jaka._jaka_id, effector_id, pos,
                                                       lowerLimits=jaka._jaka.joint_lower_limits,
                                                       upperLimits=jaka._jaka.joint_upper_limits,
                                                       jointRanges=joint_ranges,
                                                       restPoses=rest_poses
                                                       )
    p.setJointMotorControlArray(jaka._jaka_id, jaka._jaka.joints_key,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=joint_poses,
                                    targetVelocities=target_velocity,
                                    forces=jaka._jaka.jointMaxForce)
    p.stepSimulation()
    #print(jaka.getGroundTruth())
    

    """while i < 1e6:
        i += 1
        jaka.debug_step()
        #printYellow(jaka._reward())
        printYellow(jaka.getTargetPos())"""
