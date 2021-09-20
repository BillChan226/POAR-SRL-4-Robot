import os
import pybullet as p
import pybullet_data
import numpy as np
from gym import spaces
from util.color_print import printGreen, printBlue, printRed, printYellow
import gym
from environments.jaka_gym.jaka import Jaka
from environments.srl_env import SRLGymEnv
import cv2
import math
import time
from environments.jaka_gym.quatation import Quaternion
GRAVITY = -9.8
RENDER_WIDTH, RENDER_HEIGHT = 224, 224
URDF_PATH = "/urdf_robot/jaka_urdf/jaka_local.urdf"


def getGlobals():
    """
    :return: (dict)
    """
    return globals()
# ssh -N -f -L localhost:6006:localhost:8097  tete@283a60820s.wicp.vip -p 17253

# python -m rl_baselines.train --env ContinousObsGymEnv-v0 --srl-model ground_truth --algo sac --continuous-actions --log-dir logs/zhixian --num-cpu 16

class ContinousObsGymEnv(SRLGymEnv):
    def __init__(self, urdf_path=URDF_PATH, max_steps=2000,
                 env_rank=0, random_target=True,
                 srl_pipe=None, is_discrete=False, random_obs=False,
                 action_repeat=1, srl_model="ground_truth",
                 control_mode = "joints",
                 seed=0, debug_mode=False, **_):
        super(ContinousObsGymEnv, self).__init__(srl_model=srl_model,
                                           relative_pos=True,
                                           env_rank=env_rank,
                                           srl_pipe=srl_pipe)

        self.seed(seed)
        self.urdf_path = urdf_path
        self._random_target = random_target
        self._observation = None
        self.debug_mode = debug_mode
        self._inmoov = None
        self._observation = None
        self.action_repeat = action_repeat
        self._jaka_id = -1
        self._random_obs = random_obs
        self._button_id= -1
        self.max_steps = max_steps
        self._step_counter = 0
        self._episode_counter = 0
        self._render = False
        self.position_control = (control_mode == "position")
        self.discrete_action = is_discrete
        self.srl_model = srl_model
        self.camera_target_pos = (0.0, 0.0, 1.0)
        self._width = RENDER_WIDTH
        self._height = RENDER_HEIGHT
        self.terminated = False
        self.n_contacts = 0
        self.state_dim = self.getGroundTruthDim()
        self._first_reset_flag = False
        self.wangyou = 1
        self.plot = True
        self.time1 = time.time()
        self.time2 = time.time()

        if debug_mode:
            self.debug_joints = []
            client_id = p.connect(p.SHARED_MEMORY)
            if client_id < 0:
                p.connect(p.GUI)
            p.resetDebugVisualizerCamera(5., 180, -41, [0.52, -0.5, 0.5])
        else:
            p.connect(p.DIRECT)
        if self.position_control and self.discrete_action:
            self.action_space = spaces.Discrete(6)
        else:
            self.action_space = spaces.Discrete(10)
        if not self.discrete_action:
            #self.action_space = spaces.Box(np.array([-3.14,-3.14,-3.14,-3.14,-3.14,-3.14]),np.array([3.14,3.14,3.14,3.14,3.14,3.14]),dtype=np.float64)
            self.action_space = spaces.Box(np.array([-3.14,-3.14,-3.14,-3.14,-3.14]),np.array([3.14,3.14,3.14,3.14,3.14]),dtype=np.float64)
        if self.srl_model == "raw_pixels":
            self.observation_space = spaces.Box(low=0, high=255, shape=(self._height, self._width, 3), dtype=np.uint8)
        elif self.position_control:  # Todo: the only possible observation for srl_model is ground truth
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        else:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)

        self.reset()

    def reset(self, generated_observation=None, state_override=None):
        self._step_counter = 0
        self._episode_counter += 1
        self.terminated = False
        self.n_contacts = 0
        self.distance = 0
        if not self._first_reset_flag:
            p.resetSimulation()
            self._first_reset_flag = True
            p.setPhysicsEngineParameter(numSolverIterations=500)
            self.plane = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"), [0, 0, 0])

            p.setGravity(0, 0, GRAVITY)

            self._jaka = Jaka(urdf_path=self.urdf_path, positional_control=self.position_control)
            self._jaka_id = self._jaka.jaka_id
            x_pos = -0.1
            y_pos = 0.5
            
           
            #obs_x, obs_y = 0.3, 0.3
            obs_x, obs_y = 1, 0.3
            self.obs_x, self.obs_y = obs_x, obs_y
            self.button_uid = p.loadURDF("/urdf/simple_button.urdf", [x_pos, y_pos, 0])
            self.obstacle_id = p.loadURDF("/urdf/long_cylinder.urdf", [obs_x, obs_y, 0])
            self.button_pos = np.array([x_pos, y_pos, 0])
            self.obstacle_pos = np.array([obs_x, obs_y, 0])
        else:
            x_pos = -0.1
            y_pos = 0.5
            self.time1 = time.time()
            self.plot = True
            p.removeAllUserDebugItems()
            a = self.np_random.uniform(-1, 1)
            if self._random_obs:
            #if 0:
                x_pos = self.np_random.uniform(-0.2, 0.2)
                y_pos = self.np_random.uniform(0.4, 0.5)

            obs_x, obs_y = 1, 0.3
            if self._random_obs:
            #if 0:
                #obs_x = x_pos self.np_random.uniform(-1, 0.7)
                if a > 0:
                    obs_x = x_pos+self.np_random.uniform(0.3, 0.4)
                else:
                    obs_x = x_pos-self.np_random.uniform(0.3, 0.4)
                #obs_x = x_pos+self.np_random.uniform(0.3, 0.4)
                obs_y = y_pos + self.np_random.uniform(-0.1, 0)
            self.x_pos, self.y_pos = x_pos,y_pos
            self.obs_x, self.obs_y = obs_x, obs_y
            self.button_pos = np.array([x_pos, y_pos, 0])
            self.obstacle_pos = np.array([obs_x, obs_y, 0])
            p.resetBasePositionAndOrientation(self.button_uid, self.button_pos,
                                          [0.000000, 0.000000, 0.000000, 1.000000])
            p.resetBasePositionAndOrientation(self.obstacle_id, self.obstacle_pos,
                                          [0.000000, 0.000000, 0.000000, 1.000000])
        self._jaka.reset_joints()
        # p.resetSimulation()
        # p.setPhysicsEngineParameter(numSolverIterations=150)
        # p.setGravity(0., 0., GRAVITY)
        if self.debug_mode:
            keys = list(self._jaka.joint_name.keys())
            keys.sort()
            for i, k in enumerate(keys):
                self.debug_joints.append(p.addUserDebugParameter(self._jaka.joint_name[k].decode(),
                                                                 self._jaka.joint_lower_limits[i],
                                                                 self._jaka.joint_upper_limits[i], 0.))
        self._observation = self.get_observation()
        # if self.srl_model == "raw_pixels":
        #     self._observation = self._observation[0]
        return self._observation

    def getTargetPos(self):
        return self.button_pos
 
    def getGroundTruth(self):
        return p.getLinkState(self._jaka_id, 5)[0]
        #return self._jaka.getGroundTruth()

    def _reward(self):
        #r = - np.linalg.norm(self.getGroundTruth() - self.getTargetPos())
        d = np.linalg.norm(self.getGroundTruth() - self.getTargetPos(),2)
        do = np.linalg.norm(self.getGroundTruth() - self.obstacle_pos,2)
      
        #r = math.exp(0.7/d)
        #a=0.7
        if d < 0.4:
            r = 1/(5*d) -0.9
        if d < 0.1:
            #a = np.array(p.getLinkState(self._jaka_id, 5)[1])
            #q = Quaternion(a[3],a[0],a[1],a[2])
            #qt = Quaternion(-0.1878,0.0257,-0.109,0.9757)
            #e = np.linalg.norm(np.array(q.torpy())-np.array(qt.torpy()))
            dp = np.linalg.norm(np.array(p.getLinkState(self._jaka_id, 5)[1])-[0.0257,-0.109,0.9757,-0.1878])
            #r += 1.5-math.log(e,150)
            r += 1.5-dp
        else:
            r = -d
        #if do < 0.3:
            #r = 0.5 - 1/(5*do)
        #if d < 0.3:
        #   r = math.exp(0.1/d)-0.5
        #r += -(0.2/do)**2
        #r = math.exp(0.1/d)
        #contact_button = 20 * int(len(p.getContactPoints(self._jaka_id, self.button_uid)) > 0)
        #if self._episode_counter > 250:
        contact_obs = -40*len(p.getContactPoints(self._jaka_id, self.obstacle_id))
        #else:
            #contact_obs = -10*len(p.getContactPoints(self._jaka_id, self.obstacle_id))
        #if contact_obs < 0:
            #self.terminated = True
            #r += -1000
        #contact_obs += -20*len(p.getContactPoints(self._jaka_id, self.plane))
        #for i in range(6):
            #contact_obs += -6*len(p.getContactPoints(self._jaka_id, linkIndexA=i))
        
        r += contact_obs
        
        #r += -0.2
        #if np.linalg.norm(self.getGroundTruth() - self.getTargetPos()) < 0.5:
            #r += 0.2
        return r

    def _termination(self):
        if self.terminated or self._step_counter > self.max_steps:
            return True
        return False

    def get_observation(self):
        if self.srl_model == "ground_truth":
            if self.position_control:
            #if self.control_mode == "position":
                gr=self.getTargetPos() - self.getGroundTruth() 
                return np.append(gr,self.getGroundTruth() - self.obstacle_pos)                           
                #return np.append(gr,self.getGroundTruth() - self.obstacle_pos)+np.random.normal(size=6, scale=1/20, loc=0)
            else:
                gr=self.getTargetPos() - self.getGroundTruth()
                joints_state = list(p.getJointStates(self._jaka_id, self._jaka.joints_key))
                joints = [x[0] for x in joints_state]
                ob = np.append(gr,self.getGroundTruth() - self.obstacle_pos)
                return np.append(ob,joints)
            return self.getGroundTruth()
        elif self.srl_model == "raw_pixels":
            return self.render()[0]
        else:
            return NotImplementedError()

    @staticmethod
    def getGroundTruthDim():
            return 12

    def step(self, action, generated_observation=None, action_proba=None, action_grid_walker=None):
        sinus_moving = False
        zhixian = True
        random_moving = False

        assert int(sinus_moving) + int(random_moving) < 2

        if sinus_moving:

            # p.loadURDF()

            maxx, maxy = self.obs_x, self.obs_y

            time_ratio = np.pi*2*(self._step_counter / self.max_steps)

            obs_x, obs_y = np.cos(time_ratio), maxy

            p.resetBasePositionAndOrientation(self.obstacle_id, [obs_x, obs_y, 0], [0,0,0,1])

        if random_moving:

            xnow, ynow = self.obs_x, self.obs_y

            fields_x = [0.3-0.2, 0.3+0.2]

            fields_y = [0.4-0.3, 0.4+0.2]

            obs_step = 0.1

            move_x = np.random.uniform(-0.1,0.1)

            move_y = np.sqrt(obs_step**2-move_x**2) * (2*(np.random.uniform()>0.5)-1)

            self.obs_x = np.clip(xnow + move_x, fields_x[0], fields_x[1])

            self.obs_y = np.clip(ynow + move_y, fields_y[0], fields_y[1])

            p.resetBasePositionAndOrientation(self.obstacle_id, [self.obs_x, self.obs_y, 0], [0, 0, 0, 1])

        if zhixian:

            if self.obs_x < -1:
                self.wangyou = - 1
            if self.obs_x > 1:
                self.wangyou = 1
            if self.wangyou > 0:
                self.obs_x += -0.002
            else:
                self.obs_x += 0.002
            self.obstacle_pos = np.array([self.obs_x, self.obs_y, 0])
            p.resetBasePositionAndOrientation(self.obstacle_id, [self.obs_x, self.obs_y, 0], [0, 0, 0, 1])
        old_pos = self.getGroundTruth()
        if self.position_control and self.discrete_action:
            dv = 0.2
            dx = [-dv, dv, 0, 0, 0, 0][action]
            dy = [0, 0, -dv, dv, 0, 0][action]
            dz = [0, 0, 0, 0, -dv, dv][action]
            action = [dx, dy, dz]+np.random.normal(size=3, scale=1/20, loc=0)
            #action = [dx, dy, dz]
            for _ in range(self.action_repeat):
                self._jaka.apply_action_pos(action)
                p.stepSimulation()
            self._step_counter += 1
            reward = self._reward()
            obs = self.get_observation()
            done = self._termination()
            infos = {}
        elif self.discrete_action:
            #joints[action//2] = math.pow(-1,action%2)*0.075
            joints = [0,0,0,0,0,0]
            joints_state = list(p.getJointStates(self._jaka_id, [0,1,2,3,4,5]))
            joints = [x[0] for x in joints_state]
            joints[action//2] += math.pow(-1,action%2)*0.05
            #print(joints_state)
            self._jaka.apply_action_joints(joints)
            p.stepSimulation()
            self._step_counter += 1
            reward = self._reward()
            obs = self.get_observation()
            done = self._termination()
            infos = {}
        else:
            joints = np.append(action,0)
            #joints = action
            self._jaka.apply_action_joints(joints)
            p.stepSimulation()
            self._step_counter += 1
            reward = self._reward()
            obs = self.get_observation()
            done = self._termination()
            infos = {}
        if self.debug_mode:
            contact_obs = len(p.getContactPoints(self._jaka_id, self.obstacle_id))
            p.resetBasePositionAndOrientation(self.button_uid, self.button_pos,
                                          [0.000000, 0.000000, 0.000000, 1.000000])
            if np.linalg.norm(self.getTargetPos() - self.getGroundTruth(),2) < 0.1:
                #print(np.linalg.norm(np.array(p.getLinkState(self._jaka_id, 5)[1])-[0.0257,-0.109,0.9757,-0.1878]))
                a = np.array(p.getLinkState(self._jaka_id, 5)[1])
                q = Quaternion(a[3],a[0],a[1],a[2])
                qt = Quaternion(-0.1878,0.0257,-0.109,0.9757)
                print(np.array(q.torpy())-np.array(qt.torpy()))
            time.sleep(0.01)
            if contact_obs > 0:
                print('no')
        if 0:
            new_pos = self.getGroundTruth()
            self.distance += np.linalg.norm(new_pos - old_pos,2)
            p.addUserDebugLine(old_pos,new_pos,[1, 0, 0], 4.0)
            if np.linalg.norm(self.getGroundTruth() - self.getTargetPos()) <0.1 and self.plot:
                print(self.distance)
                print(self._step_counter)
                self.time2 = time.time()
                self.plot = False
                print(self.time2-self.time1)
            contact_obs = -6*len(p.getContactPoints(self._jaka_id, self.obstacle_id))
            contact_obs += -6*len(p.getContactPoints(self._jaka_id, self.plane))
            for i in range(6):
                contact_obs += -6*len(p.getContactPoints(self._jaka_id, linkIndexA=i))
            if contact_obs>0:
                print('no')
            
        return np.array(obs), reward, done, infos

    def render(self, mode='channel_last', img_name=None):
        """
        You could change the Yaw Pitch Roll distance to change the view of the robot
        :param mode:
        :return:
        """
        camera_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self._jaka.robot_base_pos,
            distance=2.,
            yaw=145,  # 145 degree
            pitch=-40,  # -45 degree
            roll=0,
            upAxisIndex=2
        )
        proj_matrix1 = p.computeProjectionMatrixFOV(
            fov=60, aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
            nearVal=0.15, farVal=100.0)
        # depth: the depth camera, mask: mask on different body ID
        (_, _, px, depth, mask) = p.getCameraImage(
            width=RENDER_WIDTH, height=RENDER_HEIGHT, viewMatrix=camera_matrix,
            projectionMatrix=proj_matrix1, renderer=p.ER_TINY_RENDERER)

        px, depth, mask = np.array(px), np.array(depth), np.array(mask)

        px = px.reshape(RENDER_HEIGHT, RENDER_WIDTH, -1)
        depth = depth.reshape(RENDER_HEIGHT, RENDER_WIDTH, -1)
        mask = mask.reshape(RENDER_HEIGHT, RENDER_WIDTH, -1)
        px = px[..., :-1]
        if img_name is not None:
            cv2.imwrite(img_name, px[..., [2, 1, 0]])
        if mode != 'channel_last':
            # channel first
            px = np.transpose(px, [2, 0, 1])
        return px, depth, mask

    def debug_step(self):
        current_joints = []
        for i in self.debug_joints:
            tmp_joint_control = p.readUserDebugParameter(i)
            current_joints.append(tmp_joint_control)
        for joint_state, joint_key in zip(current_joints, self._jaka.joints_key):
            p.resetJointState(self._jaka_id, joint_key, targetValue=joint_state)
        p.stepSimulation()




if __name__ == '__main__':
    jaka = ContinousObsGymEnv(debug_mode=True)
    i = 0
    import time


    for i in range(10000):
        #jaka.step(0)
        #jaka.step(3)
        #jaka.step(3)
        jaka.debug_step()
        #printYellow(jaka._reward())
        #print(np.linalg.norm(np.array(p.getLinkState(jaka._jaka_id, 5)[1])-[0.0257,-0.109,0.9757,-0.1878]))
        a = np.array(p.getLinkState(jaka._jaka_id, 5)[1])
        q = Quaternion(a[3],a[0],a[1],a[2])
        qt = Quaternion(-0.1878,0.0257,-0.109,0.9757)
        e = np.linalg.norm(np.array(q.torpy())-np.array(qt.torpy()))
        print(math.log(e,150))
        #e = qt.inverse() * q - Quaternion(1,0,0,0)
        #printYellow(e.mod())
        #printYellow(jaka.get_observation())
        time.sleep(0.1)

    #
    # while True:
    #
    #     i += 1
    #     jaka.step(0)
    #     jaka.step(3)
    #     time.sleep(0.2)
    #     printYellow(jaka._reward())
    #     # jaka.render(img_name='trash/out{}.png'.format(i))
    #     # jaka._render(img_name='trash/out{}.png'.format(i))
