import os
import pybullet as p
import pybullet_data
import numpy as np
import math
from gym import spaces
from util.color_print import printGreen, printBlue, printRed, printYellow
import gym
from environments.jaka_gym.jaka import Jaka
from environments.srl_env import SRLGymEnv
from collections import defaultdict, OrderedDict
import cv2
from environments.collision import get_collision_fn,set_position,get_collision_fn_1
#from environments.rrt import rrt
GRAVITY = -9.8
RENDER_WIDTH, RENDER_HEIGHT = 224, 224
URDF_PATH = "/urdf_robot/jaka_urdf/jaka_local.urdf"

def getGlobals():
    """
    :return: (dict)
    """
    return globals()
# ssh -N -f -L localhost:6006:localhost:8097  tete@283a60820s.wicp.vip -p 17253
# python -m rl_baselines.train --env JakaButtonGymEnv-v0 --srl-model ground_truth --algo ppo2 --log-dir logs/ --num-cpu 16

class JakaButtonObsGymEnv(SRLGymEnv):
    def __init__(self, urdf_path=URDF_PATH, max_steps=1000,
                 env_rank=0, random_target=False,
                 srl_pipe=None, is_discrete=True,
                 action_repeat=1, srl_model="ground_truth",
                 control_mode = "position",
                 seed=2, debug_mode=False, **_):
        super(JakaButtonObsGymEnv, self).__init__(srl_model=srl_model,
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
        self._button_id= -1
        self.max_steps = max_steps
        self._step_counter = 0
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

        if debug_mode:
            client_id = p.connect(p.SHARED_MEMORY)
            if client_id < 0:
                p.connect(p.GUI)
            p.resetDebugVisualizerCamera(5., 180, -41, [0.52, -0.5, 0.5])
        else:
            p.connect(p.DIRECT)
        self.action_space = spaces.Discrete(6)
        if self.srl_model == "raw_pixels":
            self.observation_space = spaces.Box(low=0, high=255, shape=(self._height, self._width, 3), dtype=np.uint8)
        else:  # Todo: the only possible observation for srl_model is ground truth
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        self.reset()

    def reset(self, generated_observation=None, state_override=None):
        self._step_counter = 0
        self.terminated = False
        self.n_contacts = 0
        if not self._first_reset_flag:
            p.resetSimulation()
            self._first_reset_flag = True
            p.setPhysicsEngineParameter(numSolverIterations=150)
            self.plane = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"), [0, 0, 0])

            p.setGravity(0, 0, GRAVITY)

            self._jaka = Jaka(urdf_path=self.urdf_path, positional_control=self.position_control)
            self._jaka_id = self._jaka.jaka_id
            self.targetjoints = self._jaka.gettargetjoints([0.3,0.3,0])
            x_pos = -0.1
            y_pos = 0.4
            
            if self._random_target:
                x_pos += 0.15 * self.np_random.uniform(-1, 1)
                y_pos += 0.3 * self.np_random.uniform(-1, 1)
            obs_x, obs_y = 0.3, 0.3
            self.obs_x, self.obs_y = 0.3, 0.3
            self.button_uid = p.loadURDF("/urdf/simple_button.urdf", [x_pos, y_pos, 0])
            self.obstacle_id = p.loadURDF("/urdf/long_cylinder.urdf", [obs_x, obs_y, 0])
            self.button_pos = np.array([x_pos, y_pos, 0])
            self.obstacle_pos = np.array([obs_x, obs_y, 0])

        self._jaka.reset_joints()
        # p.resetSimulation()
        # p.setPhysicsEngineParameter(numSolverIterations=150)
        # p.setGravity(0., 0., GRAVITY)
        self._observation = self.get_observation()
        # if self.srl_model == "raw_pixels":
        #     self._observation = self._observation[0]
        return self._observation

    def getTargetPos(self):
        return self.button_pos
 
    def getGroundTruth(self):
        return self._jaka.getGroundTruth()

    def _reward(self):
        r = - np.linalg.norm(self.getGroundTruth() - self.getTargetPos())
        #contact_button = 20 * int(len(p.getContactPoints(self._jaka_id, self.button_uid)) > 0)
        contact_obs = - 5*len(p.getContactPoints(self._jaka_id, self.obstacle_id))
        
        r += contact_obs
        return r

    def _termination(self):
        if self.terminated or self._step_counter > self.max_steps:
            return True
        return False

    def get_observation(self):
        if self.srl_model == "ground_truth":
            if self.relative_pos:
                gr=self.getGroundTruth() - self.getTargetPos()
                np.append(gr,self.getGroundTruth() - self.obstacle_pos)
                
                return np.append(gr,self.getGroundTruth() - self.obstacle_pos)
            return self.getGroundTruth()
        elif self.srl_model == "raw_pixels":
            return self.render()[0]
        else:
            return NotImplementedError()

    @staticmethod
    def getGroundTruthDim():
        return 6
    

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
                self.obs_x += -0.005
            else:
                self.obs_x += 0.005
            p.resetBasePositionAndOrientation(self.obstacle_id, [self.obs_x, self.obs_y, 0], [0, 0, 0, 1])

        if self.position_control and self.discrete_action:
            dv = 0.2
            dx = [-dv, dv, 0, 0, 0, 0][action]
            dy = [0, 0, -dv, dv, 0, 0][action]
            dz = [0, 0, 0, 0, -dv, dv][action]
            #action = [dx, dy, dz]+np.random.normal(size=3, scale=1/30, loc=0)
            action = [dx, dy, dz]
            for _ in range(self.action_repeat):
                self._jaka.apply_action_pos(action)
                p.stepSimulation()
            self._step_counter += 1
            reward = self._reward()
            obs = self.get_observation()
            done = self._termination()
            infos = {}
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

#rrt:
def set_joint_positions(body, joints, values):
    assert len(joints) == len(values)
    for joint, value in zip(joints, values):
        p.resetJointState(body, joint, value)

def find_path(graph, start, end, path =[]):
    
    path = path + [start]
    
    if tuple(start) == tuple(end):
        return path 

    for node in graph[tuple(start)]:    
        node = tuple(node)
        #node = node.tolist
       # print(node)

        if node not in path:
            
            nextpath = find_path(graph, node, end, path) 

            if nextpath:  
                return nextpath


def create_step(p1,p2):
    
    #delta = 0.075
    delta = 0.1
    
    if np.linalg.norm(p2-p1) < delta:
        return p2

    else:
        direction_vector = (p2 - p1) / np.linalg.norm(p2-p1)
        return p1 + delta * direction_vector




def extend_rrt(q_near, q_rand):

    q_new = create_step(np.array(q_near), np.array(q_rand))
    q_new = q_new.tolist()
    
    if collision_fn(q_new):   
        pass
    else:
        """set_joint_positions(ur5, UR5_JOINT_INDICES, q_near)
        q_near_state = p.getLinkState(ur5, 3)[0]
        
        set_joint_positions(ur5, UR5_JOINT_INDICES, q_new)
        q_next_state = p.getLinkState(ur5, 3)[0]"""
        #jaka._jaka.apply_action_pos(q_new-q_near)
        set_position(jaka._jaka_id,[0,1,2,3,4,5],q_new,5)
        p.addUserDebugLine(q_near,q_new,[0, 1, 0], 1.0)
        #p.addUserDebugLine(q_near_state,q_next_state,[0, 1, 0], 1.0)
        
        return q_near, q_new
    
    return None, None

def dist_fn(p1, p2):
    return math.sqrt( ((p2[0] - p1[0]) ** 2) + ((p2[1] - p1[1]) ** 2) + ((p2[2] - p1[2]) ** 2) )

def rrt(max_iter, start_conf, end_conf):
    
    graph = defaultdict(list)
    graph_list = []
    graph_list.append(start_conf)
    
    bias = 0.05 * max_iter
    counter = 0
    
    for i in range(max_iter):
        if counter == bias:
            q_rand = end_conf
            counter = 0
        else:
            """rand_joint_3 = np.random.uniform(-np.pi, np.pi, 1)
            rand_joint_2 = np.random.uniform(2*-np.pi, 2*np.pi, 1)
            rand_joint_1 = np.random.uniform(2*-np.pi, 2*np.pi, 1)  """
            q_rand =  np.random.uniform(-1, 1, 3) 
            #q_rand_2 =  np.random.uniform(-2, 2, 1)
            #q_rand_3 =  np.random.uniform(0, 2, 1)
            counter += 1    
            
            #rand_conf = [rand_joint_1, rand_joint_2, rand_joint_3]
            #q_rand = [rand_conf[0][0], rand_conf[1][0], rand_conf[2][0]]
        
        dist = float('inf')
        for q in graph_list:
            curr_dist = dist_fn(q, q_rand) 
            if curr_dist < dist:
                dist = curr_dist
                q_near = list(q)
        
        q_near, q_new = extend_rrt(q_near, q_rand)    

        if q_new is not None:
            graph_list.append(q_new)
            graph[tuple(q_near)].append(q_new)
            
            dist_to_goal = dist_fn(q_new, end_conf) 
            
            if dist_to_goal <= 0.1:
                
                graph_list.append(end_conf)
                graph[tuple(q_new)].append(end_conf)
                
                path_conf = find_path(graph, start_conf, end_conf)
                
                return path_conf
    pass

if __name__ == '__main__':
    jaka = JakaButtonObsGymEnv(debug_mode=True)
    obstacles = [jaka.plane,jaka.obstacle_id]
    start_conf = jaka.getGroundTruth()
    
    goal_conf = jaka.getTargetPos()
    
    max_iter = 10000
    path_conf = None
    import time
    collision_fn = get_collision_fn_1(jaka._jaka_id,[0,1,2,3,4,5], 5,obstacles=obstacles,
                                       attachments=[], self_collisions=True,
                                       disabled_collisions=set())
    path_conf= rrt(max_iter, (0.63709654,-0.15492569 ,0.4745605), (-0.1,0.4,0.1))
    #path_conf= rrt(max_iter, start_conf, goal_conf, collision_fn)
    print(path_conf)
    if path_conf is None:
        # pause here
        raw_input("no collision-free path is found within the time budget, finish?")
        
    else:
        jaka._jaka.reset_joints()
        count = 0
        for q in path_conf:            
            q_start = jaka.getGroundTruth()
            print(q_start)
            #d = dist_fn(q_start, goal_conf)
            joints_state = p.getJointStates(jaka._jaka_id, [0,1,2,3,4,5])[0]
            d = np.linalg.norm(q_start-goal_conf)
            #jaka._jaka.apply_pos(q,joints_state)
            set_position(jaka._jaka_id,[0,1,2,3,4,5],q,5)
            p.stepSimulation()
            q_end = jaka.getGroundTruth()
            if d > 0.07:
                p.addUserDebugLine(q_start,q_end,[1, 0, 0], 4.0)
            if d < 0.07 and count > 2:
                p.addUserDebugLine(q_start,q_end,[1, 0, 0], 4.0)
            count += 1

        # execute the path 
    while True:
        jaka._jaka.reset_joints()
        for q in path_conf:
            q_start = jaka.getGroundTruth()
            joints_state = p.getJointStates(jaka._jaka_id, [0,1,2,3,4,5])[0]
            #jaka._jaka.apply_pos(q,joints_state)
            set_position(jaka._jaka_id,[0,1,2,3,4,5],q,5)
            p.stepSimulation()
            a=0
            for obstacle in obstacles:
                a += len(p.getContactPoints(jaka._jaka_id, obstacle))
            if a >0:
               print(a)
            p.stepSimulation()
            time.sleep(0.5)        



"""
    for i in range(10000):
        jaka.step(0)
        jaka.step(3)
        jaka.step(3)
        #printYellow(jaka._reward())
        printYellow(jaka.get_observation())
        time.sleep(0.1)"""

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
