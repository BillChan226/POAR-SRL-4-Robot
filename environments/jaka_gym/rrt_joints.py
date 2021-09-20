from __future__ import division
from collections import defaultdict, OrderedDict
import pybullet as p
import pybullet_data
import numpy as np
import time
import math
import time
import argparse
import random
from environments.jaka_gym.rrt_obstacle import JakaButtonObsGymEnv
from environments.collision_joints import get_collision_fn

UR5_JOINT_INDICES = [0, 1, 2]

def set_joint_positions(body, joints, values):
    assert len(joints) == len(values)
    for joint, value in zip(joints, values):
        p.resetJointState(body, joint, value)


def draw_sphere_marker(position, radius, color):
   vs_id = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
   marker_id = p.createMultiBody(basePosition=position, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vs_id)
   return marker_id


def remove_marker(marker_id):
   p.removeBody(marker_id)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--birrt', action='store_true', default=False)
    parser.add_argument('--smoothing', action='store_true', default=False)
    args = parser.parse_args()
    return args

def find_path(graph, start, end, path =[]):
    
    path = path + [start]
    
    if tuple(start) == tuple(end):
        return path 

    for node in graph[tuple(start)]:    
        node = tuple(node)
        if node not in path:
            
            nextpath = find_path(graph, node, end, path) 

            if nextpath:  
                return nextpath


def create_step(p1,p2):
    
    delta = 0.15
    
    if np.linalg.norm(p2-p1) < delta:
        return p2

    else:
        direction_vector = (p2 - p1) / np.linalg.norm(p2-p1)
        return p1 + delta * direction_vector


def create_smooth_step(p1,p2):
    
    delta = 0.03
    
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
        set_joint_positions(jaka._jaka_id, [0,1,2,3,4,5], q_near)
        q_near_state = p.getLinkState(jaka._jaka_id, 5)[0]
        
        set_joint_positions(jaka._jaka_id, [0,1,2,3,4,5], q_new)
        q_next_state = p.getLinkState(jaka._jaka_id, 5)[0]
        p.addUserDebugLine(q_near_state,q_next_state,[0, 1, 0], 1.0)
        
        return q_near, q_new
    
    return None, None

def extend_rrt_b(q_near, q_rand, count):
    if count % 2 == 0:
        color = [0,0,1]
    else:
        color = [0,1,0]
        
    q_new = create_step(np.array(q_near), np.array(q_rand))
    q_new = q_new.tolist()
    
    if collision_fn(q_new):   
        pass
    else:
        set_joint_positions(ur5, UR5_JOINT_INDICES, q_near)
        q_near_state = p.getLinkState(ur5, 3)[0]
        
        set_joint_positions(ur5, UR5_JOINT_INDICES, q_new)
        q_next_state = p.getLinkState(ur5, 3)[0]
        p.addUserDebugLine(q_near_state,q_next_state,color, 1.0)
        
        return q_near, q_new
    
    return None, None

        
def dist_fn(p1, p2):
    return math.sqrt( ((p2[0] - p1[0]) ** 2) + ((p2[1] - p1[1]) ** 2) + ((p2[2] - p1[2]) ** 2) )

def rrt(max_iter, start_conf, end_conf):
    
    graph = defaultdict(list)
    graph_list = []
    graph_list.append(start_conf)
    
    bias = 0.01 * max_iter
    counter = 0
    
    for i in range(max_iter):
        if counter == bias:
            q_rand = end_conf
            counter = 0
        else:
            rand_joint_3 = np.random.uniform(-np.pi, np.pi, 1)
            rand_joint_2 = np.random.uniform(-1.58, 3.58, 5)
            #rand_joint_1 = np.random.uniform(2*-np.pi, 2*np.pi, 1)    
            counter += 1    
            
            rand_conf = [rand_joint_3, rand_joint_2]
            q_rand = [rand_conf[0][0], rand_conf[1][0], rand_conf[1][1],rand_conf[1][2],rand_conf[1][3],rand_conf[1][4]]
        
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
            
            if dist_to_goal <= 0.15:
                
                graph_list.append(end_conf)
                graph[tuple(q_new)].append(end_conf)
                
                path_conf = find_path(graph, start_conf, end_conf)
                
                return path_conf
    pass


def birrt(max_iter, start_conf, end_conf):
    
    graph_1 = defaultdict(list)
    graph_list_1 = []
    graph_list_1.append(start_conf)
    
    graph_2 = defaultdict(list)
    graph_list_2 = []
    graph_list_2.append(end_conf)
    
    count = 1
    color_count = 1
    for i in range(max_iter):
        
        if count % 2 == 0:
            T_1 = graph_2
            T_1_list = graph_list_2
            T_2 = graph_1
            T_2_list = graph_list_1
            color = [0,0,1]
        else:
            T_1 = graph_1
            T_1_list = graph_list_1
            T_2 = graph_2
            T_2_list = graph_list_2
            color = [0,1,0]
        
        rand_joint_3 = np.random.uniform(-np.pi, np.pi, 1)
        rand_joint_2 = np.random.uniform(2*-np.pi, 2*np.pi, 1)
        rand_joint_1 = np.random.uniform(2*-np.pi, 2*np.pi, 1)
        
        rand_conf = [rand_joint_1, rand_joint_2, rand_joint_3]
        q_rand = [rand_conf[0][0], rand_conf[1][0], rand_conf[2][0]]
    
        dist = float('inf')
        for q in T_1_list:
            curr_dist = dist_fn(q, q_rand) 
            if curr_dist < dist:
                dist = curr_dist
                q_near_1 = list(q)
        
        q_near_1, q_new_1 = extend_rrt_b(q_near_1, q_rand, count)
        
        count += 1 

        if q_new_1 is not None:
            T_1_list.append(q_new_1)
            T_1[tuple(q_near_1)].append(q_new_1)

            dist = float('inf')
            for q in T_2_list:
                curr_dist = dist_fn(q, q_new_1) 
                if curr_dist < dist:
                    dist = curr_dist
                    q_near_2 = list(q)
            
            q_near_2, q_new_2 = extend_rrt_b(q_near_2, q_new_1, count) 

            if q_new_2 is not None:
                T_2_list.append(q_new_2)
                T_2[tuple(q_near_2)].append(q_new_2)
                
                dist_to_connect = dist_fn(q_new_1, q_new_2)    
                
                if dist_to_connect < 0.1:
                    
                    graph_1[tuple(q_new_1)].append(q_new_2)                    
                    graph_2[tuple(q_new_2)].append(q_new_1)

                    path_conf_1 = find_path(graph_1, start_conf, q_new_2)
                    path_conf_2 = find_path(graph_2, end_conf, q_new_1)

                    for q in reversed(path_conf_2):
                        path_conf_1.append(q)
                    path_conf = list(OrderedDict.fromkeys(path_conf_1))  
                
                    return path_conf
                
    pass


def birrt_smoothing(smooth_iter, max_iter, start_conf, end_conf):
    path_conf = birrt(max_iter, start_conf, end_conf)

    for i in range(smooth_iter):
        list_len = len(path_conf) - 2
        rand_indx_1 = random.randint(1, list_len)
        rand_indx_2 = random.randint(1, list_len)

        rand_point_1 = path_conf[rand_indx_1]
        rand_point_2 = path_conf[rand_indx_2]
        
        if rand_indx_2 > rand_indx_1:
            d = dist_fn(rand_point_1, rand_point_2)
            conflict = True
            new_path = []
            curr_point = rand_point_1
            
            while d > 0.02:
                q_new = create_smooth_step(np.array(curr_point), np.array(rand_point_2))
                q_new = q_new.tolist()
                
                if collision_fn(q_new):
                    conflict = True
                    break
                else:
                    conflict = False
                    new_path.append(tuple(q_new))
                    
                curr_point = q_new
                d = dist_fn(q_new, rand_point_2)
        
            if not conflict:
                path_conf[rand_indx_1:rand_indx_2] = []
                path_conf[rand_indx_1:rand_indx_1] = new_path
        
        elif rand_indx_1 > rand_indx_2:
            
            d = dist_fn(rand_point_2, rand_point_1)
            conflict = True
            new_path = []
            curr_point = rand_point_2
            
            while d > 0.02:
                q_new = create_smooth_step(np.array(curr_point), np.array(rand_point_1))        
                q_new = q_new.tolist()

                if collision_fn(q_new):
                    conflict = True
                    break
                else:
                    conflict = False
                    new_path.append(tuple(q_new))
                    
                curr_point = q_new
                d = dist_fn(q_new, rand_point_1)
    
            if not conflict:
                path_conf[rand_indx_2:rand_indx_1] = []
                path_conf[rand_indx_2:rand_indx_2] = new_path       
                
    path_conf = list(OrderedDict.fromkeys(path_conf))
    return path_conf      

    
if __name__ == "__main__":
    args = get_args()
    jaka = JakaButtonObsGymEnv(debug_mode=True)
    obstacles = [jaka.plane,jaka.obstacle_id]

    # start and goal
    start_conf = (0, 0, 1, 1, 1, 1)
    start_position = (0.63709654,-0.15492569 ,0.4745605)
    goal_conf = (-1.686, 2.575, 1.462, 1.951, -1.553, 1)
    goal_position = (-0.1,0.4,0.1)
    #goal_marker = draw_sphere_marker(position=goal_position, radius=0.02, color=[1, 0, 0, 1])
    

    
    # additional variables
    max_iter = 10000
    smooth_iter = 100
    
    # place holder to save the solution path
    path_conf = None
    time_start = time.time()
    # get the collision checking function
    #from collision_joints import get_collision_fn
    collision_fn = get_collision_fn(jaka._jaka_id, [0,1,2,3,4,5], obstacles=obstacles,
                                       attachments=[], self_collisions=True,
                                       disabled_collisions=set())

    if args.birrt:
        if args.smoothing:
            # using birrt with smoothing
            path_conf = birrt_smoothing(smooth_iter, max_iter, start_conf, goal_conf)
        else:
            # using birrt without smoothing
            path_conf = birrt(max_iter, start_conf, goal_conf)
    else:
        # using rrt
        path_conf= rrt(max_iter, start_conf, goal_conf)
        
    if path_conf is None:
        # pause here
        raw_input("no collision-free path is found within the time budget, finish?")
        
    else:        

        if args.birrt:
            jaka._jaka.reset_joints()
            count = 0
            for q in path_conf:            
                q_start = p.getLinkState(jaka._jaka_id, 3)[0]
                set_joint_positions(jaka._jaka_id, [0,1,2,3,4,5], q)
                p.stepSimulation()
                q_end = p.getLinkState(jaka._jaka_id, 5)[0]  
                if count >= 1:
                    p.addUserDebugLine(q_start,q_end,[1, 0, 0], 4.0)
                count += 1
        else:
            count = 0
            d=0
            jaka._jaka.reset_joints()
           # start_time=time.time()
            for q in path_conf:            
                q_start = p.getLinkState(jaka._jaka_id, 5)[0]
                #d = dist_fn(q_start, goal_position)
                set_joint_positions(jaka._jaka_id, [0,1,2,3,4,5], q)
                q_end = p.getLinkState(jaka._jaka_id, 5)[0]  
                d += dist_fn(q_start, q_end)
                if d > 0.07:
                    p.addUserDebugLine(q_start,q_end,[1, 0, 0], 4.0)
                if d < 0.07 and count > 2:
                    p.addUserDebugLine(q_start,q_end,[1, 0, 0], 4.0)
                count += 1
            current_time=time.time()
            print(count,d)
            print(current_time-time_start)

        # execute the path 
        while True:
            jaka._jaka.reset_joints()
            for q in path_conf:
                set_joint_positions(jaka._jaka_id, [0,1,2,3,4,5], q)
                p.stepSimulation()
                a=0
                for obstacle in obstacles:
                    a += len(p.getContactPoints(jaka._jaka_id, obstacle))
                if a >0:
                    print('no')
                time.sleep(0.2)
