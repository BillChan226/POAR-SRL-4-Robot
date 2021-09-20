import os
import numpy as np
from matplotlib import pyplot as plt
import pybullet as p
import pybullet_data
import cv2

from util.color_print import printGreen, printBlue, printRed, printYellow

#URDF_PATH = "/urdf/jaka_urdf/jaka_local.urdf"
#URDF_PATH = "/urdf/JAKA/JakaRviz/Zu3/urdf/Zu3-local.urdf"
URDF_PATH = "/urdf/JAKA/JakaRviz/Zu7/urdf/jaka.urdf.xacro"
GRAVITY = -9.8
RENDER_WIDTH, RENDER_HEIGHT = 224, 224


class Jaka:
    def __init__(self, urdf_path, base_position=(0.0, 0.0, 0.2),
                 debug_mode=False, use_null_space=True, positional_control=True):
        self.urdf_path = urdf_path
        self.debug_mode = debug_mode
        self.positional_control = positional_control

        self.jaka_id = -1
        self.num_joints = -1
        self.robot_base_pos = base_position
        self.joint_lower_limits, self.joint_upper_limits, self.jointMaxForce, self.jointMaxVelocity = \
            [], [], [], []
        self.joint_name = {}
        self.debug_joints = []
        self.joints_key = []
        self.effector_id = 5
        self.initial_joints_state = [0, 0, 1, 1, 1, 1]
        self.effector_pos = None

        self.use_null_space = use_null_space
        if self.debug_mode:
            client_id = p.connect(p.SHARED_MEMORY)
            if client_id < 0:
                p.connect(p.GUI)
            # The camera information for debug (GUI graphical interface)
            p.resetDebugVisualizerCamera(2., 180, -41, [0.52, -0.2, 1])
            debug_joints = []
            self.debug_joints = debug_joints
        else:
            client_id = p.connect(p.SHARED_MEMORY)
            if client_id < 0:
                p.connect(p.DIRECT)
        self.reset()

    def reset(self):
        self.jaka_id = p.loadURDF(self.urdf_path)
        self.num_joints = p.getNumJoints(self.jaka_id)
        p.resetBasePositionAndOrientation(self.jaka_id, self.robot_base_pos,
                                          [0.000000, 0.000000, 0.000000, 1.000000])
        self.get_joint_info()
        for jointIndex in self.joints_key:
            p.resetJointState(self.jaka_id, jointIndex, self.initial_joints_state[jointIndex])
        p.stepSimulation()

    def apply_action(self, action):
        if self.positional_control:
            self.apply_action_pos(action)
        else:
            self.apply_action_joints(action)

    def render(self, mode='channel_last'):
        return self._render(mode=mode)

    ##############################################################
    # 我是分割线，下面的函数基本上不太用动了 ############################
    ##############################################################
    def reset_joints(self):
        for jointIndex in self.joints_key:
            p.resetJointState(self.jaka_id, jointIndex, self.initial_joints_state[jointIndex])

    def getGroundTruth(self):
        if self.positional_control:
            position = p.getLinkState(self.jaka_id, self.effector_id)[0]
            return np.array(position)
        else:  # control by joint and return the joint state (joint position)
            # we can add joint velocity as joint state, but here we didnt, getJointState can get us more information
            joints_state = p.getJointStates(self.jaka_id, self.joints_key)
            return np.array(joints_state)[:, 0]

    def get_joint_info(self):
        num_joints = self.num_joints
        for i in range(num_joints):
            infos = p.getJointInfo(self.jaka_id, i)	
            """
            if infos[2] == p.JOINT_REVOLUTE:
                self.joint_name[i] = infos[1]
                self.joint_lower_limits.append(infos[8])
                self.joint_upper_limits.append(infos[9])
                self.jointMaxForce.append(infos[10])
                self.jointMaxVelocity.append(infos[11])
                self.joints_key.append(i)
            """
            self.joint_name[i] = infos[1]
            self.joint_lower_limits.append(-3.14)
            self.joint_upper_limits.append(3.14)
            self.jointMaxForce.append(1000)
            self.jointMaxVelocity.append(0.5)
            self.joints_key.append(i)
           
        if self.debug_mode:
            keys = list(self.joint_name.keys())
            keys.sort()
            for i, key in enumerate(keys):
                self.debug_joints.append(p.addUserDebugParameter(self.joint_name[key].decode(),
                                                                 self.joint_lower_limits[i],
                                                                 self.joint_upper_limits[i], 0.))

    def apply_action_joints(self, motor_commands):
        """
        Apply the action to the inmoov robot 53 joints that can be moved, so please provide an array with 53 values
        :param motor_commands: the motor command for the joints of robot, a direct control way
        """
        assert len(motor_commands) == len(self.joints_key), "Error, please provide control commands for all joints"
        num_control = len(motor_commands)
        target_velocities = [0] * num_control
        position_gains = [0.3] * num_control
        velocity_gains = [1] * num_control
        joint_targets = np.clip(a=motor_commands, a_min=self.joint_lower_limits, a_max=self.joint_upper_limits)
        p.setJointMotorControlArray(bodyUniqueId=self.jaka_id,
                                    controlMode=p.POSITION_CONTROL,
                                    jointIndices=self.joints_key,
                                    targetPositions=joint_targets,
                                    targetVelocities=target_velocities,
                                    forces=self.jointMaxForce,
                                    positionGains=position_gains,
                                    velocityGains=velocity_gains
                                    )

    def apply_action_pos(self, pos_commands):
        dx, dy, dz = pos_commands
        effector_id = self.effector_id
        link_pos = p.getLinkState(self.jaka_id, effector_id)
        self.effector_pos = link_pos[0]
        target_pos = (self.effector_pos[0] + dx, self.effector_pos[1] + dy, self.effector_pos[2] + dz)
        rest_poses = self.initial_joints_state
        joint_ranges = [4] * len(self.joints_key)
        target_velocity = [0] * len(self.joints_key)
        if self.use_null_space:
            joint_poses = p.calculateInverseKinematics(self.jaka_id, effector_id, target_pos,
                                                       lowerLimits=self.joint_lower_limits,
                                                       upperLimits=self.joint_upper_limits,
                                                       jointRanges=joint_ranges,
                                                       restPoses=rest_poses
                                                       )
        else:  # use regular KI solution
            joint_poses = p.calculateInverseKinematics(self.jaka_id, self.effector_id, target_pos)
        p.setJointMotorControlArray(self.jaka_id, self.joints_key,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=joint_poses,
                                    targetVelocities=target_velocity,
                                    forces=self.jointMaxForce)
        p.stepSimulation()

    def _render(self, mode='channel_last', img_name=None):
        """
        You could change the Yaw Pitch Roll distance to change the view of the robot
        :param mode:
        :return:
        """
        camera_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self.robot_base_pos,
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

        px = px.reshape(RENDER_WIDTH, RENDER_WIDTH, -1)
        depth = depth.reshape(RENDER_WIDTH, RENDER_WIDTH, -1)
        mask = mask.reshape(RENDER_WIDTH, RENDER_WIDTH, -1)
        px = px[..., :-1]
        if img_name is not None:
            cv2.imwrite(img_name, px[..., [2, 1, 0]])
        if mode != 'channel_last':
            # channel first
            px = np.transpose(px, [2, 0, 1])
        return px, depth, mask

    def debugger_step(self):
        assert self.debug_mode, "Error: the debugger_step is only allowed in debug mode"
        current_joints = []
        for j in self.debug_joints:
            tmp_joint_control = p.readUserDebugParameter(j)
            current_joints.append(tmp_joint_control)
        for joint_state, joint_key in zip(current_joints, self.joints_key):
            p.resetJointState(self.jaka_id, joint_key, targetValue=joint_state)
        p.stepSimulation()


if __name__ == '__main__':
    jaka = Jaka(URDF_PATH, debug_mode=True)
    _urdf_path = pybullet_data.getDataPath()
    planeId = p.loadURDF(os.path.join(_urdf_path, "plane.urdf"))
    # p.loadURDF(os.path.join('urdf/', "cylinder.urdf"), [1, 0, 0])
    # p.loadURDF(os.path.join('urdf/', "wall.urdf"), [1, 0, 0])
    obs_id = p.loadURDF(os.path.join('urdf/', "long_cylinder.urdf"), [0.3, 0.3, 0])
    id1 = p.loadURDF("/urdf/simple_button.urdf", [0, 0.4, 0])
    p.setGravity(0,0,-10)
    i = 0
    print(jaka.joint_name)
    import ipdb

    while True:
        i += 1
        jaka.debugger_step()
        #if len(p.getContactPoints(jaka.jaka_id, obs_id)) > 0:
            #ipdb.set_trace()
        # print(jaka.getGroundTruth())
        # jaka.apply_action_pos([-0.1, 0.1, 0.2])
        # jaka._render(img_name='trash/out{}.png'.format(i))
