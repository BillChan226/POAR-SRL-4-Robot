import os
import numpy as np
from matplotlib import pyplot as plt
import pybullet as p

from environments.inmoov.joints_registry import joint_registry, control_joint
# debugger
# colorful print
from util.color_print import printGreen, printBlue, printRed, printYellow
from ipdb import set_trace as tt

URDF_PATH = "/urdf_robot/"
GRAVITY = -9.8
# 修改这里来改变图像的尺寸
RENDER_WIDTH, RENDER_HEIGHT = 224, 224


class Inmoov:
    def __init__(self, urdf_path=URDF_PATH, positional_control=True, debug_mode=False, use_null_space=True):
        """
        To initialize an inmoov robot with controlable joint
        :param urdf_path: (str) the path to the file, where the inmoov robot urdf file is
        :param positional_control: (bool)
        :param debug_mode: (bool) if True, it will give us a pybullet GUI interface
        :param use_null_space: (bool) the parameter for the inverseKinetic solver, since the solution
        to a inverse Kinetic problem can be non-unique, it will give us a solution that be
        constrained in the feasible domain defined in the urdf file.

        """
        self.urdf_path = urdf_path
        self._renders = True
        self.debug_mode = debug_mode
        self.inmoov_id = -1
        self.num_joints = -1  # number of the joints
        self.robot_base_pos = [0, 0, 0]  # the base position of inmoov robot
        # effectorID = 28: right hand
        # effectorID = 59: left hand
        self.effectorId = 28
        self.effector_pos = None
        # joint information
        # jointName, (jointLowerLimit, jointUpperLimit), jointMaxForce, jointMaxVelocity, linkName, parentIndex
        self.joint_name = {}
        self.joint_lower_limits = None
        self.joint_upper_limits = None
        self.jointMaxForce = None
        self.jointMaxVelocity = None
        self.joints_key = None
        # camera position
        self.camera_target_pos = (0.0, 0.0, 1.0)
        # Control mode: by joint or by effector position
        self.positional_control = positional_control
        # inverse Kinematic solver, ref: Pybullet
        self.use_null_space = use_null_space
        if self.debug_mode:
            client_id = p.connect(p.SHARED_MEMORY)
            if client_id < 0:
                p.connect(p.GUI)
            # The camera information for debug (GUI graphical interface)
            p.resetDebugVisualizerCamera(5., 180, -41, [0.52, -0.2, -0.33])
            # To debug the joints of the Inmoov robot
            debug_joints = []
            self.joints_key = []
            # I dig out the joint information (Name and order) manually and store it in to the joint_registry file
            # In fact this is a little bit stupid, the correct way to get joint information is to be found in the
            # function self.get_joint_info()
            for joint_index in control_joint:
                self.joints_key.append(joint_index)
                # This will add some slider on the GUI
                debug_joints.append(p.addUserDebugParameter(control_joint[joint_index], -3.14, 3.14, 0))
            self.debug_joints = debug_joints
        else:
            client_id = p.connect(p.SHARED_MEMORY)
            if client_id < 0:
                p.connect(p.DIRECT)
        self.reset()

    def reset_joints(self):
        """
        Reset robot joints to initial position for faster resetting
        """
        for jointIndex in self.joints_key:
            p.resetJointState(self.inmoov_id, jointIndex, 0.)
        self.effector_pos = p.getLinkState(self.inmoov_id, self.effectorId)[0]

    def reset(self):
        """
        Reset the environment
        """
        self.inmoov_id = p.loadURDF(os.path.join(self.urdf_path, 'inmoov_colmass.urdf'), self.robot_base_pos)
        self.num_joints = p.getNumJoints(self.inmoov_id)
        self.get_joint_info()
        for jointIndex in self.joints_key:
            p.resetJointState(self.inmoov_id, jointIndex, 0.)
        # get the effector world position
        self.effector_pos = p.getLinkState(self.inmoov_id, self.effectorId)[0]
        # plt.ion()
        # # get link information
        # ######################## debug part #######################
        # # this debug part will plot a 3D representation of the inmoov annotated links
        # from mpl_toolkits.mplot3d import Axes3D
        # #To plot the link index by graphical representation
        # link_position = []
        # p.getBasePositionAndOrientation(self.inmoov_id)
        # for i in range(100):
        #     print("linkWorldPosition, , , , workldLinkFramePosition", i)
        #     link_state = p.getLinkState(self.inmoov_id, i)
        #     if link_state is not None:
        #         link_position.append(link_state[0])
        #
        # link_position = np.array(link_position).T
        # print(link_position.shape)
        #
        # fig = plt.figure("3D link plot")
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(link_position[0], link_position[1], link_position[2], c='r', marker='o')
        # for i in range(link_position.shape[1]):
        #     # ax.annotate(str(i), (link_position[0,i], link_position[1,i], link_position[2,i]) )
        #     ax.text(link_position[0,i], link_position[1,i], link_position[2,i], str(i))
        # # ax.set_xlim([-1, 1])
        # # ax.set_ylim([-1, 1])
        # ax.set_xlim([-.25, .25])
        # ax.set_ylim([-.25, .25])
        # ax.set_zlim([1, 2])
        # plt.show()
        # ####################### debug part #######################

    def getGroundTruth(self):
        if self.positional_control:
            position = p.getLinkState(self.inmoov_id, self.effectorId)[0]
            return np.array(position)
        else:  # control by joint and return the joint state (joint position)
            # we can add joint velocity as joint state, but here we didnt, getJointState can get us more infomation
            joints_state = p.getJointStates(self.inmoov_id, self.joints_key)
            return np.array(joints_state)[:, 0]

    def getGroundTruthDim(self):
        if self.positional_control:
            return 3
        else:
            return len(self.joints_key)

    def step(self, action):
        raise NotImplementedError

    def _termination(self):
        raise NotImplementedError

    def _reward(self):
        raise NotImplementedError

    @staticmethod
    def get_effector_dimension(self):
        """
        The DOF for the effector
        :return: three dimension for x, y and z
        """
        return 3

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
        p.setJointMotorControlArray(bodyUniqueId=self.inmoov_id,
                                    controlMode=p.POSITION_CONTROL,
                                    jointIndices=self.joints_key,
                                    targetPositions=joint_targets,
                                    targetVelocities=target_velocities,
                                    forces=self.jointMaxForce,
                                    positionGains=position_gains,
                                    velocityGains=velocity_gains
                                    )
        # # Same functionality, but upper lines work faster
        # for i in range(num_control):
        #     # p.setJointMotorControl2(bodyUniqueId=self.inmoov_id, jointIndex=CONTROL_JOINT[i],
        #     #                         controlMode=p.POSITION_CONTROL, targetPosition=joint_poses[i],
        #     #                         targetVelocity=0, force=self.jointMaxForce[i],
        #     #                         maxVelocity=self.max_velocity, positionGain=0.3, velocityGain=1)
        #     p.setJointMotorControl2(bodyUniqueId=self.inmoov_id, jointIndex=CONTROL_JOINT[i],
        #                             controlMode=p.POSITION_CONTROL, targetPosition=joint_poses[i],
        #                             targetVelocity=0, force=self.jointMaxForce[i],
        #                             maxVelocity=4., positionGain=0.3, velocityGain=1)
        # p.stepSimulation()

    def apply_action_pos(self, motor_commands):
        """
        Apply the action to the inmoov robot joint.
        This is a continual version of control, it can move along x, y, z axis
        :param motor_commands: [dx, dy, dz]
        """
        # TODO: Add orientation control information for a better physical representation
        # assert len(motor_commands) == (3+4) x,y,z + Quaternion
        assert len(motor_commands) == 3, "Invalid input commands, please use a 3D: x,y,z information"
        dx, dy, dz = motor_commands
        # I found that the control has error, that means that the the robot may not effect the exact [dx, dy, dz]
        # movement. and the best way to get the current position is to getLinkState
        # instead of running a self.variable += [dx, dy, dz]
        joint_position = p.getLinkState(self.inmoov_id, self.effectorId)
        current_state = joint_position[0]
        self.effector_pos = current_state
        # TODO: it might be better to constraint the effector position in a constraint box? (the effector can not move
        #  anywhere we want anyway..)
        target_pos = (current_state[0] + dx, current_state[1] + dy, current_state[2] + dz)
        # Compute the inverse kinematics for every revolute joint
        num_control_joints = len(self.joints_key)
        # position at rest
        rest_poses = [0] * num_control_joints
        # don't know what is its influence
        joint_ranges = [4] * num_control_joints
        # the velocity at the target position
        target_velocity = [0] * num_control_joints
        if self.use_null_space:
            # TODO: Add orientation control (The current version support only the positional control)
            # Solve the Inverse Kinematics for joints to arrive at the target position for effector link
            joint_poses = p.calculateInverseKinematics(self.inmoov_id, self.effectorId, target_pos,
                                                       lowerLimits=self.joint_lower_limits,
                                                       upperLimits=self.joint_upper_limits,
                                                       jointRanges=joint_ranges,
                                                       restPoses=rest_poses
                                                       )
        else:  # use regular KI solution
            joint_poses = p.calculateInverseKinematics(self.inmoov_id, self.effectorId, target_pos)
        p.setJointMotorControlArray(self.inmoov_id, self.joints_key,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=joint_poses,
                                    targetVelocities=target_velocity,
                                    #  maxVelocities=self.jointMaxVelocity,
                                    forces=self.jointMaxForce)
        # Run the simulation step
        p.stepSimulation()
        # # I left these lines below, since the setJointMotorControlArray do not support "maxVelocity", yet I don't know
        # # the effect.
        # for i, index in enumerate(self.joints_key):
        #     joint_info = self.joint_name[index]
        #     jointMaxForce, jointMaxVelocity = joint_info[2:4]
        #
        #     p.setJointMotorControl2(bodyUniqueId=self.inmoov_id, jointIndex=index, controlMode=p.POSITION_CONTROL,
        #                             targetPosition=joint_poses[i], targetVelocity=0, force=self.jointMaxForce[i],
        #                             maxVelocity=self.max_velocity, positionGain=0.3, velocityGain=1)

    def apply_action(self, motor_commands):
        if self.positional_control:
            self.apply_action_pos(motor_commands)
        else:
            self.apply_action_joints(motor_commands)

    def get_joint_info(self):
        """
        This method is to register the joint information into the class, more information for method "p.getJointInfo"
        to be found in the pybullet quickStart
        From this we can see the fact that:
        - no joint damping is set
        - some of the joints are reserved???
        - none of them has joint Friction
        - we have 53 revolte joints (that can be moved)
        """
        # self.joints_key = []
        self.joint_lower_limits, self.joint_upper_limits, self.jointMaxForce, self.jointMaxVelocity = [], [], [], []
        for i in range(self.num_joints):
            info = p.getJointInfo(self.inmoov_id, i)
            # if info[7] != 0:
            #     print(info[1], "has friction")
            # if info[6] != 0:
            #     print(info[1], "has damping")
            if info[2] == p.JOINT_REVOLUTE:
                # jointName, (jointLowerLimit, jointUpperLimit), jointMaxForce, jointMaxVelocity, linkName, parentIndex
                # (info[1], (info[8], info[9]), info[10], info[11], info[12], info[16])
                self.joint_name[i] = info[1]
                # if info[1] == b'right_bicep':
                #     printYellow(info)
                # if info[1] == b'left_bicep':
                #     printGreen(info)
                self.joint_lower_limits.append(info[8])
                self.joint_upper_limits.append(info[9])
                self.jointMaxForce.append(info[10])
                self.jointMaxVelocity.append(info[11])
                # self.joints_key.append(i)

    def debugger_step(self):
        """
        控制 debug slider来控制机器人
        if you run self.debugger_step, the sliders on the GUI will be activated and keep catching info from sliders,
        and control the joints as you indicated
        :return:
        """
        assert self.debug_mode, "Error: the debugger_step is only allowed in debug mode"

        current_joints = []
        # The order is as the same as the self.joint_key
        for j in self.debug_joints:
            tmp_joint_control = p.readUserDebugParameter(j)
            current_joints.append(tmp_joint_control)
        for joint_state, joint_key in zip(current_joints, self.joints_key):
            p.resetJointState(self.inmoov_id, joint_key, targetValue=joint_state)
        p.stepSimulation()

        # self.robot_render()
        # These lines will let the camera information project on the GUI, so set self._renders True to see it
        # if self._renders:
        #     view_matrix1 = p.computeViewMatrixFromYawPitchRoll(
        #         cameraTargetPosition=self.camera_target_pos,
        #         distance=2.,
        #         yaw=145,  # 145 degree
        #         pitch=-36,  # -36 degree
        #         roll=0,
        #         upAxisIndex=2
        #     )
        #     proj_matrix1 = p.computeProjectionMatrixFOV(
        #         fov=60, aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
        #         nearVal=0.1, farVal=100.0)
        #     # depth: the depth camera, mask: mask on different body ID
        #     (_, _, px, depth, mask) = p.getCameraImage(
        #         width=RENDER_WIDTH, height=RENDER_HEIGHT, viewMatrix=view_matrix1,
        #         projectionMatrix=proj_matrix1, renderer=p.ER_TINY_RENDERER)

    def debugger_camera(self):
        if self.debug_mode:
            TeTe = "Stupid"
            print(TeTe)

    def robot_render(self):
        """
        The image from the robot eye
        link 17 and 21
        :return:
        """
        right_eye_state = p.getLinkState(self.inmoov_id, 17)
        # tt()
        left_eye_state = p.getLinkState(self.inmoov_id, 21)
        right_eye_pos = np.array(right_eye_state[0])
        right_eye_orn = right_eye_state[1]
        left_eye_pos, left_eye_orn = np.array(left_eye_state[0]), left_eye_state[1]

        focus = 0.5
        # TODO: bad angle
        r_rotation_matrix = np.array(p.getMatrixFromQuaternion(right_eye_orn)).reshape([3,3]).T
        l_rotation_matrix = np.array(p.getMatrixFromQuaternion(left_eye_orn)).reshape([3,3]).T
        front = np.array([0, focus, 0])
        right_target_pos = right_eye_pos + front@r_rotation_matrix
        left_target_pos = left_eye_pos + front @ l_rotation_matrix
        right_eye_matrix = p.computeViewMatrix(
            cameraEyePosition = right_eye_pos,
            cameraTargetPosition = right_target_pos,
            cameraUpVector = [0, 0, 1]
        )
        left_eye_matrix = p.computeViewMatrix(
            cameraEyePosition=left_eye_pos,
            cameraTargetPosition=left_target_pos,
            cameraUpVector=[0, 0, 1]
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=100, aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
            nearVal=0.1, farVal=100.0)

        # depth: the depth camera, mask: mask on different body ID
        (_, _, px, depth, mask) = p.getCameraImage(
            width=RENDER_WIDTH, height=RENDER_HEIGHT, viewMatrix=right_eye_matrix,
            projectionMatrix=proj_matrix, renderer=p.ER_TINY_RENDERER)
        right_px, right_depth, right_mask = np.array(px), np.array(depth), np.array(mask)
        (_, _, px, depth, mask) = p.getCameraImage(
            width=RENDER_WIDTH, height=RENDER_HEIGHT, viewMatrix=left_eye_matrix,
            projectionMatrix=proj_matrix, renderer=p.ER_TINY_RENDERER)
        left_px, left_depth, left_mask = np.array(px), np.array(depth), np.array(mask)
        if self._renders:
            plt.ion()
            figsize = np.array([3, 5]) * 3
            fig = plt.figure("Inmoov", figsize=figsize)
            ax1 = fig.add_subplot(321)
            ax1.imshow(left_px)
            ax1.set_title("Left")
            ax2 = fig.add_subplot(322)
            ax2.imshow(right_px)
            ax2.set_title("Right")
            ax3 = fig.add_subplot(323)
            ax3.imshow(left_depth)
            ax4 = fig.add_subplot(324)
            ax4.imshow(right_depth)
            ax5 = fig.add_subplot(325)
            ax5.imshow(left_mask)
            ax6 = fig.add_subplot(326)
            ax6.imshow(right_mask)
            # rgb_array = np.array(px)
            # self.image_plot = plt.imshow(rgb_array)
            # self.image_plot.axes.grid(False)
            # plt.title("Inmoov Robot Simulation")
            fig.suptitle('Inmoov Simulation: Two Cameras View', fontsize=32)
            plt.draw()
            # To avoid too fast drawing conflict
            plt.pause(0.00001)

        # TODO

    def render(self, num_camera=1):
        """
        This function will not actually be used, but can inspire you to create camera in the simulation
        :param num_camera:
        :return:
        """
        view_matrix1 = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self.camera_target_pos,
            distance=2.,
            yaw=145,  # 145 degree
            pitch=-36,  # -36 degree
            roll=0,
            upAxisIndex=2
        )
        proj_matrix1 = p.computeProjectionMatrixFOV(
            fov=60, aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
            nearVal=0.1, farVal=100.0)
        # depth: the depth camera, mask: mask on different body ID
        (_, _, px, depth, mask) = p.getCameraImage(
            width=RENDER_WIDTH, height=RENDER_HEIGHT, viewMatrix=view_matrix1,
            projectionMatrix=proj_matrix1, renderer=p.ER_TINY_RENDERER)
        px, depth, mask = np.array(px), np.array(depth), np.array(mask)

        if self._renders:
            plt.ion()
            if num_camera == 1:
                figsize = np.array([3, 1]) * 5
            else:
                figsize = np.array([3, 2]) * 5
            fig = plt.figure("Inmoov", figsize=figsize)

            if num_camera == 2:
                view_matrix2 = p.computeViewMatrixFromYawPitchRoll(
                    cameraTargetPosition=(0.316, 0.316, 1.0),
                    distance=1.2,
                    yaw=90,  # 145 degree
                    pitch=-13,  # -36 degree
                    roll=0,
                    upAxisIndex=2
                )
                proj_matrix2 = p.computeProjectionMatrixFOV(
                    fov=60, aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
                    nearVal=0.1, farVal=100.0)
                (_, _, px2, depth2, mask2) = p.getCameraImage(
                    width=RENDER_WIDTH, height=RENDER_HEIGHT, viewMatrix=view_matrix2,
                    projectionMatrix=proj_matrix2, renderer=p.ER_TINY_RENDERER)
                ax1 = fig.add_subplot(231)
                ax1.imshow(px)
                ax1.set_title("rgb_1")
                ax2 = fig.add_subplot(232)
                ax2.imshow(depth)
                ax2.set_title("depth_1")
                ax3 = fig.add_subplot(233)
                ax3.imshow(mask)
                ax3.set_title("mask_1")
                ax1 = fig.add_subplot(234)
                ax1.imshow(px2)
                ax1.set_title("rgb_2")
                ax2 = fig.add_subplot(235)
                ax2.imshow(depth2)
                ax2.set_title("depth_2")
                ax3 = fig.add_subplot(236)
                ax3.imshow(mask2)
                ax3.set_title("mask_2")
            elif num_camera == 1:  # only one camera
                ax1 = fig.add_subplot(131)
                ax1.imshow(px)
                ax1.set_title("rgb_1")
                ax2 = fig.add_subplot(132)
                ax2.imshow(depth)
                ax2.set_title("depth_1")
                ax3 = fig.add_subplot(133)
                ax3.imshow(mask)
                ax3.set_title("mask_1")
            # rgb_array = np.array(px)
            # self.image_plot = plt.imshow(rgb_array)
            # self.image_plot.axes.grid(False)
            # plt.title("Inmoov Robot Simulation")
            fig.suptitle('Inmoov Simulation: Two Cameras View', fontsize=32)
            plt.draw()
            # To avoid too fast drawing conflict
            plt.pause(0.00001)

