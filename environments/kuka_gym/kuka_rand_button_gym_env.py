from .kuka_button_gym_env import *

MAX_STEPS = 1000
BALL_FORCE = 10


class KukaRandButtonGymEnv(KukaButtonGymEnv):
    """
    Gym wrapper for Kuka environment with a push button in a random position
        and some random objects
    :param urdf_root: (str) Path to pybullet urdf files
    :param renders: (bool) Whether to display the GUI or not
    :param is_discrete: (bool) Whether to use discrete or continuous actions
    :param multi_view :(bool) if TRUE -> returns stacked images of the scene on 6 channels (two cameras)
    :param name: (str) name of the folder where recorded data will be stored
    :param max_distance: (float) Max distance between end effector and the button (for negative reward)
    :param action_repeat: (int) Number of timesteps an action is repeated (here it is equivalent to frameskip)
    :param shape_reward: (bool) Set to true, reward = -distance_to_goal
    :param action_joints: (bool) Set actions to apply to the joint space
    :param record_data: (bool) Set to true, record frames with the rewards.
    :param random_target: (bool) Set the button position to a random position on the table
    :param force_down: (bool) Set Down as the only vertical action allowed
    :param state_dim: (int) When learning states
    :param learn_states: (bool)
    :param verbose: (bool) Whether to print some debug info
    :param save_path: (str) location where the saved data should go
    :param env_rank: (int) the number ID of the environment
    :param srl_pipe: (Queue, [Queue]) contains the input and output of the SRL model
    :param srl_model: (str) The SRL_model used
    """
    
    def __init__(self, name="kuka_rand_button_gym", **kwargs):
        super(KukaRandButtonGymEnv, self).__init__(name=name, **kwargs)

        self.max_steps = MAX_STEPS

    def reset(self):
        self.terminated = False
        self.n_contacts = 0
        self.n_steps_outside = 0
        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._timestep)
        p.loadURDF(os.path.join(self._urdf_root, "plane.urdf"), [0, 0, -1])

        self.table_uid = p.loadURDF(os.path.join(self._urdf_root, "table/table.urdf"), 0.5000000, 0.00000, -.820000,
                                    0.000000, 0.000000, 0.0, 1.0)

        # Initialize button position
        x_pos = 0.5
        y_pos = 0
        if self._random_target:
            x_pos += 0.15 * self.np_random.uniform(-1, 1)
            y_pos += 0.3 * self.np_random.uniform(-1, 1)

        self.button_uid = p.loadURDF("/urdf/simple_button.urdf", [x_pos, y_pos, Z_TABLE])
        self.button_pos = np.array([x_pos, y_pos, Z_TABLE])

        rand_objects = ["duck_vhacd.urdf", "lego/lego.urdf", "cube_small.urdf"]
        for _ in range(10):
            obj = rand_objects[np.random.randint(len(rand_objects))]
            x_pos = 0.5 + 0.15 * self.np_random.uniform(-1, 1)
            y_pos = 0 + 0.3 * self.np_random.uniform(-1, 1)
            if (x_pos < self.button_pos[0] - 0.1) or (x_pos > self.button_pos[0] + 0.1) or (
                    y_pos < self.button_pos[1] - 0.1) or (y_pos > self.button_pos[1] + 0.1):
                p.loadURDF(os.path.join(self._urdf_root, obj), [x_pos, y_pos, Z_TABLE + 0.1])

        self.sphere = p.loadURDF(os.path.join(self._urdf_root, "sphere_small.urdf"), [0.25, -0.2, Z_TABLE + 0.3])

        p.setGravity(0, 0, -10)
        self._kuka = kuka.Kuka(urdf_root_path=self._urdf_root, timestep=self._timestep,
                               use_inverse_kinematics=(not self.action_joints),
                               small_constraints=(not self._random_target))
        self._env_step_counter = 0
        # Close the gripper and wait for the arm to be in rest position
        for _ in range(500):
            if self.action_joints:
                self._kuka.applyAction(list(np.array(self._kuka.joint_positions)[:7]) + [0, 0])
            else:
                self._kuka.applyAction([0, 0, 0, 0, 0])
            p.stepSimulation()

        # Randomize init arm pos: take 5 random actions
        for _ in range(N_RANDOM_ACTIONS_AT_INIT):
            if self._is_discrete:
                action = [0, 0, 0, 0, 0]
                sign = 1 if self.np_random.rand() > 0.5 else -1
                action_idx = self.np_random.randint(3)  # dx, dy or dz
                action[action_idx] += sign * DELTA_V
            else:
                if self.action_joints:
                    joints = np.array(self._kuka.joint_positions)[:7]
                    joints += DELTA_THETA * self.np_random.normal(joints.shape)
                    action = list(joints) + [0, 0]
                else:
                    action = np.zeros(5)
                    rand_direction = self.np_random.normal((3,))
                    # L2 normalize, so that the random direction is not too high or too low
                    rand_direction /= np.linalg.norm(rand_direction, 2)
                    action[:3] += DELTA_V_CONTINUOUS * rand_direction

            self._kuka.applyAction(list(action))
            p.stepSimulation()

        self._observation = self.getExtendedObservation()

        self.button_pos = np.array(p.getLinkState(self.button_uid, BUTTON_LINK_IDX)[0])
        self.button_pos[2] += BUTTON_DISTANCE_HEIGHT  # Set the target position on the top of the button
        if self.saver is not None:
            self.saver.reset(self._observation, self.button_pos, self.getGroundTruth())

        if self.srl_model != "raw_pixels":
            return self.getSRLState(self._observation)

        return np.array(self._observation)

    def step(self, action):
        # force applied to the ball
        if self._env_step_counter == 10:
            force = np.random.normal(size=(3,))
            force[2] = 0  # set up force to 0, so that our amplitude is correct
            force = force / np.linalg.norm(force, 2) * BALL_FORCE  # set force amplitude
            force[2] = 1  # up force to reduce friction
            force = np.abs(force)  #  go towards the center of the table
            p.applyExternalForce(self.sphere, -1, force, [0, 0, 0], p.WORLD_FRAME)

        return super(KukaRandButtonGymEnv, self).step(action)
