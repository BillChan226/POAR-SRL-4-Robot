from environments.jaka_gym.obstacle_gym_env import JakaButtonObsGymEnv
import numpy as np
import gym
from stable_baselines import A2C
"""
jaka = JakaButtonObsGymEnv(debug_mode=True)
i = 0
import time
for i in range(10000):
    q_start = jaka.getGroundTruth()
    jaka.step(0)
    q_end = jaka.getGroundTruth()
    print(np.linalg.norm(q_end - q_start))
    q_start = jaka.getGroundTruth()
    jaka.step(1)
    q_end = jaka.getGroundTruth()
    print(np.linalg.norm(q_end - q_start))
    q_start = jaka.getGroundTruth()
    jaka.step(2)
    q_end = jaka.getGroundTruth()
    print(np.linalg.norm(q_end - q_start))
    #jaka.step(3)
    #jaka.step(3)
        #printYellow(jaka._reward())
        #printYellow(jaka.get_observation())
    time.sleep(0.1)
"""
from stable_baselines.common.vec_env import VecEnv, VecNormalize, DummyVecEnv, SubprocVecEnv, VecFrameStack
from environments.utils import makeEnv, dynamicEnvLoad

envs = makeEnv('JakaButtonObsGymEnv-v0',0,0,'logs/',0)
#envs = SubprocVecEnv(envs,'spawn')

model = A2C('MlpPolicy', envs)
model = A2C.load("logs/JakaButtonObsGymEnv-v0/ground_truth/a2c/21-06-12_14h54_10")
obs = env.reset()
for i in range(200):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    time.sleep(0.1)

