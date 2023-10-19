#!/bin/bash
# preparing for IEEE Intelligent systems

env="MobileRobotGymEnv-v0"

cd ..
#env="OmnirobotEnv-v0"
#
# 2023/10/10
# 图一：各算法对比
# 涉及到的算法：a2c acer ppo2 srl_split_a1r5i2f1 srl_combination_a1r5i2f1 srl_split_a10r5i1f1 srl_decoupling_split srl_decoupling_combination 缺：增加MMD loss的模型
#
#### a2c
#python -m rl_baselines.train --seed 0 --log-dir logs/POAR/a2c/ --algo a2c --num-timesteps 2000000 --env $env  --srl-model raw_pixels --num-cpu 8 --gpu 1 -r #--port 9999
#python -m rl_baselines.train --seed 1 --log-dir logs/POAR/a2c/ --algo a2c --num-timesteps 2000000 --env $env  --srl-model raw_pixels --num-cpu 8 --gpu 1 -r #--port 9999
#python -m rl_baselines.train --seed 2 --log-dir logs/POAR/a2c/ --algo a2c --num-timesteps 2000000 --env $env  --srl-model raw_pixels --num-cpu 8 --gpu 1 -r #--port 9999
#
#### acer
#python -m rl_baselines.train --seed 0 --log-dir logs/POAR/acer/ --algo acer --num-timesteps 2000000 --env $env  --srl-model raw_pixels --num-cpu 8 --gpu 1 -r #--port 9999
#python -m rl_baselines.train --seed 1 --log-dir logs/POAR/acer/ --algo acer --num-timesteps 2000000 --env $env  --srl-model raw_pixels --num-cpu 8 --gpu 1 -r #--port 9999
#python -m rl_baselines.train --seed 2 --log-dir logs/POAR/acer/ --algo acer --num-timesteps 2000000 --env $env  --srl-model raw_pixels --num-cpu 8 --gpu 1 -r #--port 9999
#
#### ppo2
#python -m rl_baselines.train --seed 0 --log-dir logs/POAR/ppo2/ --algo ppo2 --num-timesteps 2000000 --env $env  --srl-model raw_pixels --num-cpu 8 --gpu 1 -r #--port 9999
#python -m rl_baselines.train --seed 1 --log-dir logs/POAR/ppo2/ --algo ppo2 --num-timesteps 2000000 --env $env  --srl-model raw_pixels --num-cpu 8 --gpu 1 -r #--port 9999
#python -m rl_baselines.train --seed 2 --log-dir logs/POAR/ppo2/ --algo ppo2 --num-timesteps 2000000 --env $env  --srl-model raw_pixels --num-cpu 8 --gpu 1 -r #--port 9999

#### srl_split_a1r5i2f1
#python -m rl_baselines.train --seed 0 --log-dir logs/POAR/srl_split_a1r5i2f1/ --algo poar --num-timesteps 2000000 --env $env --structure srl_autoencoder --losses autoencoder:1:200 reward:5:-1 inverse:2:20 forward:1:10 --num-cpu 8 --gpu 0 -r
#python -m rl_baselines.train --seed 1 --log-dir logs/POAR/srl_split_a1r5i2f1/ --algo poar --num-timesteps 2000000 --env $env --structure srl_autoencoder --losses autoencoder:1:200 reward:5:-1 inverse:2:20 forward:1:10 --num-cpu 8 --gpu 0 -r
#python -m rl_baselines.train --seed 2 --log-dir logs/POAR/srl_split_a1r5i2f1/ --algo poar --num-timesteps 2000000 --env $env --structure srl_autoencoder --losses autoencoder:1:200 reward:5:-1 inverse:2:20 forward:1:10 --num-cpu 8 --gpu 0 -r

#### srl_combination_a1r5i2f1
#python -m rl_baselines.train --seed 0 --log-dir logs/POAR/srl_combination_a1r5i2f1/ --algo poar --num-timesteps 2000000 --env $env --structure srl_autoencoder --losses autoencoder:1:230 reward:5:-1 inverse:2:-1 forward:1:-1 --num-cpu 8 --gpu 0 -r
#python -m rl_baselines.train --seed 1 --log-dir logs/POAR/srl_combination_a1r5i2f1/ --algo poar --num-timesteps 2000000 --env $env --structure srl_autoencoder --losses autoencoder:1:230 reward:5:-1 inverse:2:-1 forward:1:-1 --num-cpu 8 --gpu 0 -r
#python -m rl_baselines.train --seed 2 --log-dir logs/POAR/srl_combination_a1r5i2f1/ --algo poar --num-timesteps 2000000 --env $env --structure srl_autoencoder --losses autoencoder:1:230 reward:5:-1 inverse:2:-1 forward:1:-1 --num-cpu 8 --gpu 0 -r
#
#### srl_split_a10r5i1f1
#python -m rl_baselines.train --seed 0 --log-dir logs/POAR/srl_splits_a10r5i1f1/ --algo poar --num-timesteps 2000000 --env $env --structure srl_autoencoder --losses autoencoder:10:200 reward:5:-1 inverse:1:20 forward:1:10 --num-cpu 8 --gpu 0 -r
#python -m rl_baselines.train --seed 1 --log-dir logs/POAR/srl_splits_a10r5i1f1/ --algo poar --num-timesteps 2000000 --env $env --structure srl_autoencoder --losses autoencoder:10:200 reward:5:-1 inverse:1:20 forward:1:10 --num-cpu 8 --gpu 0 -r
#python -m rl_baselines.train --seed 2 --log-dir logs/POAR/srl_splits_a10r5i1f1/ --algo poar --num-timesteps 2000000 --env $env --structure srl_autoencoder --losses autoencoder:10:200 reward:5:-1 inverse:1:20 forward:1:10 --num-cpu 8 --gpu 0 -r
#
#
#### srl_decoupling_split
#### 0 - Generate datasets for SRL (random policy)
##python -m environments.dataset_generator --num-cpu 8 --name MobileRobot_Reach --env $env --num-episode 250 -f
#### Train SRL
##python train.py --data-folder data/MobileRobot_Reach  -bs 32 --epochs 2 --state-dim 230 --training-set-size 20000 --losses autoencoder:1:200 reward:5:-1 inverse:2:20 forward:1:10
#### Train policy
#cd ..
python -m rl_baselines.train --seed 0 --algo ppo2 --srl-model srl_splits --num-timesteps 2000000 --env $env --log-dir logs/POAR/srl_split_decoupling/  --num-cpu 8 --gpu 1 -r
python -m rl_baselines.train --seed 1 --algo ppo2 --srl-model srl_splits --num-timesteps 2000000 --env $env --log-dir logs/POAR/srl_split_decoupling/  --num-cpu 8 --gpu 1 -r
python -m rl_baselines.train --seed 2 --algo ppo2 --srl-model srl_splits --num-timesteps 2000000 --env $env --log-dir logs/POAR/srl_split_decoupling/  --num-cpu 8 --gpu 1 -r
#
#### srl_decoupling_combination
#python -m rl_baselines.train --seed 0 --algo ppo2 --srl-model srl_combination --num-timesteps 2000000 --env $env --log-dir logs/POAR/srl_combination_decoupling/  --num-cpu 8 --gpu 0 -r
#python -m rl_baselines.train --seed 1 --algo ppo2 --srl-model srl_combination --num-timesteps 2000000 --env $env --log-dir logs/POAR/srl_combination_decoupling/  --num-cpu 8 --gpu 0 -r
#python -m rl_baselines.train --seed 2 --algo ppo2 --srl-model srl_combination --num-timesteps 2000000 --env $env --log-dir logs/POAR/srl_combination_decoupling/  --num-cpu 8 --gpu 0 -r








### PPO2
#python -m rl_baselines.train --seed 0 --log-dir logs/POAR/ppo2/ --algo ppo2 --num-timesteps 2000000 --env $env  --srl-model raw_pixels --num-cpu 8 --gpu 1 -r --port 9999
#python -m rl_baselines.train --seed 1 --log-dir logs/POAR/ppo2/ --algo ppo2 --num-timesteps 2000000 --env $env  --srl-model raw_pixels --num-cpu 8 --gpu 1 -r --port 9999
#python -m rl_baselines.train --seed 2 --log-dir logs/POAR/ppo2/ --algo ppo2 --num-timesteps 2000000 --env $env  --srl-model raw_pixels --num-cpu 8 --gpu 1 -r --port 9999
#
#
##POAR with PPO liked autoencoder
#python -m rl_baselines.train --seed 0 --log-dir logs/POAR/ae/ --algo poar --structure  autoencoder --num-timesteps 2000000 --env $env  --srl-model raw_pixels --num-cpu 8 --gpu 1 -r --port 9999
#python -m rl_baselines.train --seed 1 --log-dir logs/POAR/ae/ --algo poar --structure  autoencoder --num-timesteps 2000000 --env $env  --srl-model raw_pixels --num-cpu 8 --gpu 1 -r --port 9999
#python -m rl_baselines.train --seed 2 --log-dir logs/POAR/ae/ --algo poar --structure  autoencoder --num-timesteps 2000000 --env $env  --srl-model raw_pixels --num-cpu 8 --gpu 1 -r --port 9999
#




#POAR with PPO liked autoencoder with batch norm
#python -m rl_baselines.train --seed 0 --log-dir logs/POAR/aeMlp/ --algo poar --structure  autoencoderMlp --num-timesteps 2000000 --env $env  --srl-model raw_pixels --num-cpu 8 --gpu 0 -r --port 9999
#python -m rl_baselines.train --seed 1 --log-dir logs/POAR/aeMlp/ --algo poar --structure  autoencoderMlp --num-timesteps 2000000 --env $env  --srl-model raw_pixels --num-cpu 8 --gpu 0 -r --port 9999
#python -m rl_baselines.train --seed 2 --log-dir logs/POAR/aeMlp/ --algo poar --structure  autoencoderMlp --num-timesteps 2000000 --env $env  --srl-model raw_pixels --num-cpu 8 --gpu 0 -r --port 9999


#POAR with PPO liked autoencoder with batch norm
#python -m rl_baselines.train --seed 0 --log-dir logs/POAR/aeBN/ --algo poar --structure  autoencoderBN --num-timesteps 2000000 --env $env  --srl-model raw_pixels --num-cpu 8 --gpu 0 -r --port 9999
#python -m rl_baselines.train --seed 1 --log-dir logs/POAR/aeBN/ --algo poar --structure  autoencoderBN --num-timesteps 2000000 --env $env  --srl-model raw_pixels --num-cpu 8 --gpu 0 -r --port 9999
#python -m rl_baselines.train --seed 2 --log-dir logs/POAR/aeBN/ --algo poar --structure  autoencoderBN --num-timesteps 2000000 --env $env  --srl-model raw_pixels --num-cpu 8 --gpu 0 -r --port 9999


#POAR with PPO liked autoencoder with batch norm
#python -m rl_baselines.train --seed 0 --log-dir logs/POAR/srl_split_a5i1f1/ --algo poar --structure  srl --num-timesteps 2000000 --env $env  --srl-model raw_pixels --num-cpu 8 --gpu 0 -r --port 9999
#python -m rl_baselines.train --seed 1 --log-dir logs/POAR/srl_split_a5i1f1/ --algo poar --structure  srl --num-timesteps 2000000 --env $env  --srl-model raw_pixels --num-cpu 8 --gpu 0 -r --port 9999
#python -m rl_baselines.train --seed 2 --log-dir logs/POAR/srl_split_a5i1f1/ --algo poar --structure  srl --num-timesteps 2000000 --env $env  --srl-model raw_pixels --num-cpu 8 --gpu 0 -r --port 9999


# PPO Ground Truth
#python -m rl_baselines.train --seed 0 --log-dir logs/POAR/GT/ --algo ppo2 --num-timesteps 2000000 --env $env  --srl-model ground_truth --num-cpu 8 --gpu 1 -r --port 9999
#python -m rl_baselines.train --seed 1 --log-dir logs/POAR/GT/ --algo pp02 --num-timesteps 2000000 --env $env  --srl-model ground_truth --num-cpu 8 --gpu 1 -r --port 9999
#python -m rl_baselines.train --seed 2 --log-dir logs/POAR/GT/ --algo ppo2 --num-timesteps 2000000 --env $env  --srl-model ground_truth --num-cpu 8 --gpu 1 -r --port 9999
