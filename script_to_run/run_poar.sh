#!/bin/bash

env="MobileRobotGymEnv-v0"

cd ..
#env="OmnirobotEnv-v0"
#
#

#
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
python -m rl_baselines.train --seed 0 --log-dir logs/POAR/GT/ --algo ppo2 --num-timesteps 2000000 --env $env  --srl-model ground_truth --num-cpu 8 --gpu 1 -r --port 9999
python -m rl_baselines.train --seed 1 --log-dir logs/POAR/GT/ --algo pp02 --num-timesteps 2000000 --env $env  --srl-model ground_truth --num-cpu 8 --gpu 1 -r --port 9999
python -m rl_baselines.train --seed 2 --log-dir logs/POAR/GT/ --algo ppo2 --num-timesteps 2000000 --env $env  --srl-model ground_truth --num-cpu 8 --gpu 1 -r --port 9999
