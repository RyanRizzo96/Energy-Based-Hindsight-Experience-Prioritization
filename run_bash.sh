#!/usr/bin/env bash
source ~/.virtualenvs/baseline_env/bin/activate
echo $HOME

export OPENAI_LOG_FORMAT=stdout,log,csv,tensorboard

# for seed in $(seq 0 4); do OPENAI_LOGDIR=/home/ubuntu/RL_baselines/.log/4_seeds/log/FetchSlide-$seed mpirun -np 16 python3 -m baselines.run --alg=her --env=FetchSlide-v1  --num_timesteps=400000 --seed=$seed --layers=3 --save_path=/home/ubuntu/RL_baselines/.log/4_seeds/plocicies/FetchSlide-$seed; done
for seed in $(seq 0 1); do OPENAI_LOGDIR=/home/ubuntu/Energy-Based-Prioritization/EBP_mod/FPUSH100/temperature_0.2-$seed mpirun -np 16 python3 -m baselines.run --alg=her --env=FetchPush-v1 --num_timesteps=100000 --seed=$seed; done
# python3 -m baselines.run --alg=her --env=FetchPickAndPlace-v1 --num_timesteps=5000

# mpirun -np 16 python3 -m baselines.run --alg=her --env=FetchPickAndPlace-v1 --num_timesteps=200000 --save_path=EBP_policies/FetchPAP200k/trial_1-seed4 --seed=4

