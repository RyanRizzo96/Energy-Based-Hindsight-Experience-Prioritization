#!/usr/bin/env bash
source ~/.virtualenvs/baseline_env/bin/activate
echo $HOME

export OPENAI_LOG_FORMAT=stdout,log,csv,tensorboard
# for seed in $(seq 4 5); do OPENAI_LOGDIR=/home/ubuntu/RL_baselines/.log/two_seeds/run_1/FPAP_200k-$seed mpirun -np 16 python3 -m baselines.run --alg=her --env=FetchPickAndPlace-v1 --num_timesteps=200000 --seed=$seed --layers=3; done
# for seed in $(seq 0 5); do OPENAI_LOGDIR=/home/ubuntu/RL_baselines/.log/run_1/layers_6-$seed mpirun -np 16 python3 -m baselines.run --alg=her --env=FetchPush-v1 --num_timesteps=10000 --seed=$seed --layers=6; done

# python3 -m baselines.run --alg=her --env=FetchReach-v1 --num_timesteps=5000 --log_path=.log/tensorboard_test/actor_critic_loss/run4

mpirun -np 16 python3 -m baselines.run --alg=her --env=FetchReach-v1 --num_timesteps=10000 --save_path=EBP_policies/initial_tests/FetchReach/trial_1