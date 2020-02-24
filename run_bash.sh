#!/usr/bin/env bash
source ~/.virtualenvs/baseline_env/bin/activate

echo $HOME
#/Users/ryanr/B.Eng/MCAST_Degree_4/Thesis/code/gym/RL_baselines/new_logs/initial_tests/FetchReach_5k/trial_logdir
# echo $PWD

export OPENAI_LOG_FORMAT=stdout,log,csv,tensorboard
# PYTHONHUNTERCONFIG="stdlib=False,force_colors=True,source__contains="train
# python3 -m baselines.run --alg=her --env=FetchReach-v1  --num_timesteps=5000

#coverage run -m baselines.run --alg=her --env=FetchReach-v1  --num_timesteps=5000
#coverage report -m -i
#coverage html -i

# python3 -m baselines.run --alg=her --env=FetchPush-v1 --num_timesteps=0 --load_path=/Users/ryanr/B.Eng/MCAST_Degree_4/Thesis/code/baseline_code/baselines/policies/her/FPUSH_200k/standard --play

#for seed in $(seq 5 6); do OPENAI_LOGDIR=/home/ubuntu/RL_baselines/.log/two_seeds/FPUSH_200k-$seed mpirun -np 16 python3 -m baselines.run --alg=her --env=FetchPush-v1 --num_timesteps=200000 --seed=$seed ; done
#for seed in $(seq 5 6); do OPENAI_LOGDIR=/home/ubuntu/RL_baselines/.log/two_seeds/FPUSH_200k-$seed mpirun -np 16 python3 -m baselines.run --alg=her --env=FetchPush-v1 --num_timesteps=200000 --seed=$seed --layers=3; done

# for seed in $(seq 4 5); do OPENAI_LOGDIR=/home/ubuntu/RL_baselines/.log/two_seeds/run_1/FPAP_200k-$seed mpirun -np 16 python3 -m baselines.run --alg=her --env=FetchPickAndPlace-v1 --num_timesteps=200000 --seed=$seed --layers=3; done
# for seed in $(seq 0 5); do OPENAI_LOGDIR=/home/ubuntu/RL_baselines/.log/run_1/layers_6-$seed mpirun -np 16 python3 -m baselines.run --alg=her --env=FetchPush-v1 --num_timesteps=10000 --seed=$seed --layers=6; done

# python3 -m baselines.run --alg=her --env=FetchReach-v1 --num_timesteps=5000 --log_path=.log/tensorboard_test/actor_critic_loss/run4

python3 -m baselines.run --alg=her --env=FetchReach-v1 --num_timesteps=500
# TEST 2
# mpirun -np 19 python3 -m baselines.run --alg=her --env=FetchPickAndPlace-v1  --num_timesteps=1000000 --save_path=.models/her/FPAP_1mil_run3

# mpirun -np 19 python3 -m baselines.run --alg=her --env=FetchPickAndPlace-v1  --num_timesteps=1000 --save_path=.models/her/ec2_test_5k_test

#OPENAI_LOGDIR=$HOME/B.Eng/MCAST_Degree_4/Thesis/code/gym/RL_baselines/new_logs/initial_tests/FetchReach_5k/trial_logdir
# mpirun -np 16 python3 -m baselines.run --alg=her --env=FetchPush-v1 --num_timesteps=200000 --save_path=new_policies/initial_tests/FetchPush_200k/trial_1 --log_path=new_logs/initial_tests/FetchPush_200k/trial_1

# mpirun -np 16

# mpirun -np 19 python3 -m baselines.run --num_env=5 --alg=her --env=FetchReach-v1 --num_timesteps=5000

# python3 csv_plot.py

# python3 /Users/ryanr/B.Eng/MCAST_Degree_4/Thesis/code/gym/RL_baselines/baselines/her/experiment/plot.py --/Users/ryanr/B.Eng/MCAST_Degree_4/Thesis/code/gym/RL_baselines/.log/two_seeds/saved/run_1/Fetch_Pick_And_Place/FPAP_200k-4