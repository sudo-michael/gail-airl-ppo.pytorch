# Train expert
python train_expert.py --cuda --env_id InvertedPendulum-v2 --num_steps 100000 --seed 0

# Collect Data
python collect_demo.py \
    --cuda --env_id InvertedPendulum-v2 \
    --weight weights/InvertedPendulum-v2.pth \
    --buffer_size 1000000 --std 0.01 --p_rand 0.0 --seed 0

# Train GAIL
python train_imitation.py \
    --algo gail --cuda --env_id InvertedPendulum-v2 \
    --buffer buffers/InvertedPendulum-v2/size1000000_std0.01_prand0.0.pth \
    --num_steps 100000 --eval_interval 5000 --rollout_length 2000 --seed 0

# 
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so

# bad data
python collect_demo.py --cuda --weight logs/Hopper-v3/sac/good/seed0-20211130-1100/model/step200000/actor.pth --buffer_size 1000000 --std 0.1 --p_rand 0.2 --seed 0 --name 200k_random_02


# hopper
python train_imitation.py  --algo gail --cuda --buffer buffers/size1000000_std0.0_prand0.0_last_expert.pth  --num_steps 100000 --eval_interval 5000 --rollout_length 2000 --seed 0

# 

python train_imitation.py ---buffer buffers/Hopper-v3/size1000000_std0.1_prand0.2_200k_random_02.pth --cuda --env_id Hopper-v3



size1000000_std0.0_prand0.0_step200k.pth



python train_imitation.py  --algo gail --cuda --buffer buffers/size1000000_std0.0_prand0.0_last_expert.pth  --num_steps 100000 --eval_interval 5000 --rollout_length 2000 --seed 0

# python merge_buffer.py --b1 buffers/Hopper-v3/size1000000_std0.0_prand0.0_last_expert.pth --b2 buffers/Hopper-v3/size1000000_std0.0_prand0.0_step200k.pth --env_id Hopper-v3 --name 14good34step200k