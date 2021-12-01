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


