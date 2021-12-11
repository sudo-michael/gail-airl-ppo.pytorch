# Train Experts
```
python train_expert.py --cuda --env_id InvertedPendulum-v2 --num_steps 100000 --seed 0 --run_name inverted_expert
python train_expert.py --cuda --env_id Hopper-v3 --num_steps 350000 --seed 0 --run_name hopper_expert
python train_expert.py --cuda --env_id LunarLanderContinuous-v2 --num_steps 500000 --seed 0 --run_name ll_expert
 ```

# Gather Data
## Inverted Pendulum-v2
```
python collect_demo.py \
    --cuda --env_id InvertedPendulum-v2 \
    --weight logs/InvertedPendulum-v2/sac/inverted_expert/seed0-20211210-1816/model/step100000/actor.pth \
    --buffer_size 1000000 --std 0.00 --p_rand 0.0 --seed 0 --name expert
```
Mean return of the expert is 1000.0
```
python collect_demo.py \
    --cuda --env_id InvertedPendulum-v2 \
    --weight logs/InvertedPendulum-v2/sac/inverted_expert/seed0-20211210-1816/model/step100000/actor.pth \
    --buffer_size 1000000 --std 0.01 --p_rand 0.1 --seed 0 --name std001prand01
```
Mean return of the expert is 114.875
```
python collect_demo.py \
    --cuda --env_id InvertedPendulum-v2 \
    --weight logs/InvertedPendulum-v2/sac/inverted_expert/seed0-20211210-1816/model/step100000/actor.pth \
    --buffer_size 1000000 --std 0.05 --p_rand 0.1 --seed 0 --name std005prand01
```
Mean return of the expert is 108.3

```
python collect_demo.py \
    --cuda --env_id InvertedPendulum-v2 \
    --weight logs/InvertedPendulum-v2/sac/inverted_expert/seed0-20211210-1816/model/step100000/actor.pth \
    --buffer_size 1000000 --std 0.05 --p_rand 0.2 --seed 0 --name std005prand02
```
Mean return of the expert is 41.6

## LunarLanderContinuous-v2
```
python collect_demo.py \
    --cuda --env_id LunarLanderContinuous-v2 \
    --weight logs/LunarLanderContinuous-v2/sac/ll_expert/seed0-20211210-1717/model/step500000/actor.pth \
    --buffer_size 1000000 --std 0.00 --p_rand 0.0 --seed 0 --name expert
```
Mean return of the expert is 281.02710092596993

```
python collect_demo.py \
    --cuda --env_id LunarLanderContinuous-v2 \
    --weight logs/LunarLanderContinuous-v2/sac/ll_expert/seed0-20211210-1717/model/step500000/actor.pth \
    --buffer_size 1000000 --std 0.01 --p_rand 0.2 --seed 0 --name std001prand02
```
Mean return of the expert is 239.58554752707735
```
python collect_demo.py \
    --cuda --env_id LunarLanderContinuous-v2 \
    --weight logs/LunarLanderContinuous-v2/sac/ll_expert/seed0-20211210-1717/model/step500000/actor.pth \
    --buffer_size 1000000 --std 0.05 --p_rand 0.3 --seed 0 --name std005prand03
```
Mean return of the expert is 129.53273970584647
```
python collect_demo.py \
    --cuda --env_id LunarLanderContinuous-v2 \
    --weight logs/LunarLanderContinuous-v2/sac/ll_expert/seed0-20211210-1717/model/step500000/actor.pth \
    --buffer_size 1000000 --std 0.05 --p_rand 0.4 --seed 0 --name std005prand04
```
Mean return of the expert is 58.97021609555025


# Hopper-v3
```
python collect_demo.py \
    --cuda --env_id Hopper-v3 \
    --weight logs/Hopper-v3/sac/hopper_expert/seed0-20211210-1817/model/step350000/actor.pth  \
    --buffer_size 1000000 --std 0.00 --p_rand 0.0 --seed 0 --name expert
```
Mean return of the expert is 3325.183129272476
```
python collect_demo.py \
    --cuda --env_id Hopper-v3 \
    --weight logs/Hopper-v3/sac/hopper_expert/seed0-20211210-1817/model/step350000/actor.pth  \
    --buffer_size 1000000 --std 0.01 --p_rand 0.1 --seed 0 --name std001prand01
```
Mean return of the expert is 969.3244951468977
```
python collect_demo.py \
    --cuda --env_id Hopper-v3 \
    --weight logs/Hopper-v3/sac/hopper_expert/seed0-20211210-1817/model/step350000/actor.pth  \
    --buffer_size 1000000 --std 0.05 --p_rand 0.1 --seed 0 --name std005prand01
```
Mean return of the expert is 909.5563037007959
```
python collect_demo.py \
    --cuda --env_id Hopper-v3 \
    --weight logs/Hopper-v3/sac/hopper_expert/seed0-20211210-1817/model/step350000/actor.pth  \
    --buffer_size 1000000 --std 0.05 --p_rand 0.2 --seed 0 --name std005prand02
```
Mean return of the expert is 605.6933871114717

# Imitation Learning
## Inverted Pendulum-v2
python train_imitation.py \
    --algo gail --cuda --env_id InvertedPendulum-v2 \
    --buffer buffers/InvertedPendulum-v2/size1000000_std0.05_prand0.1_std005prand01.pth   \
    --num_steps 100000 --eval_interval 5000 --rollout_length 2000 --seed 0 \
    --run_name std005prand01

python train_imitation.py \
    --algo gail --cuda --env_id InvertedPendulum-v2 \
    --buffer buffers/InvertedPendulum-v2/size1000000_std0.0_prand0.0_expert.pth \
    --num_steps 100000 --eval_interval 5000 --rollout_length 2000 --seed 0 \
    --run_name expert

python train_imitation.py \
    --algo gail --cuda --env_id InvertedPendulum-v2 \
    --buffer buffers/InvertedPendulum-v2/size1000000_std0.01_prand0.1_std001prand01.pth   \
    --num_steps 100000 --eval_interval 5000 --rollout_length 2000 --seed 0 \
    --run_name std001prand01

python train_imitation.py \
    --algo gail --cuda --env_id InvertedPendulum-v2 \
    --buffer buffers/InvertedPendulum-v2/size1000000_std0.05_prand0.2_std005prand02.pth \
    --num_steps 100000 --eval_interval 5000 --rollout_length 2000 --seed 0 \
    --run_name std005prand02

## Hopper-v3
python train_imitation.py \
    --algo gail --cuda --env_id Hopper-v3 \
    --buffer buffers/Hopper-v3/size1000000_std0.01_prand0.1_std001prand01.pth \
    --num_steps 100000 --eval_interval 5000 --rollout_length 50000 --seed 0 \
    --run_name std001prand01

python train_imitation.py \
    --algo gail --cuda --env_id Hopper-v3 \
    --buffer buffers/Hopper-v3/size1000000_std0.05_prand0.1_std005prand01.pth \
    --num_steps 100000 --eval_interval 5000 --rollout_length 50000 --seed 0 \
    --run_name stdo005prand01

python train_imitation.py \
    --algo gail --cuda --env_id Hopper-v3 \
    --buffer buffers/Hopper-v3/size1000000_std0.05_prand0.2_std005prand02.pth \
    --num_steps 100000 --eval_interval 5000 --rollout_length 50000 --seed 0 \
    --run_name stdo005prand02

python train_imitation.py \
    --algo gail --cuda --env_id Hopper-v3 \
    --buffer buffers/Hopper-v3/size1000000_std0.0_prand0.0_expert.pth \
    --num_steps 100000 --eval_interval 5000 --rollout_length 50000 --seed 0 \
    --run_name expert




# Mujoco Rendering Issues
```
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
```

