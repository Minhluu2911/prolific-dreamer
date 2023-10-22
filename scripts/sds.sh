#!/bin/sh

# Image generation with prolific_dream 2d 

# 1 code book
### sds cfg 7.5
python prolific_dreamer2d.py \
        --num_steps 10000 --log_steps 50 --lr 0.03 \
        --model_path 'stabilityai/stable-diffusion-2-1-base' \
        --loss_weight '1m_alphas_cumprod' \
        --t_schedule random --generation_mode 'sds' \
        --prompt "a photograph of an astronaut riding a horse" \
        --height 512 --width 512 --batch_size 1 --guidance_scale 7.5 \
        --log_progress true --save_x0 true \
        --work_dir 'work_dir/one_code_book/sds_cfg_7.5' \
        --codebook_interpolate false


### sds cfg 100.0
python prolific_dreamer2d.py \
        --num_steps 10000 --log_steps 50 --lr 0.03 \
        --model_path 'stabilityai/stable-diffusion-2-1-base' \
        --loss_weight '1m_alphas_cumprod' \
        --t_schedule random --generation_mode 'sds' \
        --prompt "a photograph of an astronaut riding a horse" \
        --height 512 --width 512 --batch_size 1 --guidance_scale 100.0 \
        --log_progress true --save_x0 true \
        --work_dir 'work_dir/one_code_book/sds_cfg_100.0' \
        --codebook_interpolate false

### sds cfg 7.5 dreamtime
python prolific_dreamer2d.py \
        --num_steps 10000 --log_steps 50 --lr 0.03 \
        --model_path 'stabilityai/stable-diffusion-2-1-base' \
        --loss_weight_type 'dreamtime' --t_schedule 'dreamtime' \
        --generation_mode 'sds' \
        --prompt "a photograph of an astronaut riding a horse" \
        --height 512 --width 512 --batch_size 1 --guidance_scale 7.5 \
        --log_progress true --save_x0 true \
        --work_dir 'work_dir/one_code_book/sds_cfg_7.5_dreamtime' \
        --codebook_interpolate false


### sds cfg 100.0 dreamtime
python prolific_dreamer2d.py \
        --num_steps 10000 --log_steps 50 --lr 0.03 \
        --model_path 'stabilityai/stable-diffusion-2-1-base' \
        --loss_weight_type 'dreamtime' --t_schedule 'dreamtime' \
        --generation_mode 'sds' \
        --prompt "a photograph of an astronaut riding a horse" \
        --height 512 --width 512 --batch_size 1 --guidance_scale 100.0 \
        --log_progress true --save_x0 true \
        --work_dir 'work_dir/one_code_book/sds_cfg_100.0_dreamtime' \
        --codebook_interpolate false


# 2 code book and interpolate
### sds cfg 7.5
python prolific_dreamer2d.py \
        --num_steps 10000 --log_steps 50 --lr 0.03 \
        --model_path 'stabilityai/stable-diffusion-2-1-base' \
        --loss_weight '1m_alphas_cumprod' \
        --t_schedule random --generation_mode 'sds' \
        --prompt "a photograph of an astronaut riding a horse" \
        --height 512 --width 512 --batch_size 1 --guidance_scale 7.5 \
        --log_progress true --save_x0 true \
        --work_dir 'work_dir/interpolate_code_book/sds_cfg_7.5' \
        --codebook_interpolate true


### sds cfg 100.0
python prolific_dreamer2d.py \
        --num_steps 10000 --log_steps 50 --lr 0.03 \
        --model_path 'stabilityai/stable-diffusion-2-1-base' \
        --loss_weight '1m_alphas_cumprod' \
        --t_schedule random --generation_mode 'sds' \
        --prompt "a photograph of an astronaut riding a horse" \
        --height 512 --width 512 --batch_size 1 --guidance_scale 100.0 \
        --log_progress true --save_x0 true \
        --work_dir 'work_dir/interpolate_code_book/sds_cfg_100.0' \
        --codebook_interpolate true

### sds cfg 7.5 dreamtime
python prolific_dreamer2d.py \
        --num_steps 10000 --log_steps 50 --lr 0.03 \
        --model_path 'stabilityai/stable-diffusion-2-1-base' \
        --loss_weight_type 'dreamtime' --t_schedule 'dreamtime' \
        --generation_mode 'sds' \
        --prompt "a photograph of an astronaut riding a horse" \
        --height 512 --width 512 --batch_size 1 --guidance_scale 7.5 \
        --log_progress true --save_x0 true \
        --work_dir 'work_dir/interpolate_code_book/sds_cfg_7.5_dreamtime' \
        --codebook_interpolate true


### sds cfg 100.0 dreamtime
python prolific_dreamer2d.py \
        --num_steps 10000 --log_steps 50 --lr 0.03 \
        --model_path 'stabilityai/stable-diffusion-2-1-base' \
        --loss_weight_type 'dreamtime' --t_schedule 'dreamtime' \
        --generation_mode 'sds' \
        --prompt "a photograph of an astronaut riding a horse" \
        --height 512 --width 512 --batch_size 1 --guidance_scale 100.0 \
        --log_progress true --save_x0 true \
        --work_dir 'work_dir/interpolate_code_book/sds_cfg_100.0_dreamtime' \
        --codebook_interpolate true


# tune vsd
# 1 code book
### vsd cfg 7.5
python prolific_dreamer2d.py \
        --num_steps 10000 --log_steps 50 \
        --seed 1024 --lr 0.001 --phi_lr 0.0001 --use_t_phi true \
        --model_path 'stabilityai/stable-diffusion-2-1-base' \
        --loss_weight_type '1m_alphas_cumprod' --t_schedule 'random' \
        --generation_mode 'vsd' \
        --phi_model 'lora' --lora_scale 1. --lora_vprediction True \
        --prompt "a photograph of an astronaut riding a horse" \
        --height 512 --width 512 --batch_size 1 --guidance_scale 7.5 \
        --log_progress true --save_x0 true --save_phi_model true \
        --work_dir 'work_dir/one_code_book/tune_vsd/vsd_cfg_7.5_lr0.001' \
        --codebook_interpolate false \

### vsd cfg 100
python prolific_dreamer2d.py \
        --num_steps 10000 --log_steps 50 \
        --seed 1024 --lr 0.001 --phi_lr 0.0001 --use_t_phi true \
        --model_path 'stabilityai/stable-diffusion-2-1-base' \
        --loss_weight_type '1m_alphas_cumprod' --t_schedule 'random' \
        --generation_mode 'vsd' \
        --phi_model 'lora' --lora_scale 1. --lora_vprediction True \
        --prompt "a photograph of an astronaut riding a horse" \
        --height 512 --width 512 --batch_size 1 --guidance_scale 100.0 \
        --log_progress true --save_x0 true --save_phi_model true \
        --work_dir 'work_dir/one_code_book/tune_vsd/vsd_cfg_100.0_lr0.001' \
        --codebook_interpolate false