export CUDA_VISIBLE_DEVICES=1,2,3,4,5,7 

torchrun --nnode=1 --nproc_per_node=6 train_mamba.py \
    --model /nvme/hf_models/mamba-1.4b \
    --tokenizer /nvme/hf_models/EleutherAI/gpt-neox-20b \
    --learning_rate 5e-5 \
    --batch_size 18 \
    --gradient_accumulation_steps 1 \
    --optim paged_adamw_8bit \
    --num_epochs 3 \
    --output_dir /nvme/zecheng/ckpt/mamba-chat;