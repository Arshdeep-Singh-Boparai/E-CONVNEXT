#!bin/bash
python /path/to/directory/ConvNeXt/main.py --epochs 100 \
                --model convnext_tiny \
                --data_set image_folder \
                --data_path /path/to/data/CIFAR-10/train \
                --eval_data_path /path/to/data/CIFAR-10/test \
                --nb_classes 10 \
                --num_workers 1 \
                --warmup_epochs 0 \
                --save_ckpt true \
                --output_dir model_ckpt \
                --finetune /path/to/pruned/checkpoints/convnext_tiny_1k_224_ema.pth \
                --cutmix 0 \
                --mixup 0 --lr 4e-4 \
                --enable_wandb true --wandb_ckpt true
