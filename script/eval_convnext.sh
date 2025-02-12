export CUDA_VISIBLE_DEVICES=0
python main.py --model convnext_tiny_3M_pruned_cifar10 --eval true \
--resume /home/arshdeep/ConvNeXt/github_checkpoints/cifar10/3M_checkpoint-best.pth \
--input_size 224 --drop_path 0 \
--data_path /home/arshdeep/ConvNeXt/dataset/CIFAR-10/ \
--eval_data_path /home/arshdeep/ConvNeXt/dataset/CIFAR-10/val \

