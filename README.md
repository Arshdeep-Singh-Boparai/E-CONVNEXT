# An efficient ConvNeXt-T evaluation on CIFAR-10 and Tiny ImageNet


We propose **Efficient ConvNeXt**[1], an efficient version of ConvNeXt [2] model. We apply operator-norm based filter pruning approach. Efficient ConvNeXt is accurate, more  efficient (with reduced parameter count and computations) than that of ConvNeXt. We provide efficient ConvNeXt-T model checkpoints to use the efficient version along with the 
[[ConvNeXt](https://github.com/facebookresearch/ConvNeXt/tree/main?tab=readme-ov-file) repository]. For the repostiory setup, please follow [2].


[1]  [Efficient CNNs via Passive Fitler Pruning](https://arxiv.org/pdf/2304.02319). Under review in IEEE TASLP.\
[Arshdeep Singh](https://www.surrey.ac.uk/people/arshdeep-singh), [Mark D PLumbley](https://www.surrey.ac.uk/people/mark-plumbley)\
[CVSSP](https://www.surrey.ac.uk/centre-vision-speech-signal-processing), University of Surrey, UK\


[2] [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545). CVPR 2022.\
[Zhuang Liu](https://liuzhuang13.github.io), [Hanzi Mao](https://hanzimao.me/), [Chao-Yuan Wu](https://chaoyuan.org/), [Christoph Feichtenhofer](https://feichtenhofer.github.io/), [Trevor Darrell](https://people.eecs.berkeley.edu/~trevor/) and [Saining Xie](https://sainingxie.com)\
Facebook AI Research, UC Berkeley\
[[`arXiv`](https://arxiv.org/abs/2201.03545)][[`video`](https://www.youtube.com/watch?v=QzCjXqFnWPE)]




## Installation
Please check [INSTALL.md](INSTALL.md) for installation instructions. 


## Checkpoints
Downloads the dataset, checkpoints from [this link](https://zenodo.org/records/14861717).



## Evaluation on CIFAR-10 dataset for various pruned models, unpruned models
Once the dataset, checkpoints are downloaded, use below instructions to reproduce the results with '~/script/eval_convnext.sh'.

```
python main.py --model convnext_tiny_unpruned_cifar10 --eval true \
--resume /path/to/checkpoint/cifar10/unpruned_checkpoint-best.pth \
--input_size 224 --drop_path 0 \
--data_path /path/to/dataset/CIFAR-10/ \
--eval_data_path /path/to/dataset/CIFAR-10/val \
```
This should give 
```
* Acc@1 95.700 Acc@5 99.890 loss 0.210
```


```
python main.py --model convnext_tiny_21M_pruned_cifar10 --eval true \
--resume /path/to/checkpoint/cifar10/21M_checkpoint-best.pth \
--input_size 224 --drop_path 0 \
--data_path /path/to/dataset/CIFAR-10/ \
--eval_data_path /path/to/dataset/CIFAR-10/val \
```
This should give 
```
* Acc@1 95.060 Acc@5 99.290 loss 0.254
```

```
python main.py --model convnext_tiny_16M_pruned_cifar10 --eval true \
--resume /path/to/checkpoint/cifar10/16M_checkpoint-best.pth \
--input_size 224 --drop_path 0 \
--data_path /path/to/dataset/CIFAR-10/ \
--eval_data_path /path/to/dataset/CIFAR-10/val \
```
This should give 
```
* Acc@1 95.030 Acc@5 99.140 loss 0.260
```

```
python main.py --model convnext_tiny_13M_pruned_cifar10 --eval true \
--resume /path/to/checkpoint/cifar10/13M_checkpoint-best.pth \
--input_size 224 --drop_path 0 \
--data_path /path/to/dataset/CIFAR-10/ \
--eval_data_path /path/to/dataset/CIFAR-10/val \
```
This should give 
```
* Acc@1 95.130 Acc@5 99.330 loss 0.249
```

```
python main.py --model convnext_tiny_3M_pruned_cifar10 --eval true \
--resume /path/to/checkpoint/cifar10/3M_checkpoint-best.pth \
--input_size 224 --drop_path 0 \
--data_path /path/to/dataset/CIFAR-10/ \
--eval_data_path /path/to/dataset/CIFAR-10/val \\
```
This should give 
```
* Acc@1 90.240 Acc@5 99.560 loss 0.364
```


## Evaluation on Tiny ImageNet dataset for various pruned models, unpruned models
```
python main.py --model convnext_tiny_unpruned_tinyimagenet --eval true \
--resume /path/to/checkpoint/tinyimage/unpruned_checkpoint-best.pth \
--input_size 224 --drop_path 0 \
--data_path /path/to/dataset/TinyImageNet_dataset/ \
--eval_data_path /path/to/dataset/TinyImageNet_dataset/val \
```

This should give 
```
* Acc@1 78.170 Acc@5 93.510 loss 0.916
```

```
python main.py --model convnext_tiny_21M_pruned_tinyimagenet --eval true \
--resume /path/to/checkpoint//tinyimage/21M_mixup_checkpoint-best.pth \
--input_size 224 --drop_path 0 \
--data_path /path/to/dataset/TinyImageNet_dataset/ \
--eval_data_path /path/to/dataset/TinyImageNet_dataset/val \
```

This should give 
```
* Acc@1 75.410 Acc@5 90.220 loss 1.163
```

```
python main.py --model convnext_tiny_21M_pruned_tinyimagenet --eval true \
--resume /path/to/checkpoint//tinyimage/21M_checkpoint-best.pth \
--input_size 224 --drop_path 0 \
--data_path /path/to/dataset/TinyImageNet_dataset/ \
--eval_data_path /path/to/dataset/TinyImageNet_dataset/val \
```

This should give 
```
* Acc@1 74.230 Acc@5 88.550 loss 1.276
```

```
python main.py --model convnext_tiny_16M_pruned_tinyimagenet --eval true \
--resume /path/to/checkpoint//tinyimage/16M_checkpoint-best.pth \
--input_size 224 --drop_path 0 \
--data_path /path/to/dataset/TinyImageNet_dataset/ \
--eval_data_path /path/to/dataset/TinyImageNet_dataset/val \
```

This should give 
```
* Acc@1 73.940 Acc@5 88.620 loss 1.282
```



```
python main.py --model convnext_tiny_13M_pruned_tinyimagenet --eval true \
--resume /path/to/checkpoint//tinyimage/13M_checkpoint-best.pth \
--input_size 224 --drop_path 0 \
--data_path /path/to/dataset/TinyImageNet_dataset/ \
--eval_data_path /path/to/dataset/TinyImageNet_dataset/val \
```

This should give 
```
* Acc@1 73.090 Acc@5 88.130 loss 1.337
```

```
python main.py --model convnext_tiny_3M_pruned_tinyimagenet --eval true \
--resume /path/to/checkpoint//tinyimage/3M_checkpoint-best.pth \
--input_size 224 --drop_path 0 \
--data_path /path/to/dataset/TinyImageNet_dataset/ \
--eval_data_path /path/to/dataset/TinyImageNet_dataset/val \
```

This should give 
```
* Acc@1 59.940 Acc@5 82.280 loss 1.704
```




## Training
See [TRAINING.md](TRAINING.md) for training and fine-tuning instructions, as suggested by [ConvNeXt](https://github.com/facebookresearch/ConvNeXt/tree/main?tab=readme-ov-file) repository.

## Pruning
For pruning algorithm, see [pruning.md](pruning.md)



## Acknowledgement
This repository is built using the [ConvNeXt](https://github.com/facebookresearch/ConvNeXt/tree/main?tab=readme-ov-file) repository. 

## Citation
If you find this repository helpful, please consider citing:
```
@Article{Singh2023efficient,
  author  = {Arshdeep Singh, Mark D Plumbley},
  title   = {Efficient CNNs via passive filter pruning},
  journal = {arXiv preprint arXiv:2304.02319},
  year    = {2023},
}
```

