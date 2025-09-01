# Efficient ConvNeXt-T: Evaluation on CIFAR-10 and Tiny ImageNet

We introduce **Efficient ConvNeXt** [1], an efficient variant of ConvNeXt [2], obtained via an operator-norm-based filter pruning approach. Efficient ConvNeXt achieves competitive accuracy while significantly reducing parameter count and computational cost compared to the original ConvNeXt.  

This repository provides checkpoints of Efficient ConvNeXt-T, enabling direct use of the efficient models alongside the official [ConvNeXt repository](https://github.com/facebookresearch/ConvNeXt/tree/main). For repository setup, please follow the instructions in [2].

---

## 🔗 References
- **Efficient ConvNeXt (Ours):** [Efficient CNNs via Passive Filter Pruning](https://ieeexplore.ieee.org/document/10966165), *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, 2025.  
  [Arshdeep Singh](https://www.surrey.ac.uk/people/arshdeep-singh), [Mark D. Plumbley](https://www.surrey.ac.uk/people/mark-plumbley)  
  [CVSSP](https://www.surrey.ac.uk/centre-vision-speech-signal-processing), University of Surrey, UK  

- **ConvNeXt (Baseline):** [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545), CVPR 2022.  
  [Zhuang Liu](https://liuzhuang13.github.io), [Hanzi Mao](https://hanzimao.me/), [Chao-Yuan Wu](https://chaoyuan.org/), [Christoph Feichtenhofer](https://feichtenhofer.github.io/), [Trevor Darrell](https://people.eecs.berkeley.edu/~trevor/), [Saining Xie](https://sainingxie.com)  

---

## ⚙️ Installation
Please refer to [INSTALL.md](INSTALL.md) for detailed installation instructions.  

---

## 📥 Checkpoints and Datasets
All checkpoints and datasets can be downloaded from [Zenodo](https://zenodo.org/records/14861717).  

- CIFAR-10 checkpoints (unpruned and pruned versions)  
- Tiny ImageNet checkpoints (unpruned and pruned versions)  

---

## 📊 Evaluation

### Evaluation on **CIFAR-10**
Once the dataset and checkpoints are downloaded, you can reproduce results using the provided script:  

```bash
python main.py --model convnext_tiny_unpruned_cifar10 --eval true   --resume /path/to/checkpoint/cifar10/unpruned_checkpoint-best.pth   --input_size 224 --drop_path 0   --data_path /path/to/dataset/CIFAR-10/   --eval_data_path /path/to/dataset/CIFAR-10/val
```

Expected result:  
```
* Acc@1 95.70  Acc@5 99.89  loss 0.210
```

Other pruned variants:

| Model                              | #Params | Top-1 Acc. | Top-5 Acc. | Loss  |
|-----------------------------------|---------|------------|------------|-------|
| convnext_tiny_unpruned_cifar10    | 28M     | 95.70      | 99.89      | 0.210 |
| convnext_tiny_21M_pruned_cifar10  | 21M     | 95.06      | 99.29      | 0.254 |
| convnext_tiny_16M_pruned_cifar10  | 16M     | 95.03      | 99.14      | 0.260 |
| convnext_tiny_13M_pruned_cifar10  | 13M     | 95.13      | 99.33      | 0.249 |
| convnext_tiny_3M_pruned_cifar10   | 3M      | 90.24      | 99.56      | 0.364 |

---

### Evaluation on **Tiny ImageNet**
Example (unpruned model):

```bash
python main.py --model convnext_tiny_unpruned_tinyimagenet --eval true   --resume /path/to/checkpoint/tinyimage/unpruned_checkpoint-best.pth   --input_size 224 --drop_path 0   --data_path /path/to/dataset/TinyImageNet_dataset/   --eval_data_path /path/to/dataset/TinyImageNet_dataset/val
```

Expected result:  
```
* Acc@1 78.17  Acc@5 93.51  loss 0.916
```

Other pruned variants:

| Model                                   | #Params | Top-1 Acc. | Top-5 Acc. | Loss  |
|----------------------------------------|---------|------------|------------|-------|
| convnext_tiny_unpruned_tinyimagenet    | 28M     | 78.17      | 93.51      | 0.916 |
| convnext_tiny_21M_pruned_tinyimagenet  | 21M     | 75.41      | 90.22      | 1.163 |
| convnext_tiny_21M_pruned_tinyimagenet* | 21M     | 74.23      | 88.55      | 1.276 |
| convnext_tiny_16M_pruned_tinyimagenet  | 16M     | 73.94      | 88.62      | 1.282 |
| convnext_tiny_13M_pruned_tinyimagenet  | 13M     | 73.09      | 88.13      | 1.337 |
| convnext_tiny_3M_pruned_tinyimagenet   | 3M      | 59.94      | 82.28      | 1.704 |

> *Two different 21M checkpoints are provided (`21M_mixup_checkpoint-best.pth` and `21M_checkpoint-best.pth`).  

---

## 🚀 Training
Training and fine-tuning instructions are provided in [TRAINING.md](TRAINING.md), following the methodology from the official [ConvNeXt repository](https://github.com/facebookresearch/ConvNeXt/tree/main).  

---

## ✂️ Pruning
Details of the pruning algorithm can be found in [Pruning/readme.md](Pruning/readme.md).  

---

## 🙏 Acknowledgement
This repository builds upon the excellent [ConvNeXt repository](https://github.com/facebookresearch/ConvNeXt).  

---

## 📖 Citation
If you find this repository useful, please cite:  

```bibtex
@ARTICLE{10966165,
  author={Singh, Arshdeep and Plumbley, Mark D.},
  journal={IEEE/ACM Transactions on Audio, Speech and Language Processing}, 
  title={Efficient CNNs via Passive Filter Pruning}, 
  year={2025},
  volume={33},
  pages={1763-1774},
  doi={10.1109/TASLPRO.2025.3561589},
  keywords={Passive filters; Filter pruning; CNNs; ConvNeXt; DCASE; image classification; low-complexity; PANNs; ResNet50; VGGish}
}
```
