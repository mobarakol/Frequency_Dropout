# Frequency_Dropout
## Classification tasks
The code is adopted from a close baseline model CBS (https://github.com/pairlab/CBS) <br>
To train Frequency Dropout for Randomized Filtering (FD-RF) model

```
python main.py --dataset cifar100 --alg res --data ./data/  --num_epochs 300
```

To train Frequency Dropout for Gaussian Filtering (FD-GF) model

```
python main.py --dataset cifar100 --alg res --data ./data/  --num_epochs 300 --use_gf

```
To train CBS model

```
python main.py --dataset cifar100 --alg res --data ./data/  --num_epochs 300 --use_cbs
```

## Segmentation tasks
The dataset is publicly available in the challenge portal: <br>
[Multi-Centre, Multi-Vendor & Multi-Disease Cardiac Image Segmentation Challenge (M&Ms)](https://www.ub.edu/mnms/)

To train Frequency Dropout for Randomized Filtering (FD-RF) model

```
 python main.py --batch_size 4  --epochs 400 --train_vendor A --freq_max_all 16 16 3 --dropout_p_all .5 .55 .8 --use_fd
```

To train baseline model
```
python main.py --batch_size 4  --epochs 400 --train_vendor A --freq_max_all 16 16 3 --dropout_p_all 1 1 1
```
To train CBS model
```
python main.py --batch_size 4  --epochs 400 --train_vendor A --freq_max_all 16 16 3 --dropout_p_all 1 1 1 --use_cbs
```

The paper can be cited by using below bibtex.

```bibtex
@inproceedings{islam2022frequency,
  title={Frequency Dropout: Feature-Level Regularization via Randomized Filtering},
  author={Islam, Mobarakol and Glocker, Ben},
  booktitle={ECCV 2022 MEDICAL COMPUTER VISION WORKSHOP},
  year={2022},
}
```
