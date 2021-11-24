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
