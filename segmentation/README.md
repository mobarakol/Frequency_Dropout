# Frequency_Dropout
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
