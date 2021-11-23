
import argparse
import os
import random
#from random import random
import numpy as np
import pathlib
import torch
from unet_model import unet
from unet_model import DataAugmenter
from datasets import get_datasets_mnms_single_vendor, determinist_collate
from evaluation_metrics import EDiceLoss
import copy

def seed_everything(seed=12):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def step_train(data_loader, model, criterion, metric, optimizer): 
    model.train()   
    data_aug = DataAugmenter(p=0.8).cuda()
    for i, batch in enumerate(data_loader):
        targets = batch["label"].squeeze(1).cuda(non_blocking=True)
        inputs = batch["image"].float().cuda()
        inputs = data_aug(inputs)
        segs = model(inputs)
        segs = data_aug.reverse(segs)
        loss_ = criterion(segs, targets)
        optimizer.zero_grad()
        loss_.backward()
        optimizer.step()
        
def step_valid(data_loader, model, criterion, metric):
    model.eval()
    losses, metrics, vendors = [], [], []
    for i, batch in enumerate(data_loader):
        targets = batch["label"].squeeze(1).cuda(non_blocking=True)
        inputs = batch["image"].float().cuda()
        vendors.extend(batch["vendor"])
        segs = model(inputs)
        loss_ = criterion(segs, targets)
        segs = segs.data.max(1)[1].squeeze_(1)
        metric_ = metric(segs.detach().cpu(), targets.detach().cpu())
        metrics.extend(metric_)
        losses.append(loss_.item())
        
    return np.mean(losses), metrics
    
def main():
    parser = argparse.ArgumentParser(description='Brats Training')
    parser.add_argument('--lr', default=1e-4, type=float,help='initial learning rate')
    parser.add_argument('--weight_decay', '--weight-decay', default=0., type=float, help='weight decay')
    parser.add_argument('--batch_size', default=2, type=int,help='mini-batch size')
    parser.add_argument('--num_classes', default=4, type=int, help="num of classes")
    parser.add_argument('--in_channels', default=1, type=int, help="num of input channels")
    parser.add_argument('--ls_smoothing', default=0.0, type=float, help='LS smoothing factor')
    parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
    parser.add_argument('--data_root', default='/vol/biomedic3/mi615/datasets/MNMS', help='data directory')
    parser.add_argument('--use_base', default=False, action='store_true', help='base vs CBS or FD')
    parser.add_argument('--use_cbs', default=False, action='store_true', help='cbs or not')
    parser.add_argument('--use_fd', default=False, action='store_true', help='frequency drop or not')
    parser.add_argument('--std', default=1, type=float)
    parser.add_argument('--std_factor', default=0.9, type=float)
    parser.add_argument('--cbs_epoch', default=5, type=int)
    parser.add_argument('--kernel_size', default=3, type=int)
    parser.add_argument('--train_vendor', default='A', help="options:[A,B]")
    parser.add_argument('--freq_min_all', default=[0.2, 0.2, 0], nargs='+', type=float)
    parser.add_argument('--freq_max_all', default=[1.0, 3.0, 1.0], nargs='+', type=float)
    parser.add_argument('--dropout_p_all', default=[0.4, 0.5, 0.8], nargs='+', type=float)

    args = parser.parse_args() 
    args.save_folder = pathlib.Path("ckpt")
    args.save_folder.mkdir(parents=True, exist_ok=True)
    train_dataset, val_dataset, val_dataset_all = get_datasets_mnms_single_vendor(data_root=args.data_root, vendor = args.train_vendor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
        pin_memory=False, num_workers=2, collate_fn=determinist_collate)

    print('train sample:',len(train_dataset), 'train minibatch:',len(train_loader),\
          'valid sample:',len(val_dataset), 'valid minibatch:',len(val_loader))

    if args.use_base:
        model = unet.Unet(inplanes=args.in_channels, num_classes=args.num_classes).cuda()
    elif args.use_cbs or args.use_fd:
        model = unet.Unet_FD(inplanes=args.in_channels, num_classes=args.num_classes, args=args).cuda()    

    model = torch.nn.DataParallel(model)
    criterion_dice = EDiceLoss().cuda()
    criterion = torch.nn.CrossEntropyLoss().cuda()
    best_ckpt_name = 'model_best_cbs_{}_fd_{}_v_{}'.format(int(args.use_cbs), int(args.use_fd), args.train_vendor)
    
    print('ckpt name:', best_ckpt_name)
    best_ckpt_dir = os.path.join(str(args.save_folder), best_ckpt_name)
    metric = criterion_dice.metric_mnms
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr, weight_decay=args.weight_decay, eps=1e-4)
    best_epoch, best_dice, best_dices = 0, 0, [0,0,0]

    for epoch in range(args.epochs): 
        if args.use_cbs: 
            model.module.get_new_kernels_cbs(epoch)
            model.cuda()
        elif args.use_fd:
            model.module.get_new_kernels()
            model.cuda()

        step_train(train_loader, model, criterion, metric, optimizer)
        with torch.no_grad():
            validation_loss, dice_metrics = step_valid(val_loader, model, criterion, metric)
            dice_metrics = list(zip(*dice_metrics))
            dice_metrics = [torch.tensor(dice, device="cpu").numpy() for dice in dice_metrics]
            avg_dices = np.mean(dice_metrics,1)
            mean_avg_dice = np.mean(avg_dices)
        if mean_avg_dice > best_dice:
            best_dice = mean_avg_dice
            best_epoch = epoch
            best_dices = avg_dices
            torch.save(dict(state_dict=model.state_dict()),best_ckpt_dir )

        print('epoch:%d/%d, loss:%.4f, cur avg dice:%.4f, best epoch:%d, best dice:%.4f[LV:%.4f, RV:%.4f, MYO:%.4f]'
                        %(epoch, args.epochs, validation_loss, mean_avg_dice, best_epoch, best_dice, best_dices[0],
                        best_dices[1], best_dices[2]))

if __name__ == '__main__':
    seed_everything()
    main()
