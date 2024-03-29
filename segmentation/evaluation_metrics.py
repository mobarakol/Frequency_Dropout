import torch
import torch.nn as nn

class EDiceLoss(nn.Module):
    """Dice loss tailored to Brats need.
    """

    def __init__(self, do_sigmoid=True):
        super(EDiceLoss, self).__init__()
        self.do_sigmoid = do_sigmoid
        self.labels = ["ET", "TC", "WT"]
        self.device = "cpu"

    def binary_dice(self, inputs, targets, metric_mode=False):
        #targets = torch.tensor(targets)
        smooth = 1.
        if metric_mode:
            #print(targets.unique())
            if targets.sum() == 0:
                if inputs.sum() == 0:
                    return torch.tensor(1., device="cuda")
                else:
                    return torch.tensor(0., device="cuda")
        # Threshold the pred
        intersection = EDiceLoss.compute_intersection(inputs, targets)
        if metric_mode:
            dice = (2 * intersection) / ((inputs.sum() + targets.sum()) * 1.0)
        else:
            dice = (2 * intersection + smooth) / (inputs.pow(2).sum() + targets.pow(2).sum() + smooth)
        if metric_mode:
            return dice
        return 1 - dice

    @staticmethod
    def compute_intersection(inputs, targets):
        intersection = torch.sum(inputs * targets)
        return intersection

    def forward(self, inputs, target):
        dice = 0
        for i in range(target.size(1)):
            dice = dice + self.binary_dice(inputs[:, i, ...], target[:, i, ...], i)

        final_dice = dice / target.size(1)
        return final_dice

    def metric_brats(self, inputs, target):
        dices = []
        for j in range(target.size(0)):
            dice = []           
            dice.append(self.binary_dice(inputs[j]==3, target[j]==3, True))
            dice.append(self.binary_dice(torch.logical_or(inputs[j]==1, inputs[j]==3), torch.logical_or(target[j]==1, target[j]==3), True))
            dice.append(self.binary_dice(inputs[j]>0, target[j]>0, True))
            dices.append(dice)
            
        return dices

    def metric_mnms(self, inputs, target):
        dices = []
        for j in range(target.size(0)):
            dice = []
            for i in range(1, 4):           
                dice.append(self.binary_dice(inputs[j]==i, target[j]==i, True))
            dices.append(dice)
            
        return dices