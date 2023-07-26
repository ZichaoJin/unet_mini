import torch.nn.functional as F
from loss import dice_loss
import numpy as np
import glob

# get all images from a file
def get_from_path(path):
    img_list = []
    files = glob.glob(path + "/*")
    for file in files:
        img_list.append(np.load(file,allow_pickle=True))
    return img_list

# calculate loss
def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss

def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))

# save train loss and val loss into lists for later plot
def save_loss(metrics, epoch_samples, phase, train_loss_list, val_loss_list):
    for k in metrics.keys():
        if k == "loss":
            outputs = metrics[k] / epoch_samples
    if phase == "train":
        train_loss_list.append(outputs)
        return train_loss_list
    else:
        val_loss_list.append(outputs)
        return val_loss_list

def reverse_transform(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    inp = (inp * 255).astype(np.uint8)
    
    return inp