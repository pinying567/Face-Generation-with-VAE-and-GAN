import torch
import numpy as np

def cvt_image(img):
    """Convert tensor image (3xHxW) to numpy image (uint8)"""
    img = (img.numpy() * 0.5 + 0.5) * 255                # convert data range from [-1, 1] to [0, 255]
    img = np.transpose(img.astype(np.uint8), (1, 2, 0))  # 3xHxW -> HxWx3         
    return img

def lr_decay(optimizer, decay_rate=.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate

def get_lambda(epoch, max_epoch):
    p = epoch / max_epoch
    return 2.0 / (1 + np.exp(-10.0 * p)) - 1.0

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(1.0 / batch_size))
    return res

class averageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count