import math
import torch
from torch.optim.lr_scheduler import _LRScheduler
import torch.distributed as dist
from enum import Enum

class CosineDecayLR(_LRScheduler):
  def __init__(self, optimizer, T_max, alpha=1e-4,
               t_mul=2, lr_mul=0.9,
               last_epoch=-1,
               warmup_step=300,
               logger=None):
    self.T_max = T_max
    self.alpha = alpha
    self.t_mul = t_mul
    self.lr_mul = lr_mul
    self.warmup_step = warmup_step
    self.logger = logger
    self.last_restart_step = 0
    self.flag = True
    super(CosineDecayLR, self).__init__(optimizer, last_epoch)

    self.min_lrs = [b_lr * alpha for b_lr in self.base_lrs]
    self.rise_lrs = [1.0 * (b - m) / self.warmup_step 
                     for (b, m) in zip(self.base_lrs, self.min_lrs)]

  def get_lr(self):
    T_cur = self.last_epoch - self.last_restart_step
    assert T_cur >= 0
    if T_cur <= self.warmup_step and (not self.flag):
      base_lrs = [min_lr + rise_lr * T_cur
              for (base_lr, min_lr, rise_lr) in 
                zip(self.base_lrs, self.min_lrs, self.rise_lrs)]
      if T_cur == self.warmup_step:
        self.last_restart_step = self.last_epoch
        self.flag = True
    else:
      base_lrs = [self.alpha + (base_lr - self.alpha) *
            (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
            for base_lr in self.base_lrs]
    if T_cur == self.T_max:
      self.last_restart_step = self.last_epoch
      self.min_lrs = [b_lr * self.alpha for b_lr in self.base_lrs]
      self.base_lrs = [b_lr * self.lr_mul for b_lr in self.base_lrs]
      self.rise_lrs = [1.0 * (b - m) / self.warmup_step 
                     for (b, m) in zip(self.base_lrs, self.min_lrs)]
      self.T_max = int(self.T_max * self.t_mul)
      self.flag = False
    
    return base_lrs

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
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

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def weights_init(m, deepth=0, max_depth=2):
  if deepth > max_depth:
    return
  if isinstance(m, torch.nn.Conv2d):
    torch.nn.init.kaiming_uniform_(m.weight.data)
    if m.bias is not None:
      torch.nn.init.constant_(m.bias.data, 0)
  elif isinstance(m, torch.nn.Linear):
    m.weight.data.normal_(0, 0.01)
    if m.bias is not None:
      m.bias.data.zero_()
  elif isinstance(m, torch.nn.BatchNorm2d):
    return
  elif isinstance(m, torch.nn.ReLU):
    return
  elif isinstance(m, torch.nn.Module):
    deepth += 1
    for m_ in m.modules():
      weights_init(m_, deepth)
  else:
    raise ValueError("%s is unk" % m.__class__.__name__)





