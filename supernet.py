import torch
import torch.nn as nn
import numpy as np
from candblks import ChannelShuffle

class MixedOp(nn.Module):
  """Mixed operation.
  Weighted sum of blocks.
  """
  def __init__(self, blocks):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for op in blocks:
      self._ops.append(op)

  def forward(self, x, weights):
    tmp = []
    for i, op in enumerate(self._ops):
      r = op(x)
      w = weights[..., i].reshape((-1, 1, 1, 1))
      res = w * r
      tmp.append(res)
    return sum(tmp)

class FBNet(nn.Module):

  def __init__(self, num_classes, blocks,
               init_theta=1.0,
               speed_f='./speed.txt',
               alpha=0,
               lat_const=-1,
               dim_feature=1984):
    super(FBNet, self).__init__()
    init_func = lambda x: nn.init.constant_(x, init_theta)
    
    self._alpha = alpha
    self._criterion = nn.CrossEntropyLoss().cuda()

    self.theta = []
    self.original_thteta = []
    self._ops = nn.ModuleList()
    self._blocks = blocks
    
    self.lat_constr = torch.Tensor([lat_const]).cuda()
    
    tmp = []
    input_conv_count = 0
    for b in blocks:
      if isinstance(b, nn.Module):
        tmp.append(b)
        input_conv_count += 1
      else:
        break
    self._input_conv = nn.Sequential(*tmp)
    self._input_conv_count = input_conv_count
    for b in blocks:
      if isinstance(b, list):
        num_block = len(b)
        theta = nn.Parameter(torch.ones((num_block, )).cuda(), requires_grad=True)
        origin_theta = nn.Parameter(torch.ones((num_block,)).cuda(), requires_grad=False)
        init_func(theta)
        init_func(origin_theta)
        self.theta.append(theta)
        self.original_thteta.append(origin_theta)
        
        self._ops.append(MixedOp(b))
        input_conv_count += 1
    tmp = []
    for b in blocks[input_conv_count:]:
      if isinstance(b, nn.Module):
        tmp.append(b)
        input_conv_count += 1
      else:
        break
    self._output_conv = nn.Sequential(*tmp)

    # assert len(self.theta) == 22
    with open(speed_f, 'r') as f:
      _speed = f.readlines()
    self._speed = [[float(t) for t in s.strip().split(' ')] for s in _speed]

    max_len = max([len(s) for s in self._speed])
    iden_s = 0.0
    iden_s_c = 0
    for s in self._speed:
      if len(s) == max_len:
        iden_s += s[max_len - 1]
        iden_s_c += 1
    iden_s /= iden_s_c
    for i in range(len(self._speed)):
      if len(self._speed[i]) == (max_len - 1):
        self._speed[i].append(iden_s)

    self._speed = torch.tensor(self._speed, requires_grad=False)

    self.classifier = nn.Linear(dim_feature, num_classes)

  # softmax_type: 0 = gumbel, 1 = softmax
  def forward(self, input, target, amplifier=1.0, separation=0.0, theta_list=None): 
      
    batch_size = input.size()[0]
    self.batch_size = batch_size
    
    data = self._input_conv(input)
  
    theta_idx = 0
    lat = []
    onehot_lat = []
    diff = []
    pdist = torch.nn.PairwiseDistance(2)
    
    for l_idx in range(self._input_conv_count, len(self._blocks)):
      block = self._blocks[l_idx]
      if isinstance(block, list):
        blk_len = len(block)
        if theta_list is None:
          theta = self.theta[theta_idx]
        else:
          theta = theta_list[theta_idx]
        t = theta.repeat(batch_size, 1)
        
        weight = nn.functional.softmax(t, dim = 1)
        lat_weight = nn.functional.softmax(t*amplifier, dim = 1) 
        onehot_weight = nn.functional.one_hot(torch.argmax(t, dim = 1), num_classes=len(t[0]))
        
        speed = self._speed[theta_idx][:blk_len].to(weight.device)

        lat_ = lat_weight * speed.repeat(batch_size, 1)
        #lat_ = weight * speed.repeat(batch_size, 1)
        
        onehot_lat_ = onehot_weight * speed.repeat(batch_size, 1)

        lat.append(torch.sum(lat_))
        onehot_lat.append(torch.sum(onehot_lat_))

        data = self._ops[theta_idx](data, weight)
      
        softmax_theta = nn.functional.softmax(theta, dim = -1)
        softmax_origin_theta = nn.functional.softmax(self.original_thteta[theta_idx], dim = -1)
        diff.append(pdist(softmax_theta, softmax_origin_theta))
        
        theta_idx += 1
      else:
        break                

    data = self._output_conv(data)
    lat = sum(lat)
    onehot_lat = sum(onehot_lat)


    data = nn.functional.avg_pool2d(data, data.size()[2:])
    data = data.reshape((batch_size, -1))
    logits = self.classifier(data)

    self.ce = self._criterion(logits, target).sum()
    self.lat_loss = lat / batch_size
    self.onehot_lat_loss = onehot_lat / batch_size
  
    diff = sum(diff)

    if self.onehot_lat_loss <= self.lat_constr:
      self.loss = self.ce
    else:
      self.loss = self.ce + (self.lat_loss.pow(self._alpha) / self.lat_constr.pow(self._alpha)) - separation * diff

    pred = torch.argmax(logits, dim=1)
    self.acc = torch.sum(pred == target).float() / batch_size
    return self.loss, self.ce, self.acc, self.lat_loss, self.onehot_lat_loss