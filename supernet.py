import torch
import torch.nn as nn
import numpy as np

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
               energy_f='./energy.txt',
               alpha=0,
               beta=0,
               gamma=0,
               delta=0,
               sf_type=0,
               lat_const=-1,
               loss_type=0,
               dim_feature=1984):
    super(FBNet, self).__init__()
    init_func = lambda x: nn.init.constant_(x, init_theta)
    
    self._alpha = alpha
    self._beta = beta
    self._gamma = gamma
    self._delta = delta
    self._criterion = nn.CrossEntropyLoss().cuda()

    self.theta = []
    self._ops = nn.ModuleList()
    self._blocks = blocks
    
    self.softmax_type = sf_type
    self.lat_constr = torch.Tensor([lat_const]).cuda()
    self.loss_type = loss_type
    
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
        init_func(theta)
        self.theta.append(theta)
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

  ###########################################33
    energy_f = energy_f

    with open(energy_f, 'r') as f:
      _energy = f.readlines()

    self._energy = [[float (t) for t in s.strip().split(' ')] for s in _energy]
#############################################
    # TODO
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
###################################################3
    max_len = max([len(s) for s in self._energy])
    iden_s = 0.0
    iden_s_c = 0
    for s in self._energy:
      if len(s) == max_len:
        iden_s += s[max_len - 1]
        iden_s_c += 1
    iden_s /= iden_s_c
    for i in range(len(self._energy)):
      if len(self._energy[i]) == (max_len - 1):
        self._energy[i].append(iden_s)
#######################################################
    self._speed = torch.tensor(self._speed, requires_grad=False)

#######################################################
    self._energy = torch.tensor(self._energy, requires_grad=False)
#######################################################
    self.classifier = nn.Linear(dim_feature, num_classes)
    # TODO
    # nn.Sequential(nn.BatchNorm2d(dim_feature)
    # nn.Linear(dim_feature, num_classes))

  # softmax_type: 0 = gumbel, 1 = softmax
  def forward(self, input, target, temperature=5.0, theta_list=None): 
    batch_size = input.size()[0]
    self.batch_size = batch_size
    data = self._input_conv(input)
    theta_idx = 0
    lat = []
    onehot_lat = []
    ener = []
    for l_idx in range(self._input_conv_count, len(self._blocks)):
      block = self._blocks[l_idx]
      if isinstance(block, list):
        blk_len = len(block)
        if theta_list is None:
          theta = self.theta[theta_idx]
        else:
          theta = theta_list[theta_idx]
        t = theta.repeat(batch_size, 1)
        if self.softmax_type == 0:
          weight = nn.functional.gumbel_softmax(t,
                                temperature)
          #onehot_weight = nn.functional.gumbel_softmax(t,temperature,True)
          onehot_weight = nn.functional.one_hot(torch.argmax(t, dim =1), num_classes=len(t[0]))
        else:
          weight = nn.functional.softmax(t, dim=1)
          #onehot_weight = nn.functional.gumbel_softmax(t,hard=True)
          onehot_weight = nn.functional.one_hot(torch.argmax(t, dim =1), num_classes=len(t[0]))
          
        speed = self._speed[theta_idx][:blk_len].to(weight.device)
        energy = self._energy[theta_idx][:blk_len].to(weight.device)
        lat_ = weight * speed.repeat(batch_size, 1)
        onehot_lat_ = onehot_weight * speed.repeat(batch_size, 1)
        ener_ = weight * energy.repeat(batch_size, 1)
        lat.append(torch.sum(lat_))
        onehot_lat.append(torch.sum(onehot_lat_))
        ener.append(torch.sum(ener_))
        data = self._ops[theta_idx](data, weight)
        theta_idx += 1
      else:
        break

    data = self._output_conv(data)
    lat = sum(lat)
    onehot_lat = sum(onehot_lat)
    ener = sum(ener)

    data = nn.functional.avg_pool2d(data, data.size()[2:])
    data = data.reshape((batch_size, -1))
    logits = self.classifier(data)

    self.ce = self._criterion(logits, target).sum()
    self.lat_loss = lat / batch_size
    self.onehot_lat_loss = onehot_lat / batch_size
    self.ener_loss = ener / batch_size
        
    # if self.lat_constr == -1:
    #   self.loss = self.ce +  self._alpha * self.lat_loss.pow(self._beta) + self._gamma * self.ener_loss.pow(self._delta)
    # else:
    #   if self.loss_type == 1:
    #     self.loss = self.ce  * ( self.lat_loss.pow(self._alpha) / self.lat_constr.pow(self._alpha) ) 
    #   if self.loss_type == 2: # weighted sum
    #     self.loss = self.ce * 0.6 + (self.lat_loss.pow(self._alpha) /  self.lat_constr.pow(self._alpha)) * 0.4
    #   if self.loss_type == 3: # pareto-optimal
    #     if self.lat_loss <= self.lat_constr:
    #       self.loss = self.ce
    #     else:
    #       self.loss = self.ce * ( self.lat_loss.pow(self._alpha) / self.lat_constr.pow(self._alpha) ) 
    
    self.loss = self.ce
    
    pred = torch.argmax(logits, dim=1)
    # succ = torch.sum(pred == target).cpu().numpy() * 1.0
    self.acc = torch.sum(pred == target).float() / batch_size
    return self.loss, self.ce, self.acc, self.lat_loss, self.ener_loss, self.onehot_lat_loss

