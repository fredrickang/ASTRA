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
               beta=0,
               gamma=0,
               delta=0,
               sf_type=0,
               lat_const=-1,
               loss_type=0,
               seperate = 0,
               dim_feature=1984):
    super(FBNet, self).__init__()
    init_func = lambda x: nn.init.constant_(x, init_theta)
    
    self._alpha = alpha
    self._beta = beta
    self._criterion = nn.CrossEntropyLoss().cuda()

    self.theta = []
    self._ops = nn.ModuleList()
    self._blocks = blocks
    
    self.softmax_type = sf_type
    self.lat_constr = torch.Tensor([lat_const]).cuda()
    self.loss_type = loss_type
    self.seperate = seperate
    
    self.encodings = []
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
        encoding = onehot(len(theta)).cuda().eval()
        self.encodings.append(encoding)
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
  def forward(self, input, target, temperature=5.0, theta_list=None): 
    
    batch_size = input.size()[0]
    self.batch_size = batch_size
    
    data = self._input_conv(input)
  
    theta_idx = 0
    lat = []
    onehot_lat = []
    for l_idx in range(self._input_conv_count, len(self._blocks)):
      block = self._blocks[l_idx]
      if isinstance(block, list):
        blk_len = len(block)
        if theta_list is None:
          theta = self.theta[theta_idx]
        else:
          theta = theta_list[theta_idx]
        t = theta.repeat(batch_size, 1)
        
        if self.loss_type == 1:
          weight = nn.functional.softmax(t, dim = 1)
          lat_weight = nn.functional.softmax(theta, dim=-1).repeat(batch_size, 1)
        
        if self.loss_type == 2:
          weight = nn.functional.softmax(t* 100000, dim = 1) 
          lat_weight = weight
        
        if self.loss_type == 3:
          weight = nn.functional.softmax(t, dim = 1)
          lat_weight = nn.functional.softmax(t*100000, dim = 1) 
          
        onehot_weight = nn.functional.one_hot(torch.argmax(t, dim = 1), num_classes=len(t[0]))
        
        speed = self._speed[theta_idx][:blk_len].to(weight.device)

        lat_ = lat_weight * speed.repeat(batch_size, 1)
        onehot_lat_ = onehot_weight * speed.repeat(batch_size, 1)

        lat.append(torch.sum(lat_))
        onehot_lat.append(torch.sum(onehot_lat_))

        data = self._ops[theta_idx](data, weight)
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

    if self.lat_loss <= self.lat_constr:
      self.loss = self.ce
    else:
      self.loss = self.ce + (self.lat_loss.pow(self._alpha) / self.lat_constr.pow(self._alpha))
        
    pred = torch.argmax(logits, dim=1)
    self.acc = torch.sum(pred == target).float() / batch_size
    return self.loss, self.ce, self.acc, self.lat_loss, self.onehot_lat_loss


class ChildNet(nn.Module):
    def __init__(self, theta_f='_theta_epoch_91.txt'):
        super(ChildNet, self).__init__()
        
        with open(theta_f) as f:
            block_nos = []
            for layer,line in enumerate(f):
                l = []
                for value in line.split(' '):
                    l.append(value)
                block_nos.append(l.index(max(l)))
        #print('Layer: {} Block: {}'.format(layer+1,block+1))
        
        self.arch = []
        block_list = []
        
        expansion = [1, 1, 3, 6, 1, 1, 3, 6]
        kernel = [3, 3, 3, 3, 5, 5, 5, 5]
        group = [1, 2, 1, 1, 1, 2, 1, 1]
        
        in_ch = [16, 16, 24, 32, 64, 112, 184]
        out_ch = [16, 24, 32, 64, 112, 184, 352]
        n = [1, 4, 4, 4, 4, 4, 1]
        stride = [1, 2, 2, 1, 1, 1, 1]
        
        block_list.append(nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1))
        
        in_ch = 16
        count = 0
        for i in range(len(n)):
            stride_p = stride[i]
            for j in range(n[i]):
                k = block_nos[count]
                if k==8:
                    block_list.append(IdentityBlock)
                else:
                    block_list.append(FBNetBlock(C_in=in_ch, C_out=out_ch[i], 
                                                 kernel_size=kernel[k], stride=stride_p, 
                                                 expansion=expansion[k], group=group[k]))
                in_ch = out_ch[i]
                stride_p = 1
                count = count + 1
        
        block_list.append(nn.Conv2d(in_channels=352, out_channels=1984, kernel_size=1, stride=1))
        
        #print(block_list)
        tmp = []
        for b in block_list:
            if isinstance(b, nn.Module):
                print("True")
                tmp.append(b)
            else:
                print("False")
            self.arch = nn.Sequential(*tmp)
        
        self.dropout = nn.Dropout(p=0.4, inplace=False)
        
        self.avgpool = nn.AvgPool2d(kernel_size=8, stride=1)
        
        self.fc = nn.Linear(in_features=1984, out_features=10)
        
    def forward(self, x):
        
        x = self.arch(x)
        
        x = self.dropout(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        
        return x
      
class IdentityBlock(nn.Module):
    def __init__(self):
        super(IdentityBlock, self).__init__()
    def forward(self, x):
        return x      
      
class FBNetBlock(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride,
              expansion, group):
        super(FBNetBlock, self).__init__()
        if kernel_size == 3:
            padding = 1
        elif kernel_size == 5:
            padding = 2
        else:
            raise ValueError("Not supported kernel_size %d" % kernel_size)
        bias_flag = True
        if group == 1:
            self.op = nn.Sequential(
            nn.Conv2d(C_in, C_in*expansion, 1, stride=1, padding=0,
                  groups=group, bias=bias_flag),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in*expansion, C_in*expansion, kernel_size, stride=stride, 
                  padding=padding, groups=C_in*expansion, bias=bias_flag),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in*expansion, C_out, 1, stride=1, padding=0, 
                  groups=group, bias=bias_flag)
            )
        else:
            self.op = nn.Sequential(
            nn.Conv2d(C_in, C_in*expansion, 1, stride=1, padding=0,
                  groups=group, bias=bias_flag),
            nn.ReLU(inplace=False),
            ChannelShuffle(group),
            nn.Conv2d(C_in*expansion, C_in*expansion, kernel_size, stride=stride, 
                  padding=padding, groups=C_in*expansion, bias=bias_flag),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in*expansion, C_out, 1, stride=1, padding=0, 
                  groups=group, bias=bias_flag),
            ChannelShuffle(group)
            )
        res_flag = ((C_in == C_out) and (stride == 1))
        self.res_flag = res_flag

    def forward(self, x):
        if self.res_flag:
            return self.op(x) + x
        else:
            return self.op(x) # + self.trans(x)
          
def softargmax1d(input, beta=100):
    *_, n = input.shape
    input = nn.functional.softmax(beta * input, dim=-1)
    indices = torch.linspace(0, 1, n).to(input.device)
    result = torch.sum((n - 1) * input * indices, dim=-1)
    
    return result
  
class onehot(nn.Module):
    def __init__(self, input_dim):
        super(onehot, self).__init__()
        self.input_dim = input_dim
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, int(self.input_dim/2)),
            nn.ReLU(inplace=False),
            nn.Linear(int(self.input_dim/2), int(self.input_dim/2)),
            nn.ReLU(inplace=False),
            nn.Linear(int(self.input_dim/2), self.input_dim),
        )
        for m in self.modules():
          if isinstance(m, nn.Linear):
            torch.nn.init.uniform_(m.weight.data)
            
    def set_input_dim(self, input_dim):
      self.input_dim = input_dim
  
    def forward(self, x):
      return self.layers(x)