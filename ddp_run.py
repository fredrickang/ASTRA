import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.backends.cudnn as cudnn

from supernet import FBNet
from candblks import get_blocks
from utils import weights_init, CosineDecayLR, AverageMeter

import os
import logging
import time
import numpy as np
import random


from torch.utils.tensorboard import SummaryWriter

torch.backends.cudnn.deterministic = True

class Config(object):
    num_cls_used = 10
    init_theta = 1.0
    w_lr = 0.1 
    w_mom = 0.9
    w_wd = 1e-4
    t_lr = 0.01 
    t_wd = 5e-4
    t_beta = (0.9, 0.999)
    total_epoch = 90
    start_w_epoch = 1
    train_portion = 0.8
    lat_constr = -1
    amplifier = 1.0
    amplifying = 1.0
    separation = 0.0
    sep_temp = 1.0
    alpha = 1.0
    lr_scheduler_params = {
    'T_max' : 400,
    'alpha' : 1e-4,
    'warmup_step' : 100,
    't_mul' : 1.5,
    'lr_mul' : 0.98,
    }
    
def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "0.0.0.0"
    os.environ["MASTER_PORT"] = "1234"
    tmp_rank = rank
    if rank >= world_size:
        tmp_rank = rank - world_size
        os.environ["MASTER_PORT"] = "1235"
    

    init_process_group(backend="nccl", rank=tmp_rank, world_size=world_size)
    torch.cuda.set_device(rank)

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        val_data: DataLoader,
        gpu_id: int,
        save_every: int,
        lr_scheduler : {},
        writer: SummaryWriter, 
        logging: str,
        config: Config
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.apply(weights_init)
        self.model = self.model.to(gpu_id)
        self.train_data = train_data
        self.val_data = val_data
        self.save_every = save_every
        self.model = DDP(model, device_ids=[gpu_id])
        
        self.params = self.model.module.parameters()
        self.theta = self.model.module.theta    
        self.config = config
        self.amplifier = config.amplifier
        self.separation = config.separation
    
        self.tensorboard = writer
        self.acc_avg = AverageMeter('acc')
        self.ce_avg = AverageMeter('ce')
        self.lat_avg = AverageMeter('lat')
        self.loss_avg = AverageMeter('loss')
        self.onehot_lat_avg = AverageMeter('onehot_lat')
        
        self.step_counter = 0
    
        self.w_optimizer = torch.optim.SGD(
            self.params,
            config.w_lr,
            momentum=config.w_mom,
            weight_decay=config.w_wd
        )
        
        self.w_scheduler = CosineDecayLR(self.w_optimizer, **lr_scheduler)
        
        self.t_optimizer = torch.optim.Adam(
            self.theta,
            lr=config.t_lr, betas = config.t_beta,
            weight_decay=config.t_wd
        )
        
        self.logging_prefix = logging
        
    def train_w(self, input, target):
        self.w_optimizer.zero_grad()
        loss, ce, acc, lat, onehot_lat = self.model(input, target, self.amplifier, self.separation)
        loss.backward()
        self.w_optimizer.step()
        
        return loss.item(), ce.item(), acc.item(), lat.item(), onehot_lat.item()

    def train_t(self, input, target):
        self.t_optimizer.zero_grad()
        loss, ce, acc, lat, onehot_lat =  self.model(input, target, self.amplifier, self.separation)
        loss.backward()
        self.t_optimizer.step()
        
        return loss.item(), ce.item(), acc.item(), lat.item(), onehot_lat.item()
    
    def amplify_amplifier(self, ):
        formal_amplifier = self.amplifier
        if self.amplifier <= 100:
            self.amplifier *= self.config.amplifying
        
        if self.gpu_id == 0:
            print("Change amplifier from %.5f to %.5f" % (formal_amplifier, self.amplifier))
    
    def tempering_separation(self, ):
        formal_separation = self.separation
        if self.separation <= 1.0:
            self.separation *= self.config.sep_temp
        
        if self.gpu_id == 0:
            print("Change Separation from %.5f to %.5f" % (formal_separation, self.separation))       
    
    # def decay_temperature(self, decay_ratio=None):
    #     formal_temp = self.temp
    #     if decay_ratio is None:
    #         if self.temp <= 100:
    #             self.temp *= self._temp_decay
    #     else:
    #         self.temp *= decay_ratio
    #     if self.gpu_id == 0:
    #         print("Change temperature from %.5f to %.5f" % (formal_temp, self.temp))
        
    #     self.tensorboard.add_scalar('Temperature', self.temp, self.step_counter)            
    
    # def temper_separation(self, temper_ratio=None):
    #     formal_separation = self.model.module._beta
    #     if temper_ratio is None:
    #         if self.model.module._beta <= 3.0:
    #             self.model.module._beta *= 1.046
    #     if self.gpu_id == 0:
    #         print("Change separation intensity from %.5f to %.5f" % (formal_separation, self.model.module._beta))
        
    #     self.tensorboard.add_scalar('Separation', self.model.module._beta, self.step_counter)
    
    def step(self, input, target, epoch, step, log_freq, func):
        input = input.cuda()
        target = target.cuda()
        
        loss, ce, acc, lat, onehot_lat = func(input, target)
        
        self.loss_avg.update(loss, input.size(0))
        self.ce_avg.update(ce, input.size(0))
        self.acc_avg.update(acc, input.size(0))
        self.lat_avg.update(lat, input.size(0))
        self.onehot_lat_avg.update(onehot_lat, input.size(0))
        
        self.tensorboard.add_scalar('Accuracy',self.acc_avg.val, self.step_counter)
        self.tensorboard.add_scalar('Total Loss', self.loss_avg.val, self.step_counter)            
        
        self.tensorboard.add_scalars('Latency', {
            'estimated': self.lat_avg.val,
            'actual': self.onehot_lat_avg.val,
            }, self.step_counter)

        self.step_counter += 1
        
        if step > 1 and (step % log_freq == 0) and (self.gpu_id == 0 or self.gpu_id == 4):
            self.toc = time.time()
            
            batch_size = self.model.module.batch_size
            speed = 1.0 * (batch_size * torch.cuda.device_count() * log_freq) / (self.toc - self.tic)
            print("Epoch[%.3d] Batch[%.3d] Speed: %.6f samples/sec LR %.5f %s %s %s %s" 
              % (epoch, step, speed, self.w_scheduler.optimizer.param_groups[0]['lr'], self.loss_avg, 
                 self.acc_avg, self.ce_avg, self.lat_avg))
            
            self.tic = time.time()
        
    def search(self, train_ds,
               val_ds,
               total_epoch,
               log_freq,
               warmup):
        
        # Warmup
        self.tic = time.time()
        for epoch in range(warmup):
            for st, (input, target) in enumerate(train_ds):
                self.step(input, target, epoch, st, log_freq,
                          lambda x, y: self.train_w(x, y))        
                self.w_scheduler.step()
                self.tensorboard.add_scalar('Learning rate curve',self.w_scheduler.last_epoch, self.w_optimizer.param_groups[0]['lr'])

        self.tic = time.time()
        for epoch in range(total_epoch):
            for st, (input, target) in enumerate(train_ds):
                self.step(input, target, epoch + warmup, st, log_freq,
                          lambda x, y: self.train_t(x, y))
            if self.gpu_id == 0:
                self.save_theta(save_path='./theta_result/{}'.format(self.logging_prefix), 
                            file_name='theta_epoch_{}.txt'.format(epoch + warmup), epoch=epoch)
            
            self.amplify_amplifier()
            self.tempering_separation()
            
            for st, (input, target) in enumerate(train_ds):
                self.step(input, target, epoch + warmup, st, log_freq,
                          lambda x, y: self.train_w(x, y))
                self.w_scheduler.step()
                        
    def save_theta(self, save_path='./theta_result', file_name = 'theta.txt',epoch=0):
        res = []
        try:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
                print("{} Created".format(save_path))
        except:
            print("Error: Failed to create the theta_dir")
    
        with open(os.path.join(save_path, file_name), 'w') as f:
            for i,t in enumerate(self.theta):
                t_list = list(t.detach().cpu().numpy())
                if(len(t_list) < 9): t_list.append(0.00)
                max_index = t_list.index(max(t_list))
                self.tensorboard.add_scalar('Layer %s'% str(i),max_index+1, epoch)
                res.append(t_list)
                s = ' '.join([str(tmp) for tmp in t_list])
                f.write(s + '\n')
            val = np.array(res)
        return res
                
def load_train_objs(config:Config):
    
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
    
    
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                download=True, transform=train_transform)

    split = int(np.floor(1*len(dataset)))
    train_set, val_set = torch.utils.data.random_split(dataset, [split, len(dataset) - split])

    blocks = get_blocks(cifar10=True)
    
    model = FBNet(num_classes= config.num_cls_used,
              blocks=blocks,
              init_theta=config.init_theta,
              alpha=config.alpha,
              lat_const=config.lat_constr,
              speed_f="speed.txt")
    return train_set, val_set, model

def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )

def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int, config: Config, exp_name: str, offset: int):
    ddp_setup(rank+offset, world_size)
    train_set, val_set, model = load_train_objs(config)
    train_data = prepare_dataloader(train_set, batch_size)
    val_data = prepare_dataloader(val_set, batch_size)
    writer = SummaryWriter("./runs/{}".format(exp_name))
    trainer = Trainer(model, train_data, val_data, rank+offset, save_every, config.lr_scheduler_params, writer, exp_name, config)
    trainer.search(train_data, val_data,
                   total_epochs, save_every, 2)
    destroy_process_group()
    

if __name__ == "__main__":
    random.seed(2222)
    torch.manual_seed(2222)
    cudnn.deterministic = True
    cudnn.benchmark = False

    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--total_epochs', default=100, type=int,  help='Total epochs to train the model')
    parser.add_argument('--save_every', default=5, type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=1024, type=int, help='Input batch size on each device (default: 1024)')
    parser.add_argument('--save', default="EXP", help="Experiment name")
    parser.add_argument('--gpus', default=1, type=int, help="Number of GPUs")
    parser.add_argument('--amplifier', default=1, type=float, help="Softmax Amplifier")
    parser.add_argument('--amplifying', default=1, type=float, help="Softmax Amplifier amplfying param")
    parser.add_argument('--separation', default=0, type=float, help="Separation loss intensity")
    parser.add_argument('--sep_temp', default=1, type=float, help="Separation amplfying param")
    parser.add_argument('--lat_constr', default=-1, type=float, help="Latency constraint; -1: No constraint, other: < constr")
    parser.add_argument('--lat_penalty', default=1, type=float, help="latency penalty")
    parser.add_argument('--gpu_offset', default=0, type=int, help="ddp gpu offset")
    args = parser.parse_args()
    
    print(args)

    config = Config()    
    config.lat_constr = args.lat_constr
    config.amplifier = args.amplifier
    config.amplifying = args.amplifying
    config.separation = args.separation
    config.sep_temp = args.sep_temp
    config.alpha = args.lat_penalty
    
    world_size = args.gpus
    
    args.save = 'search-{}-{}'.format("lat{}_amplifier{}_amplifying{}_separation{}_septemp{}".format(
        config.lat_constr, config.amplifier, config.amplifying, config.separation, config.sep_temp), 
                                      time.strftime("%Y%m%d-%H%M%S"))

    mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size, config, args.save, args.gpu_offset), nprocs=world_size)
