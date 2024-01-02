import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from supernet import FBNet
from candblks import get_blocks
from utils import weights_init, CosineDecayLR, AvgrageMeter

import os
import logging
import time
import numpy as np


from torch.utils.tensorboard import SummaryWriter

class Config(object):
    num_cls_used = 10
    init_theta = 1.0
    alpha = 0.2
    beta = 0.6
    w_lr = 0.1
    w_mom = 0.9
    w_wd = 1e-4
    t_lr = 0.01
    t_wd = 5e-4
    t_beta = (0.9, 0.999)
    init_temperature = 5.0
    temperature_decay = 0.956
    total_epoch = 90
    start_w_epoch = 1
    train_portion = 0.8
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
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        gpu_id: int,
        save_every: int,
        lr_scheduler : {},
        writer: SummaryWriter, 
        logging: str
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.apply(weights_init)
        self.model = self.model.to(gpu_id)
        self.train_data = train_data
        self.save_every = save_every
        self.model = DDP(model, device_ids=[gpu_id])
        
        self.params = self.model.module.parameters()
        self.theta = self.model.module.theta    
        self._temp_decay = 0.965
        self.temp = 5.0
        
        self.tensorboard = writer
        self.acc_avg = AvgrageMeter('acc')
        self.ce_avg = AvgrageMeter('ce')
        self.lat_avg = AvgrageMeter('lat')
        self.loss_avg = AvgrageMeter('loss')
        self.ener_avg = AvgrageMeter('ener')
        
        self.w_optimizer = torch.optim.SGD(
            self.params,
            0.1,
            momentum=0.9,
            weight_decay=1e-4
        )
        
        self.w_scheduler = CosineDecayLR(self.w_optimizer, **lr_scheduler)
        
        self.t_optimizer = torch.optim.Adam(
            self.theta,
            lr=0.01, betas = (0.9, 0.999),
            weight_decay=5e-4
        )
        
        self.logging_prefix = logging
        
    def train_w(self, input, target):
        self.w_optimizer.zero_grad()
        loss, ce, acc, lat, ener = self.model(input, target, self.temp)
        loss.backward()
        self.w_optimizer.step()
        
        return loss.item(), ce.item(), acc.item(), lat.item(), ener.item()

    def train_t(self, input, target):
        self.t_optimizer.zero_grad()
        loss, ce, acc, lat, ener =  self.model(input, target, self.temp)
        loss.backward()
        self.t_optimizer.step()
        
        return loss.item(), ce.item(), acc.item(), lat.item(), ener.item()
    
    def decay_temperature(self, decay_ratio=None):
        formal_temp = self.temp
        if decay_ratio is None:
            self.temp *= self._temp_decay
        else:
            self.temp *= decay_ratio
        if self.gpu_id == 0:
            print("Change temperature from %.5f to %.5f" % (formal_temp, self.temp))
        
        
    def step(self, input, target, epoch, step, log_freq, func):
        input = input.cuda()
        target = target.cuda()
        
        loss, ce, acc, lat, ener = func(input, target)
        
        self.loss_avg.update(loss)
        self.ce_avg.update(ce)
        self.acc_avg.update(acc)
        self.lat_avg.update(lat)
        self.ener_avg.update(ener)

        if step > 1 and (step % log_freq == 0) and self.gpu_id == 0:
            self.toc = time.time()
            
            batch_size = self.model.module.batch_size
            speed = 1.0 * (batch_size * torch.cuda.device_count() * log_freq) / (self.toc - self.tic)
            print("Epoch[%.3d] Batch[%.3d] Speed: %.6f samples/sec LR %.5f %s %s %s %s %s" 
              % (epoch, step, speed, self.w_scheduler.optimizer.param_groups[0]['lr'], self.loss_avg, 
                 self.acc_avg, self.ce_avg, self.lat_avg,self.ener_avg))
            
            self.tensorboard.add_scalar('Total Loss', self.loss_avg.getValue(), (epoch+1)*step)
            self.tensorboard.add_scalar('Accuracy',self.acc_avg.getValue(),(epoch+1)*step)
            self.tensorboard.add_scalar('Latency',self.lat_avg.getValue(),(epoch+1)*step)
            self.tensorboard.add_scalar('Energy',self.ener_avg.getValue(),(epoch+1)*step)
            
            map(lambda avg: avg.reset(), [self.loss_avg, 
                 self.acc_avg, self.ce_avg, self.lat_avg,self.ener_avg])
            self.tic = time.time()
        
    def search(self, train_w_ds, 
               train_t_ds,
               total_epoch,
               log_freq,
               warmup):
        
        # Warmup
        self.tic = time.time()
        for epoch in range(warmup):
            for st, (input, target) in enumerate(train_w_ds):
                self.step(input, target, epoch, st, log_freq,
                          lambda x, y: self.train_w(x, y))        
                self.w_scheduler.step()
                self.tensorboard.add_scalar('Learning rate curve',self.w_scheduler.last_epoch, self.w_optimizer.param_groups[0]['lr'])

        self.tic = time.time()
        for epoch in range(total_epoch):
            for st, (input, target) in enumerate(train_t_ds):
                self.step(input, target, epoch + warmup, st, log_freq,
                          lambda x, y: self.train_t(x, y))
            if self.gpu_id == 0:
                self.save_theta(save_path='./theta_result/{}'.format(self.logging_prefix), 
                            file_name='theta_epoch_%d.txt'.format(epoch + warmup), epoch=epoch)
            self.decay_temperature()
            
            for st, (input, target) in enumerate(train_w_ds):
                self.step(input, target, epoch + warmup, st, log_freq,
                          lambda x, y: self.train_w(x, y))
                self.w_scheduler.step()
    
    def save_theta(self, save_path='./theta_result', file_name = 'theta.txt',epoch=0):
        res = []
        try:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
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
    
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, 
                download=True, transform=train_transform)

    blocks = get_blocks(cifar10=True)
    
    model = FBNet(num_classes= config.num_cls_used,
              blocks=blocks,
              init_theta=config.init_theta,
              alpha=config.alpha,
              beta=config.beta,
              gamma=0,
              delta=0,
              speed_f="speed.txt",
	      energy_f="energy.txt")
    return train_set, model

def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )

def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int, config: Config, exp_name: str):
    ddp_setup(rank, world_size)
    dataset, model = load_train_objs(config)
    train_data = prepare_dataloader(dataset, batch_size)
    writer = SummaryWriter("./runs/{}".format(exp_name))
    trainer = Trainer(model, train_data, rank, save_every, config.lr_scheduler_params, writer, exp_name)
    trainer.search(train_data, train_data,
                   total_epochs, save_every, 2)
    destroy_process_group()


if __name__ == "__main__":
    np.random.seed(0)

    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--total_epochs', default=100, type=int,  help='Total epochs to train the model')
    parser.add_argument('--save_every', default=5, type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=1024, type=int, help='Input batch size on each device (default: 1024)')
    parser.add_argument('--save', default="EXP", help="Experiment name")
    parser.add_argument('--gpus', default=1, type=int, help="Number of GPUs")
    
    args = parser.parse_args()
    args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
    

    config = Config()

    world_size = args.gpus
    mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size, config, args.save), nprocs=world_size)
    
    
    