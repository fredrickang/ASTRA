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
    softmax_type = 0
    lat_constr = -1
    loss_type = 0
    total_lat_constr = 10
    p1 = 12.9
    p2 = 12.9
    p3 = 12.9
    rt_loss = 0
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
        name: str,
        model: torch.nn.Module,
        train_data: DataLoader,
        val_data: DataLoader,
        gpu_id: int,
        save_every: int,
        lr_scheduler : {},
        writer: SummaryWriter, 
        logging: str,
        config: Config, 
        rt_loss: int,
        period: float
    ) -> None:
        self.trainer_name = name
        self.gpu_id = gpu_id
        self.model = model.apply(weights_init)
        self.model = self.model.to(gpu_id)
        self.train_data = train_data
        self.val_data = val_data
        self.save_every = save_every
        self.model = DDP(model, device_ids=[gpu_id])
        
        self.params = self.model.module.parameters()
        self.theta = self.model.module.theta    
        self._temp_decay = config.temperature_decay
        self.temp = config.init_temperature
        
        self.tensorboard = writer
        self.acc_avg = AverageMeter('acc')
        self.ce_avg = AverageMeter('ce')
        self.lat_avg = AverageMeter('lat')
        self.loss_avg = AverageMeter('loss')
        self.ener_avg = AverageMeter('ener')
        self.onehot_lat_avg = AverageMeter('onehot_lat')
        self.integrated_lat_avg = AverageMeter('integrated_lat')
        
        self.rt_loss = rt_loss
        self.period = period
        
        self.tic = time.time()

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
        loss, ce, acc, lat, ener, onehot_lat = self.model(input, target, self.temp)
        loss.backward()
        self.w_optimizer.step()
        
        return loss.item(), ce.item(), acc.item(), lat.item(), ener.item(), onehot_lat.item()

    def train_w_multi(self, input, target, lat_losses, periods, multi_lat_constr):
        self.w_optimizer.zero_grad()
        loss, ce, acc, lat, ener, onehot_lat = self.model(input, target, self.temp)
        
        
        if self.rt_loss == 0:
            integrated_lat_loss = lat + sum(lat_losses)
            if integrated_lat_loss <= multi_lat_constr:
                    integrated_loss = ce
            else:
                integrated_loss = ce * (integrated_lat_loss.pow(self.model.module._alpha) / pow(multi_lat_constr, self.model.module._alpha))
        else:
            preemptive_util_sched = lat/self.period + lat_losses[0]/periods[0] + lat_losses[1]/periods[1] # need to automate
            integrated_lat_loss = preemptive_util_sched
            
            if preemptive_util_sched <= 1.0:
                integrated_loss = ce
            else:
                integrated_loss = ce + (preemptive_util_sched - 1.0) * self.model.module._alpha ## alpha should be bigger than 1
        
        ## parito-optimal loss      
        integrated_loss.backward()
        self.w_optimizer.step()
        
        return integrated_loss.item(), ce.item(), acc.item(), lat.item(), ener.item(), onehot_lat.item(), integrated_lat_loss.item()

    def train_t(self, input, target):
        self.t_optimizer.zero_grad()
        loss, ce, acc, lat, ener, onehot_lat =  self.model(input, target, self.temp)
        loss.backward()
        self.t_optimizer.step()
        
        return loss.item(), ce.item(), acc.item(), lat.item(), ener.item(), onehot_lat.item()

    def train_t_multi(self, input, target, lat_losses, periods, multi_lat_constr):
        self.t_optimizer.zero_grad()
        loss, ce, acc, lat, ener, onehot_lat = self.model(input, target, self.temp)
        
        if self.rt_loss == 0:
            integrated_lat_loss = lat + sum(lat_losses)
            if integrated_lat_loss <= multi_lat_constr:
                    integrated_loss = ce
            else:
                integrated_loss = ce * (integrated_lat_loss.pow(self.model.module._alpha) / pow(multi_lat_constr, self.model.module._alpha))
        else:
            preemptive_util_sched = lat/self.period + lat_losses[0]/periods[0] + lat_losses[1]/periods[1] # need to automate
            integrated_lat_loss = preemptive_util_sched
            
            if preemptive_util_sched <= 1.0:
                integrated_loss = ce
            else:
                integrated_loss = ce + (preemptive_util_sched - 1.0) * self.model.module._alpha ## alpha should be bigger than 1
            
        integrated_loss.backward()
        self.t_optimizer.step()
        
        return integrated_loss.item(), ce.item(), acc.item(), lat.item(), ener.item(), onehot_lat.item(), integrated_lat_loss.item()

    def validate(self, input, target):
        loss, ce, acc, lat, ener, onehot_lat = self.model(input, target, self.temp)
        return loss.item(), ce.item(), acc.item(), lat.item(), ener.item(), onehot_lat.item()

    
    def decay_temperature(self, decay_ratio=None):
        formal_temp = self.temp
        if self.temp >= 1E-7:
            if decay_ratio is None:
                self.temp *= self._temp_decay
            else:
                self.temp *= decay_ratio

        if self.gpu_id == 0:
            print("Change temperature from %.5f to %.5f" % (formal_temp, self.temp))
        
        self.tensorboard.add_scalar('Temperature', self.temp, self.step_counter)            
        
    def step(self, input, target, epoch, step, log_freq, func):
        input = input.cuda()
        target = target.cuda()
        
        loss, ce, acc, lat, ener, onehot_lat = func(input, target)
        
        self.loss_avg.update(loss, input.size(0))
        self.ce_avg.update(ce, input.size(0))
        self.acc_avg.update(acc, input.size(0))
        self.lat_avg.update(lat, input.size(0))
        self.ener_avg.update(ener, input.size(0))
        self.onehot_lat_avg.update(onehot_lat, input.size(0))
        
        self.tensorboard.add_scalar('{}/Accuracy'.format(self.trainer_name),self.acc_avg.val, self.step_counter)
        self.tensorboard.add_scalar('{}/Total Loss'.format(self.trainer_name), self.loss_avg.val, self.step_counter)               
        
        self.tensorboard.add_scalars('{}/Latency'.format(self.trainer_name), {
            'estimated': self.lat_avg.val,
            'actual': self.onehot_lat_avg.val,
            }, self.step_counter)
        #self.tensorboard.add_scalar('Latency/estimated',self.lat_avg.val,self.step_counter)
        #self.tensorboard.add_scalar('Latency/actual', self.onehot_lat_avg.val, self.step_counter)
        
        self.step_counter += 1
        
        if step > 1 and (step % log_freq == 0) and self.gpu_id == 0:
            self.toc = time.time()
            
            batch_size = self.model.module.batch_size
            speed = 1.0 * (batch_size * torch.cuda.device_count() * log_freq) / (self.toc - self.tic)
            print("Epoch[%.3d] Batch[%.3d] Speed: %.6f samples/sec LR %.5f %s %s %s %s %s" 
              % (epoch, step, speed, self.w_scheduler.optimizer.param_groups[0]['lr'], self.loss_avg, 
                 self.acc_avg, self.ce_avg, self.lat_avg,self.ener_avg))
            
            self.tic = time.time()

    def val_step(self, input, target, epoch, step, log_freq, func):
        input = input.cuda()
        target = target.cuda()
        
        loss, ce, acc, lat, ener, onehot_lat = func(input, target)
        
        # self.loss_avg.update(loss, input.size(0))
        # self.ce_avg.update(ce, input.size(0))
        # self.acc_avg.update(acc, input.size(0))
        # self.lat_avg.update(lat, input.size(0))
        # self.ener_avg.update(ener, input.size(0))
        # self.onehot_lat_avg.update(onehot_lat, input.size(0))
                
        self.tensorboard.add_scalar('{}/Val_Accuracy'.format(self.trainer_name), acc/input.size(0) , self.step_counter)
        # self.tensorboard.add_scalar('{}/Val_Total Loss'.format(self.trainer_name), self.loss_avg.val, self.step_counter)               
        
        # self.tensorboard.add_scalars('{}/Val_Latency'.format(self.trainer_name), {
        #     'estimated': self.lat_avg.val,
        #     'actual': self.onehot_lat_avg.val,
        #     }, self.step_counter)
        #self.tensorboard.add_scalar('Latency/estimated',self.lat_avg.val,self.step_counter)
        #self.tensorboard.add_scalar('Latency/actual', self.onehot_lat_avg.val, self.step_counter)
        
        # self.step_counter += 1
        
        # if step > 1 and (step % log_freq == 0) and self.gpu_id == 0:
        #     self.toc = time.time()
            
        #     batch_size = self.model.module.batch_size
        #     speed = 1.0 * (batch_size * torch.cuda.device_count() * log_freq) / (self.toc - self.tic)
        #     print("Epoch[%.3d] Batch[%.3d] Speed: %.6f samples/sec LR %.5f %s %s %s %s %s" 
        #       % (epoch, step, speed, self.w_scheduler.optimizer.param_groups[0]['lr'], self.loss_avg, 
        #          self.acc_avg, self.ce_avg, self.lat_avg,self.ener_avg))
            
        #     self.tic = time.time()



            
    def step_multi(self, input, target, epoch, step, log_freq, lat_losses, periods, multi_lat_constr, func):
        input = input.cuda()
        target = target.cuda()
        
        loss, ce, acc, lat, ener, onehot_lat, integrated_lat = func(input, target, lat_losses, periods, multi_lat_constr)
        
        self.loss_avg.update(loss, input.size(0))
        self.ce_avg.update(ce, input.size(0))
        self.acc_avg.update(acc, input.size(0))
        self.lat_avg.update(lat, input.size(0))
        self.ener_avg.update(ener, input.size(0))
        self.onehot_lat_avg.update(onehot_lat, input.size(0))
        self.integrated_lat_avg.update(integrated_lat, input.size(0))
        
        self.tensorboard.add_scalar('{}/Accuracy'.format(self.trainer_name),self.acc_avg.val, self.step_counter)
        self.tensorboard.add_scalar('{}/Total Loss'.format(self.trainer_name), self.loss_avg.val, self.step_counter)            
        
        self.tensorboard.add_scalars('{}/Latency'.format(self.trainer_name), {
            'estimated': self.lat_avg.val,
            'actual': self.onehot_lat_avg.val,
            }, self.step_counter)
        
        if self.rt_loss == 0:
            self.tensorboard.add_scalar('{}/Integrated Lat'.format(self.trainer_name), self.integrated_lat_avg.val, self.step_counter)
        else:
            self.tensorboard.add_scalar('{}/Total Utilization'.format(self.trainer_name), self.integrated_lat_avg.val, self.step_counter)
        self.step_counter += 1
        
        if step > 1 and (step % log_freq == 0) and self.gpu_id == 0:
            self.toc = time.time()
            
            batch_size = self.model.module.batch_size
            speed = 1.0 * (batch_size * torch.cuda.device_count() * log_freq) / (self.toc - self.tic)
            print("Epoch[%.3d] Batch[%.3d] Speed: %.6f samples/sec LR %.5f %s %s %s %s %s" 
              % (epoch, step, speed, self.w_scheduler.optimizer.param_groups[0]['lr'], self.loss_avg, 
                 self.acc_avg, self.ce_avg, self.lat_avg,self.ener_avg))
            
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
            self.decay_temperature()
            
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
                self.tensorboard.add_scalar('%s/Layer %s'% (self.trainer_name, str(i)),max_index+1, epoch)
                res.append(t_list)
                s = ' '.join([str(tmp) for tmp in t_list])
                f.write(s + '\n')
            val = np.array(res)
        return res
                
class Multi_Trainer:
    def __init__(
        self,
        trainer1: Trainer,
        trainer2: Trainer,
        trainer3: Trainer,
        total_constr: float,
    ) -> None:
        
        self.trainer1 = trainer1
        self.trainer2 = trainer2
        self.trainer3 = trainer3
        self.total_constr = total_constr
        
        self.tic = time.time()
        self.toc = time.time()
        
    def search(self, train_ds,val_ds, total_epochs, log_freq, warmup):
        
        ## Warmup 
        for epoch in range(warmup):
            for st, (input, target) in enumerate(train_ds):
                #trainer 1
                self.trainer1.tic = time.time()
                self.trainer1.step(input, target, epoch, st, log_freq,
                                   lambda x, y: self.trainer1.train_w(x, y))
                self.trainer1.w_scheduler.step()
                
                #trainer 2
                self.trainer2.tic = time.time()
                self.trainer2.step(input, target, epoch, st, log_freq, 
                                   lambda x, y: self.trainer2.train_w(x, y))
                self.trainer2.w_scheduler.step()
                
                #trainer 3
                self.trainer3.tic = time.time()
                self.trainer3.step(input, target, epoch, st, log_freq,
                                   lambda x, y: self.trainer3.train_w(x, y))
                self.trainer3.w_scheduler.step()
                
        for epoch in range(total_epochs):
            for st, (input, target) in enumerate(train_ds):
                self.trainer1.tic = time.time()
                self.trainer1.step_multi(input, target, epoch+warmup, st, log_freq, 
                                         [self.trainer2.lat_avg.val, self.trainer3.lat_avg.val], 
                                         [self.trainer2.period, self.trainer3.period],
                                         self.total_constr,
                                         lambda x, y, z, k, t: self.trainer1.train_t_multi(x, y, z, k, t))
            
                
                self.trainer2.tic = time.time()
                self.trainer2.step_multi(input, target, epoch+warmup, st, log_freq, 
                                         [self.trainer1.lat_avg.val, self.trainer3.lat_avg.val], 
                                         [self.trainer1.period, self.trainer3.period],
                                         self.total_constr,
                                         lambda x, y, z, k, t: self.trainer2.train_t_multi(x, y, z, k, t))
            

                self.trainer3.tic = time.time()
                self.trainer3.step_multi(input, target, epoch+warmup, st, log_freq, 
                                         [self.trainer2.lat_avg.val, self.trainer1.lat_avg.val],
                                         [self.trainer2.period, self.trainer1.period],
                                         self.total_constr,
                                         lambda x, y, z, k, t: self.trainer3.train_t_multi(x, y, z, k, t))
            
            self.trainer1.decay_temperature()                
            self.trainer2.decay_temperature()
            self.trainer3.decay_temperature()

            for st, (input, target) in enumerate(train_ds):
                self.trainer1.tic = time.time()
                self.trainer1.step_multi(input, target, epoch+warmup, st, log_freq, 
                                         [self.trainer2.lat_avg.val, self.trainer3.lat_avg.val], 
                                         [self.trainer2.period, self.trainer3.period],
                                         self.total_constr,
                                         lambda x, y, z, k, t: self.trainer1.train_w_multi(x, y, z, k, t))
                self.trainer1.w_scheduler.step()
                
                self.trainer2.tic = time.time()
                self.trainer2.step_multi(input, target, epoch+warmup, st, log_freq, 
                                         [self.trainer1.lat_avg.val, self.trainer3.lat_avg.val], 
                                         [self.trainer1.period, self.trainer3.period],                                         
                                         self.total_constr,
                                         lambda x, y, z, k, t: self.trainer2.train_w_multi(x, y, z, k, t))
                self.trainer2.w_scheduler.step()
                
                self.trainer3.tic = time.time()
                self.trainer3.step_multi(input, target, epoch+warmup, st, log_freq, 
                                         [self.trainer2.lat_avg.val, self.trainer1.lat_avg.val], 
                                         [self.trainer2.period, self.trainer1.period],
                                         self.total_constr,
                                         lambda x, y, z, k, t: self.trainer3.train_w_multi(x, y, z, k, t))
                self.trainer3.w_scheduler.step()
            
            # ## validation code

            # for st, (input, target) in enumerate(val_ds):
            #     self.trainer1.val_step(input, target, epoch, st, log_freq, 
            #                         lambda x, y: self.trainer1.validate(x,y))
            
            #     self.trainer2.val_step(input, target, epoch, st, log_freq, 
            #                         lambda x, y: self.trainer2.validate(x,y))
            
            #     self.trainer3.val_step(input, target, epoch, st, log_freq, 
            #                         lambda x, y: self.trainer3.validate(x,y))
                
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

    split = int(np.floor(0.7*len(dataset)))
    train_set, val_set = torch.utils.data.random_split(dataset, [split, len(dataset) - split])

    blocks = get_blocks(cifar10=True)
    
    model = FBNet(num_classes= config.num_cls_used,
              blocks=blocks,
              init_theta=config.init_theta,
              alpha=config.alpha,
              beta=config.beta,
              gamma=0,
              delta=0,
              sf_type=config.softmax_type,
              lat_const=config.lat_constr,
              loss_type= config.loss_type,
              speed_f="speed.txt",
	      energy_f="energy.txt")
    
    return train_set, val_set, model

def load_multi_train_objs(config:Config):
    
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
    
    model1 = FBNet(num_classes= config.num_cls_used,
              blocks=blocks,
              init_theta=config.init_theta,
              alpha=config.alpha,
              beta=config.beta,
              gamma=0,
              delta=0,
              sf_type=config.softmax_type,
              lat_const=config.lat_constr,
              loss_type= config.loss_type,
              speed_f="speed.txt",
	      energy_f="energy.txt")

    model2 = FBNet(num_classes= config.num_cls_used,
              blocks=blocks,
              init_theta=config.init_theta,
              alpha=config.alpha,
              beta=config.beta,
              gamma=0,
              delta=0,
              sf_type=config.softmax_type,
              lat_const=config.lat_constr,
              loss_type= config.loss_type,
              speed_f="speed.txt",
	      energy_f="energy.txt")
    
    model3 = FBNet(num_classes= config.num_cls_used,
              blocks=blocks,
              init_theta=config.init_theta,
              alpha=config.alpha,
              beta=config.beta,
              gamma=0,
              delta=0,
              sf_type=config.softmax_type,
              lat_const=config.lat_constr,
              loss_type= config.loss_type,
              speed_f="speed.txt",
	      energy_f="energy.txt")
    
    return train_set, val_set, model1, model2, model3

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
    train_set, val_set, model1, model2, model3 = load_multi_train_objs(config)
    train_data = prepare_dataloader(train_set, batch_size)
    val_data = prepare_dataloader(val_set, int(batch_size/2))
    writer = SummaryWriter("./runs/{}".format(exp_name))
    trainer1 = Trainer("trainer1", model1, train_data, val_data, rank, save_every, config.lr_scheduler_params, writer, exp_name, config, config.rt_loss, config.p1)
    trainer2 = Trainer("trainer2", model2, train_data, val_data, rank, save_every, config.lr_scheduler_params, writer, exp_name, config, config.rt_loss, config.p2)
    trainer3 = Trainer("trainer3", model3, train_data, val_data, rank, save_every, config.lr_scheduler_params, writer, exp_name, config, config.rt_loss, config.p3)
    
    multi_trainer = Multi_Trainer(trainer1, trainer2, trainer3, config.total_lat_constr)
    multi_trainer.search(train_data, val_data,
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
    parser.add_argument('--lr_mul', default=1, type=float, help="Learning rate multiplier")
    parser.add_argument('--temp', default=5.0, type=float, help="Gumbel Softmax Temperature")
    parser.add_argument('--temp_decay', default=0.956, type=float, help="Temperature decay ratio")
    parser.add_argument('--softmax', default=0, type=int, help="Softmax type; 0: Gumbel, 1: Softmax")
    parser.add_argument('--lat_constr', default=-1, type=float, help="Latency constraint; -1: No constraint, other: < constr")
    parser.add_argument('--loss_type', default= 0, type=int, help='Loss function type; 0: latency-aware, 1:latency-constraint, 2:weighted, 3: Pareto-optimal')
    parser.add_argument('--alpha', default=0.2, type=float, help="Latency penalty parameter")
    parser.add_argument('--p1', default=12.9, type=float, help="Model 1's period")
    parser.add_argument('--p2', default=12.9, type=float, help="Model 2's period")
    parser.add_argument('--p3', default=12.9, type=float, help="Model 3's period")
    parser.add_argument('--rt_loss', default=0, type=int, help="Loss type: RT or Not")
    
    
    args = parser.parse_args()
    args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
    if args.lat_constr == -1 and args.loss_type != 0:
        print("latency constr and loss function not aligned")
        exit(-1)    
    print(args)

    config = Config()
    config.w_lr = config.w_lr*args.lr_mul
    config.t_lr = config.t_lr*args.lr_mul
    
    config.init_temperature = args.temp
    config.temperature_decay = args.temp_decay
    config.softmax_type = args.softmax
    config.total_lat_constr = args.lat_constr
    config.alpha = args.alpha
    config.loss_type = args.loss_type
    
    config.p1 = args.p1
    config.p2 = args.p2
    config.p3 = args.p3
    
    config.rt_loss = args.rt_loss
    
    world_size = args.gpus
    mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size, config, args.save), nprocs=world_size)
