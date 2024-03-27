from __future__ import print_function
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from types import MethodType
import models
from utils.metric import accuracy, AverageMeter, Timer
import numpy as np
from torch.optim import Optimizer
import contextlib
import os
from .default import NormalNN, weight_reset, accumulate_acc
import copy
import torchvision
from utils.schedulers import CosineSchedule
from torch.autograd import Variable, Function

class Prompt(NormalNN):

    def __init__(self, learner_config):
        self.prompt_param = learner_config['prompt_param']
        super(Prompt, self).__init__(learner_config)

    def update_model(self, inputs, targets):

        # logits
        logits, prompt_loss,q = self.model(inputs, train=True)
        logits = logits[:,:self.valid_out_dim]

        # ce with heuristic
        logits[:,:self.last_valid_out_dim] = -float('inf')
        dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        total_loss = self.criterion(logits, targets.long(), dw_cls)

        # ce loss
        total_loss = total_loss + prompt_loss.sum()

        # step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.detach(), logits,q

    # sets model optimizers
    def init_optimizer(self):

        # # parse optimizer args
        # # Multi-GPU
        # if len(self.config['gpuid']) > 1:
        #     params_to_opt = list(self.model.module.prompt.parameters()) + list(self.model.module.last.parameters())
        # else:
        #     params_to_opt = list(self.model.prompt.parameters()) + list(self.model.last.parameters())
        # print('*****************************************')
        # optimizer_arg = {'params':params_to_opt,
        #                  'lr':self.config['lr'],
        #                  'weight_decay':self.config['weight_decay']}
        # if self.config['optimizer'] in ['SGD','RMSprop']:
        #     optimizer_arg['momentum'] = self.config['momentum']
        # elif self.config['optimizer'] in ['Rprop']:
        #     optimizer_arg.pop('weight_decay')
        # elif self.config['optimizer'] == 'amsgrad':
        #     optimizer_arg['amsgrad'] = True
        #     self.config['optimizer'] = 'Adam'
        # elif self.config['optimizer'] == 'Adam':
        #     optimizer_arg['betas'] = (self.config['momentum'],0.999)

        # # create optimizers
        # self.optimizer = torch.optim.__dict__[self.config['optimizer']](**optimizer_arg)
        
        # # create schedules
        # if self.schedule_type == 'cosine':
        #     self.scheduler = CosineSchedule(self.optimizer, K=self.schedule[-1])
        # elif self.schedule_type == 'decay':
        #     self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.schedule, gamma=0.1)
    # Multi-GPU
        if len(self.config['gpuid']) > 1:
            if isinstance(self.model.module.prompt,dict):
                prompt_params=[]
                for k,v in (self.model.module.prompt.items()):
                    prompt_params+=list(self.model.module.prompt[k].parameters())
            else:
                prompt_params = list(self.model.module.prompt.parameters())#+list(self.model.last.parameters())
            last_params = list(self.model.module.last.parameters())
            
            # total_param = sum(p.numel() for p in self.model.module.prompt.parameters() if p.requires_grad)+sum(p.numel() for p in self.model.module.last.parameters() if p.requires_grad)
        else:
            if isinstance(self.model.prompt,dict):
                prompt_params=[]
                for k,v in (self.model.prompt.items()):
                    prompt_params+=list(self.model.prompt[k].parameters())
            else:
                prompt_params = list(self.model.prompt.parameters())#+list(self.model.last.parameters())
            # total_param = sum(p.numel() for p in self.model.prompt.parameters() if p.requires_grad)+sum(p.numel() for p in self.model.last.parameters() if p.requires_grad)
            
            last_params = list(self.model.last.parameters())
        # print(f'*************** Total param: {total_param}*******************')
        # 기본 파라미터 그룹 설정 (last 제외)
        optimizer_arg = [{'params': prompt_params,
                        'lr': self.config['lr'],
                        'weight_decay': self.config['weight_decay']}]

        # # last 파라미터 그룹에 대한 설정 (학습률 100배 증가)
        # optimizer_arg=[{'params': last_params, 
        #                     'lr': self.config['lr'], 
        #                     'weight_decay': self.config['weight_decay']}]
        optimizer_arg.append({'params': last_params, 
                            'lr': self.config['lr'], 
                            'weight_decay': self.config['weight_decay']})
        # 옵티마이저별 추가 설정
        if self.config['optimizer'] in ['SGD', 'RMSprop']:
            for arg in optimizer_arg:
                arg['momentum'] = self.config['momentum']
        elif self.config['optimizer'] in ['Rprop']:
            for arg in optimizer_arg:
                arg.pop('weight_decay', None)  # weight_decay 제거
        elif self.config['optimizer'] == 'amsgrad':
            self.config['optimizer'] = 'Adam'
            for arg in optimizer_arg:
                arg['amsgrad'] = True
        elif self.config['optimizer'] == 'Adam':
            for arg in optimizer_arg:
                arg['betas'] = (self.config['momentum'], 0.999)

        # 옵티마이저 생성
        self.optimizer = torch.optim.__dict__[self.config['optimizer']](optimizer_arg)

        # 스케줄러 생성
        if self.schedule_type == 'cosine':
            self.scheduler = CosineSchedule(self.optimizer, K=self.schedule[-1])
        elif self.schedule_type == 'decay':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.schedule, gamma=0.1)


    def create_model(self):
        pass

    def cuda(self):
        torch.cuda.set_device(self.config['gpuid'][0])
        self.model = self.model.cuda()
        self.criterion_fn = self.criterion_fn.cuda()

        # Multi-GPU
        if len(self.config['gpuid']) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.config['gpuid'], output_device=self.config['gpuid'][0])
        return self

# Our method!
class CODAPrompt(Prompt):

    def __init__(self, learner_config):
        super(CODAPrompt, self).__init__(learner_config)

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag = 'coda',prompt_param=self.prompt_param)
        return model

# @article{wang2022dualprompt,
#   title={DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning},
#   author={Wang, Zifeng and Zhang, Zizhao and Ebrahimi, Sayna and Sun, Ruoxi and Zhang, Han and Lee, Chen-Yu and Ren, Xiaoqi and Su, Guolong and Perot, Vincent and Dy, Jennifer and others},
#   journal={European Conference on Computer Vision},
#   year={2022}
# }
class DualPrompt(Prompt):

    def __init__(self, learner_config):
        super(DualPrompt, self).__init__(learner_config)

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag = 'dual', prompt_param=self.prompt_param)
        return model

# @inproceedings{wang2022learning,
#   title={Learning to prompt for continual learning},
#   author={Wang, Zifeng and Zhang, Zizhao and Lee, Chen-Yu and Zhang, Han and Sun, Ruoxi and Ren, Xiaoqi and Su, Guolong and Perot, Vincent and Dy, Jennifer and Pfister, Tomas},
#   booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
#   pages={139--149},
#   year={2022}
# }
class L2P(Prompt):

    def __init__(self, learner_config):
        super(L2P, self).__init__(learner_config)

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag = 'l2p',prompt_param=self.prompt_param)
        return model


class L2P_video(Prompt):

    def __init__(self, learner_config):
        super(L2P_video, self).__init__(learner_config)

    def create_model(self):
        cfg = self.config
        frame_prompt = cfg['frame_prompt']
        clip = cfg['clip']
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag = 'l2p_video',prompt_param=self.prompt_param,frame_prompt=frame_prompt,clip=clip)
        return model
class L2P_spatio(Prompt):

    def __init__(self, learner_config):
        super(L2P_spatio, self).__init__(learner_config)

    def create_model(self):
        cfg = self.config
        frame_prompt = cfg['frame_prompt']
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag = 'l2p_spatio',prompt_param=self.prompt_param,frame_prompt=frame_prompt)
        return model
class CODA_video(Prompt):

    def __init__(self, learner_config):
        super(CODA_video, self).__init__(learner_config)

    def create_model(self):
        cfg = self.config
        clip = cfg['clip']
        frame_prompt = cfg['frame_prompt']
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag = 'coda_video',prompt_param=self.prompt_param,frame_prompt=frame_prompt,clip=clip)
        return model
class CODA_adapter(Prompt):

    def __init__(self, learner_config):
        super(CODA_adapter, self).__init__(learner_config)

    def create_model(self):
        cfg = self.config
        clip = cfg['clip']
        frame_prompt = cfg['frame_prompt']
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag = 'coda_adapter',prompt_param=self.prompt_param,frame_prompt=frame_prompt,clip=clip)
        return model

class L2P_adapter(Prompt):

    def __init__(self, learner_config):
        super(L2P_adapter, self).__init__(learner_config)

    def create_model(self):
        cfg = self.config
        clip = cfg['clip']
        frame_prompt = cfg['frame_prompt']
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag = 'l2p_adapter',prompt_param=self.prompt_param,frame_prompt=frame_prompt,clip=clip)
        return model