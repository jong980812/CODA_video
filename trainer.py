import os
import sys
import argparse
import torch
import numpy as np
import random
from random import shuffle
from collections import OrderedDict
import dataloaders
from dataloaders.utils import *
from torch.utils.data import DataLoader
import learners

class Trainer:

    def __init__(self, args, seed, metric_keys, save_keys):

        # process inputs
        self.seed = seed
        self.metric_keys = metric_keys
        self.save_keys = save_keys
        self.log_dir = args.log_dir
        self.batch_size = args.batch_size
        self.workers = args.workers
        self.num_test_videos = []
        # model load directory
        self.model_top_dir = args.log_dir
        self.dataset=None
        # select dataset
        self.grayscale_vis = False
        self.top_k = 1
        if args.dataset == 'CIFAR10':
            Dataset = dataloaders.iCIFAR10
            num_classes = 10
            self.dataset_size = [32,32,3]
        elif args.dataset == 'CIFAR100':
            Dataset = dataloaders.iCIFAR100
            num_classes = 100
            self.dataset_size = [32,32,3]
        elif args.dataset == 'ImageNet_R':
            Dataset = dataloaders.iIMAGENET_R
            num_classes = 200
            self.dataset_size = [224,224,3]
            self.top_k = 1
        elif args.dataset == 'DomainNet':
            Dataset = dataloaders.iDOMAIN_NET
            num_classes = 345
            self.dataset_size = [224,224,3]
            self.top_k = 1
        elif args.dataset=='UCF101':
            Dataset = dataloaders.iUCF101
            self.dataset_size = [224,224,3]
            self.top_k = 1
            num_classes = 101
        elif args.dataset=='SSV2':
            Dataset = dataloaders.iSSV2
            self.dataset_size = [224,224,3]
            self.top_k = 1
            num_classes = 174
        else:
            raise ValueError('Dataset not implemented!')
        self.dataset=args.dataset
        # upper bound flag
        if args.upper_bound_flag:
            args.other_split_size = num_classes
            args.first_split_size = num_classes
        
        if args.anno_path is not None:
            print(f"*****Load {args.anno_path} *****")
            # Load PKL
            import pickle
            with open(args.anno_path, 'rb') as file:
                anno_list = pickle.load(file)
            class_names = []
            for task in anno_list['train']:
                class_names+=list(task.keys())
            # txt 파일 내용을 읽어들여 활동 이름과 숫자를 매핑하는 딕셔너리를 생성
            activity_to_number = {}
            with open(f'./anno_list/{args.dataset}/class_list.txt', 'r') as file:  # 'your_file_path.txt'는 실제 txt 파일의 경로로 대체해야 합니다.
                for line in file:
                    number, activity_name = line.strip().split(',', 1)
                    activity_to_number[activity_name] = int(number)

            # 주어진 리스트의 각 항목을 해당하는 숫자로 변환
            activities_numbers = [activity_to_number[activity] for activity in class_names]
            class_order = activities_numbers
            class_order_logits = activities_numbers
        #!original
        else:
        # load tasks
            class_order = np.arange(num_classes).tolist()# class 순서대로 넘버 적혀있음.
            class_order_logits = np.arange(num_classes).tolist()
        
        # if self.seed > 0 and args.rand_split:
        #     print('=============================================')
        #     print('Shuffling....')
        #     print('pre-shuffle:' + str(class_order))
        #     random.seed(self.seed)
        #     random.shuffle(class_order)
        #     print('post-shuffle:' + str(class_order))
        #     print('=============================================')
        self.tasks = []#! 클래스 number 들어감. 이중리스트
        self.tasks_logits = []
        
        p = 0
        while p < num_classes and (args.max_task == -1 or len(self.tasks) < args.max_task):
            inc = args.other_split_size if p > 0 else args.first_split_size
            self.tasks.append(class_order[p:p+inc])
            self.tasks_logits.append(class_order_logits[p:p+inc])
            p += inc
        self.num_tasks = len(self.tasks)#!
        '''
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
        [10, 11, 12, 13, 14, 15, 16, 17, 18, 19], 
        [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
        '''
        self.task_names = [str(i+1) for i in range(self.num_tasks)]

        # number of tasks to perform
        if args.max_task > 0:
            self.max_task = min(args.max_task, len(self.task_names))
        else:
            self.max_task = len(self.task_names)#! 그냥 전체 태스크 들어감.

        # datasets and dataloaders
        k = 1 # number of transforms per image
        if args.model_name.startswith('vit'):
            resize_imnet = True
        else:
            resize_imnet = False
            
        if args.dataset!='UCF101' and args.dataset!='SSV2':
            train_transform = dataloaders.utils.get_transform(dataset=args.dataset, phase='train', aug=args.train_aug, resize_imnet=resize_imnet)
            test_transform  = dataloaders.utils.get_transform(dataset=args.dataset, phase='test', aug=args.train_aug, resize_imnet=resize_imnet)
            self.train_dataset = Dataset(args.dataroot, train=True, lab = True, tasks=self.tasks,
                                download_flag=True, transform=train_transform, 
                                seed=self.seed, rand_split=args.rand_split, validation=args.validation)
            # self.train_dataset  = Dataset(args.dataroot, train=False, tasks=self.tasks,
            #                         download_flag=False, transform=test_transform, 
            #                         seed=self.seed, rand_split=args.rand_split, validation=args.validation)
            self.test_dataset  = Dataset(args.dataroot, train=False, tasks=self.tasks,
                                    download_flag=False, transform=test_transform, 
                                    seed=self.seed, rand_split=args.rand_split, validation=args.validation)
        else:
            # train_transform = dataloaders.utils.get_transform(dataset=args.dataset, phase='train', aug=args.train_aug, resize_imnet=resize_imnet)
            anno_path = f'./anno_list/{args.dataset}/0.1_train.csv' if args.dataset=='SSV2' else f'./anno_list/{args.dataset}/train.csv'
            self.train_dataset = Dataset(args.dataroot, train=True, lab = True, tasks=self.tasks,
                                download_flag=True, 
                                seed=self.seed, rand_split=args.rand_split, validation=args.validation,args = args,anno_path=anno_path)
            anno_path = f'./anno_list/{args.dataset}/0.1_test.csv' if args.dataset=='SSV2' else f'./anno_list/{args.dataset}/test.csv'
    
            self.test_dataset  = Dataset(args.dataroot, train=False, tasks=self.tasks,
                                    download_flag=False,  
                                    seed=self.seed, rand_split=args.rand_split, validation=args.validation, args = args, anno_path=anno_path)
        # self.train_dataset = 
        # self.test_dataset =

        # for oracle
        self.oracle_flag = args.oracle_flag
        self.add_dim = 0

        # Prepare the self.learner (model)
        self.learner_config = {'num_classes': num_classes,
                        'lr': args.lr,
                        'debug_mode': args.debug_mode == 1,
                        'momentum': args.momentum,
                        'weight_decay': args.weight_decay,
                        'schedule': args.schedule,
                        'schedule_type': args.schedule_type,
                        'model_type': args.model_type,
                        'model_name': args.model_name,
                        'optimizer': args.optimizer,
                        'gpuid': args.gpuid,
                        'memory': args.memory,
                        'temp': args.temp,
                        'out_dim': num_classes,
                        'overwrite': args.overwrite == 1,
                        'DW': args.DW,
                        'batch_size': args.batch_size,
                        'upper_bound_flag': args.upper_bound_flag,
                        'tasks': self.tasks_logits,
                        'top_k': self.top_k,
                        'prompt_param':[self.num_tasks,args.prompt_param],
                        'frame_prompt':args.frame_prompt,
                        'clip':args.clip
                        }
        self.learner_type, self.learner_name = args.learner_type, args.learner_name
        self.learner = learners.__dict__[self.learner_type].__dict__[self.learner_name](self.learner_config)

    def task_eval(self, t_index, local=False, task='acc'):

        val_name = self.task_names[t_index]
        print('validation split name:', val_name)
        
        # eval
        self.test_dataset.load_dataset(t_index, train=True)
        test_loader  = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.workers)
        if local:
            return self.learner.validation(test_loader, task_in = self.tasks_logits[t_index], task_metric=task)
        else:
            return self.learner.validation(test_loader, task_metric=task)

    def train(self, avg_metrics):
    
        # temporary results saving
        temp_table = {}
        for mkey in self.metric_keys: temp_table[mkey] = []
        temp_dir = self.log_dir + '/temp/'
        if not os.path.exists(temp_dir): os.makedirs(temp_dir)

        # for each task
        for i in range(self.max_task):

            # save current task index
            self.current_t_index = i

            # print name
            train_name = self.task_names[i]
            print('======================', train_name, '=======================')

            # load dataset for task
            task = self.tasks_logits[i]
            if self.oracle_flag:
                self.train_dataset.load_dataset(i, train=False)
                self.learner = learners.__dict__[self.learner_type].__dict__[self.learner_name](self.learner_config)
                self.add_dim += len(task)
            else:
                self.train_dataset.load_dataset(i, train=True)
                self.add_dim = len(task)

            # set task id for model (needed for prompting)
            try:
                self.learner.model.module.task_id = i
            except:
                self.learner.model.task_id = i

            # add valid class to classifier
            self.learner.add_valid_output_dim(self.add_dim)

            # load dataset with memory
            # self.train_dataset.append_coreset(only=False)

            # load dataloader
            train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=int(self.workers))

            # increment task id in prompting modules
            if i > 0:
                try:
                    if self.learner.model.module.prompt is not None:
                        if isinstance(self.learner.model.module.prompt,torch.nn.ModuleDict):
                            if self.learner_name=='L2P_spatio':
                                self.learner.model.module.prompt['c'].process_task_count()
                                self.learner.model.module.prompt['t'].process_task_count()
                                self.learner.model.module.prompt['s'].process_task_count()
                            else:
                                self.learner.model.module.prompt['up'].process_task_count()
                                self.learner.model.module.prompt['down'].process_task_count() 
                        else:
                            self.learner.model.module.prompt.process_task_count()
                except:
                    if self.learner.model.prompt is not None:
                        self.learner.model.prompt.process_task_count()

            # learn
            self.test_dataset.load_dataset(i, train=False)
            test_loader  = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.workers)
            model_save_dir = self.model_top_dir + '/models/repeat-'+str(self.seed+1)+'/task-'+self.task_names[i]+'/'
            if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)
            avg_train_time = self.learner.learn_batch(train_loader, self.train_dataset, model_save_dir, test_loader, i)

            # save model
            self.learner.save_model(model_save_dir)
            
            # evaluate acc
            acc_table = []
            acc_table_ssl = []
            self.reset_cluster_labels = True
            for j in range(i+1):
                acc_table.append(self.task_eval(j))
            temp_table['acc'].append(np.mean(np.asarray(acc_table)))

            # save temporary acc results
            for mkey in ['acc']:
                save_file = temp_dir + mkey + '.csv'
                np.savetxt(save_file, np.asarray(temp_table[mkey]), delimiter=",", fmt='%.2f')  

            if avg_train_time is not None: avg_metrics['time']['global'][i] = avg_train_time

        return avg_metrics 
    
    def summarize_acc(self, acc_dict, acc_table, acc_table_pt):

        # unpack dictionary
        avg_acc_all = acc_dict['global']
        avg_acc_pt = acc_dict['pt']
        avg_acc_pt_local = acc_dict['pt-local']

        # Calculate average performance across self.tasks
        # Customize this part for a different performance metric
        avg_acc_history = [0] * self.max_task
        for i in range(self.max_task):
            train_name = self.task_names[i]
            cls_acc_sum = 0
            for j in range(i+1):
                val_name = self.task_names[j]
                cls_acc_sum += acc_table[val_name][train_name]
                avg_acc_pt[j,i,self.seed] = acc_table[val_name][train_name]
                avg_acc_pt_local[j,i,self.seed] = acc_table_pt[val_name][train_name]
            avg_acc_history[i] = cls_acc_sum / (i + 1)

        # Gather the final avg accuracy
        avg_acc_all[:,self.seed] = avg_acc_history

        # repack dictionary and return
        return {'global': avg_acc_all,'pt': avg_acc_pt,'pt-local': avg_acc_pt_local}
    def get_len_video(self):
        len_videos = []
        for i in range(self.max_task):
            self.test_dataset.load_dataset(i, train=True)
            len_videos.append(len(self.test_dataset))
        return len_videos
    
    def evaluate(self, avg_metrics):

        self.learner = learners.__dict__[self.learner_type].__dict__[self.learner_name](self.learner_config)

        # store results
        metric_table = {}
        metric_table_local = {}
        for mkey in self.metric_keys:
            metric_table[mkey] = {}
            metric_table_local[mkey] = {}
             
        acc_list = [[0 for i in range(self.max_task)] for j in range(self.max_task)]
        for i in range(self.max_task):

            # increment task id in prompting modules
            # if i > 0:
            #     try:
            #         if self.learner.model.module.prompt is not None:
            #             if isinstance(self.learner.model.module.prompt,torch.nn.ModuleDict):
            #                 self.learner.model.module.prompt['c'].process_task_count()
            #                 self.learner.model.module.prompt['t'].process_task_count()
            #                 self.learner.model.module.prompt['s'].process_task_count()
            #             else:
            #                 self.learner.model.module.prompt.process_task_count()
            #     except:
            #         if self.learner.model.prompt is not None:
            #             self.learner.model.prompt.process_task_count()
            if i > 0:
                try:
                    if self.learner.model.module.prompt is not None:
                        if isinstance(self.learner.model.module.prompt,torch.nn.ModuleDict):
                            if self.learner_name=='L2P_spatio':
                                self.learner.model.module.prompt['c'].process_task_count()
                                self.learner.model.module.prompt['t'].process_task_count()
                                self.learner.model.module.prompt['s'].process_task_count()
                            else:
                                self.learner.model.module.prompt['up'].process_task_count()
                                self.learner.model.module.prompt['down'].process_task_count() 
                        else:
                            self.learner.model.module.prompt.process_task_count()
                except:
                    if self.learner.model.prompt is not None:
                        self.learner.model.prompt.process_task_count()

            # load model
            model_save_dir = self.model_top_dir + '/models/repeat-'+str(self.seed+1)+'/task-'+self.task_names[i]+'/'
            self.learner.task_count = i 
            self.learner.add_valid_output_dim(len(self.tasks_logits[i]))
            self.learner.pre_steps()
            self.learner.load_model(model_save_dir)

            # set task id for model (needed for prompting)
            try:
                self.learner.model.module.task_id = i
            except:
                self.learner.model.task_id = i

            # evaluate acc
            metric_table['acc'][self.task_names[i]] = OrderedDict()
            metric_table_local['acc'][self.task_names[i]] = OrderedDict()
            self.reset_cluster_labels = True
       
            for j in range(i+1):
                val_name = self.task_names[j]
                metric_table['acc'][val_name][self.task_names[i]]= self.task_eval(j)
                acc_list[i][j]=metric_table['acc'][val_name][self.task_names[i]]
                
            for j in range(i+1):
                val_name = self.task_names[j]
                metric_table_local['acc'][val_name][self.task_names[i]] = 0#self.task_eval(j, local=True)


        # for task in range(self.max_task):
        #     accs = 
        #     for i,v in metric_table['acc'][:][task]:
        #         accs[i]=v
        #     acc_list.append(accs)
        # if self.dataset=='SSV2':
        len_videos = self.get_len_video()
        print_matrix_with_aligned_averages(acc_list,len_videos)
        # summarize metrics
        avg_metrics['acc'] = self.summarize_acc(avg_metrics['acc'], metric_table['acc'],  metric_table_local['acc'])

        return avg_metrics
def calculate_weighted_averages(matrix, n_videos):
    row_averages = [np.average(row[:i+1], weights =n_videos[:i+1]) if len(row) > 0 else 0 for i, row in enumerate(matrix)]
    overall_average = np.mean(row_averages)
    return row_averages, overall_average
def print_matrix_with_aligned_averages(matrix, n_videos):
    row_averages, overall_average = calculate_weighted_averages(matrix, n_videos)
    
    max_row_length = max(len(row) for row in matrix)
    row_format = "{" + f":<{max_row_length * 3}" + "}"
    
    for i, row in enumerate(matrix):
        row_str = ", ".join(f"{num:.2f}" for num in row)  
        formatted_row = row_format.format(row_str)  
        print(f"[{formatted_row}] | Weighted Average: {row_averages[i]:.2f}")
    
    print(" " * (max_row_length * 3 + 2) + f"Overall Weighted Average: {overall_average:.2f}") 