from torch.utils.data import Dataset, DataLoader
from cutout import Cutout
import copy
import torchvision.transforms as transforms
import random
import numpy as np
from randaugment import RandAugment
from PIL import Image
import json
import os
import torch


class TransformTwice:
    def __init__(self, transform1,transform2):
        self.transform1 = transform1
        self.transform2 = transform2

    def __call__(self, inp):
        out1 = self.transform1(inp)
        out2 = self.transform2(inp)
        return out1, out2

def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

class cifar_dataset(Dataset): 
    def __init__(self, dataset, r, noise_mode, root_dir, transform, mode, noise_file='', pred=[], probability=[], log='',indices=None,imbalance=None): 
        with open(f'LT_{dataset}.txt') as f:
            num_in_classall=eval(f.readline())
        num_classes=int(dataset[5:])
        
        self.r = r # noise ratio
        self.transform = transform
        self.mode = mode  
        self.transition = {0:0,2:0,4:7,7:7,1:1,9:1,3:5,5:3,6:6,8:8} # class transition for asymmetric noise
     
        if self.mode=='test':
            if dataset=='cifar10':                
                test_dic = unpickle('%s/test_batch'%root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))  
                self.test_label = test_dic['labels']
            elif dataset=='cifar100':
                test_dic = unpickle('%s/test'%root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))  
                self.test_label = test_dic['fine_labels']                            
        else:    
            train_data=[]
            train_label=[]
            if dataset=='cifar10': 
                for n in range(1,6):
                    dpath = '%s/data_batch_%d'%(root_dir,n)
                    data_dic = unpickle(dpath)
                    train_data.append(data_dic['data'])
                    train_label = train_label+data_dic['labels']
                train_data = np.concatenate(train_data)
            elif dataset=='cifar100':    
                train_dic = unpickle('%s/train'%root_dir)
                train_data = train_dic['data']
                train_label = train_dic['fine_labels']
            train_data = train_data.reshape((50000, 3, 32, 32))
            train_data = train_data.transpose((0, 2, 3, 1))

            if self.mode == "labeled":
                pred_idx = pred.nonzero()[0]
                pred_idx = pred_idx.flatten()
                self.probability = probability
                    
            used_indices=[]
            num_in_class=num_in_classall[imbalance].copy()
            #print(num_in_class)
            LT_data=[]
            clean_targets=[]
            LT_targets=[]
            for i in indices:
                # c=noise_label[i]
                c=train_label[i]
                if num_in_class[c]>0:
                    LT_data.append(train_data[i])
                    clean_targets.append(train_label[i])
                    used_indices.append(i)
                    num_in_class[c]=num_in_class[c]-1

                    # LT_targets.append(noise_label[i])
            num_in_class=np.asarray(num_in_classall[imbalance])
            corruptprobability=np.broadcast_to(num_in_class,[num_classes,num_classes]).copy()
            corruptprobability[np.eye(num_classes)==1]=0
            corruptprobability=corruptprobability/np.sum(corruptprobability,axis=1,keepdims=True)*self.r
            corruptprobability[np.eye(num_classes)==1]=1-self.r
            #print(corruptprobability)
            # exit()
            for i in range(len(used_indices)):
                c=clean_targets[i]
                LT_targets.append(np.random.choice(num_classes, p=corruptprobability[c]))
            
            self.train_data = LT_data 
            self.noise_label = LT_targets 


            
                
    def __getitem__(self, index):
        if self.mode=='labeled':
            img, target, prob = self.train_data[index], self.noise_label[index], self.probability[index]
            img = Image.fromarray(img)
            img1 = self.transform(img) 
            return img1, target, prob            
        elif self.mode=='all':
            img, target = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target, index        
        elif self.mode=='test':
            img, target = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target
           
    def __len__(self):
        if self.mode!='test':
            return len(self.train_data)
        else:
            return len(self.test_data)         
        
        
class cifar_dataloader():  
    def __init__(self, dataset, r, noise_mode, batch_size, num_workers, root_dir, log, noise_file='',imbalance=None):
        self.dataset = dataset
        self.r = r
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.u=7
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.log = log
        self.noise_file = noise_file
        self.imbalance=imbalance
        if self.dataset=='cifar10':
            self.transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                ])

            self.strong_transform = copy.deepcopy(self.transform_train)
            self.strong_transform.transforms.insert(0, RandAugment(3,5))
            self.cutout_transform = transforms.Compose([
                     transforms.RandomCrop(32, padding=4),
                     transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(),
                     transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                     Cutout(n_holes=1, length=16)
                     ])



            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                ])    
        elif self.dataset=='cifar100':    
            self.transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ]) 
            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ])   
            self.strong_transform = copy.deepcopy(self.transform_train)
            self.strong_transform.transforms.insert(0, RandAugment(3,5))

        indices = np.arange(50000)
        np.random.shuffle(indices)
        self.indices=indices

    def run(self,mode,pred=[],prob=[]):
        if mode=='warmup':
            all_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode="all",noise_file=self.noise_file,indices=self.indices,imbalance=self.imbalance)
            trainloader = DataLoader(
                dataset=all_dataset, 
                batch_size=64,
                shuffle=True,
                num_workers=self.num_workers)

            return trainloader
                                     
        elif mode=='train':
            labeled_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=TransformTwice(self.transform_train,self.strong_transform,), mode="labeled", noise_file=self.noise_file, pred=pred, probability=prob,log=self.log,indices=self.indices,imbalance=self.imbalance)
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size*self.u,
                shuffle=True,
                num_workers=self.num_workers,
                drop_last=True)
            
            return labeled_trainloader
        
        elif mode=='test':
            test_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_test, mode='test',indices=self.indices,imbalance=self.imbalance)      
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return test_loader
        
        elif mode=='eval_train':
            eval_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode='all', noise_file=self.noise_file,indices=self.indices,imbalance=self.imbalance)      
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                shuffle=True,
                num_workers=self.num_workers)          
            return eval_loader        
