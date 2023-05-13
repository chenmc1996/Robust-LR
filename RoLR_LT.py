from __future__ import print_function 
from torch.cuda.amp import autocast, GradScaler
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import argparse
import numpy as np
from PreResNet import *
from sklearn.mixture import GaussianMixture
import dataloader_cifar_LT as dataloader

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize') 
parser.add_argument('--warm_up', default=15, type=int, help='warm epochs') 
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--noise_mode',  default='sym')
parser.add_argument('--save_name', type=str, default='test')
parser.add_argument('--lambda_u', default=1, type=float, help='weight for unsupervised loss')
parser.add_argument('--lambda_l', default=1, type=float, help='weight for supervised loss')
parser.add_argument('--lambda_p', default=0, type=float, help='weight for penalty')
parser.add_argument('--T', default=1, type=float, help='weight for unsupervised loss')
parser.add_argument('--num_epochs', default=500, type=int)
parser.add_argument('--r', default=0.5, type=float, help='noise ratio')
parser.add_argument('--conf_mode',  default='O')
parser.add_argument('--class_weight', type=str)
parser.add_argument('--imbalance', default='0.1', type=float)
parser.add_argument('--id', default='')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--data_path', default='./cifar-10', type=str, help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--resume', default='', type=str)
args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
with open(f'LT_{args.dataset}.txt') as f:
    num_in_classall=eval(f.readline())
num_samples=sum(num_in_classall[args.imbalance])

def compute_pos_weights(cls_repr: torch.Tensor,sqrt=False) -> torch.Tensor:
    if sqrt:
        cls_repr = torch.sqrt(cls_repr)
    total_weight = cls_repr.sum()
    weights = 1/torch.div(cls_repr, total_weight)
    # Standardize the weights
    #return torch.div(weights, torch.min(weights))
    return torch.div(weights, torch.sum(weights))*args.num_class

def ENS_weight(samples_per_cls):
    beta=0.999
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * args.num_class

    weights = torch.tensor(weights).float()
    return weights.reshape(1,-1).cuda()
if args.class_weight=='ENS':
    class_weight=ENS_weight(np.asarray(num_in_classall[args.imbalance]))
elif args.class_weight=='freq':
    class_weight=compute_pos_weights(torch.Tensor(num_in_classall[args.imbalance])).reshape(1,-1).cuda()
elif args.class_weight=='freq_sqrt':
    class_weight=compute_pos_weights(torch.Tensor(num_in_classall[args.imbalance]),sqrt=True).reshape(1,-1).cuda()

class prettyfloat(float):
    def __repr__(self):
        return "%0.2f" % self

def save_model( save_name, save_path,net1,net2,optimizer1,optimizer2):
    save_filename = os.path.join(save_path, save_name)
    torch.save({'net1': net1.state_dict(),
        'net2': net2.state_dict(),
                'optimizer1': optimizer1.state_dict(),
                'optimizer2': optimizer2.state_dict(),
                }, save_filename)
    print(f"model saved: {save_filename}")

def entropy(p):
    return - torch.sum(p * torch.log(p), axis=-1)

class SoftCELoss(object):
    def __call__(self, outputs, targets,weight=None):
        if weight is None:
            probs= torch.softmax(outputs, dim=1)
            Lx = -torch.mean(torch.sum(F.log_softmax(outputs, dim=1) * targets, dim=1))
        else:
            probs= torch.softmax(outputs, dim=1)
            Lx = -torch.mean(torch.sum(F.log_softmax(outputs, dim=1) * targets * weight, dim=1))
        return Lx

def load_model(load_path,net1,net2,optimizer1,optimizer2):
    checkpoint = torch.load(load_path)
    for key in checkpoint.keys():
        if 'net1' in key:
            net1.load_state_dict(checkpoint[key])
        elif 'net2' in key:
            net2.load_state_dict(checkpoint[key])
        elif key == 'optimizer1':
            optimizer1.load_state_dict(checkpoint[key])
        elif key == 'optimizer2':
            optimizer2.load_state_dict(checkpoint[key])
        print(f"Check Point Loading: {key} is LOADED")

def train(epoch,net1,net2,optimizer,labeled_trainloader,mode='PL',num_iter = 400):
    net1.train()
    net2.eval() 


    scaler=GradScaler()

    labeled_train_iter = iter(labeled_trainloader)      
    I=2


    while True:
        try:
            inputs, labels_x, w_x = next(labeled_train_iter)
            inputs_w,inputs_s =inputs
        except:
            if I==1:
                stats_log.write(f'\n')
                stats_log.flush()
                return 
            else:
                labeled_train_iter = iter(labeled_trainloader)      
                inputs, labels_x, w_x = next(labeled_train_iter)
                inputs_w,inputs_s =inputs
                I=I-1

        w_x = w_x.view(-1,1).type(torch.FloatTensor) 
        w_x = w_x.cuda()

        batch_size = inputs_w.size(0)

        one_hot_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1,1), 1)
        one_hot_x = one_hot_x.cuda()
    

        inputs_w,inputs_s, labels_x = inputs_w.cuda(),inputs_s.cuda(), labels_x.cuda()

        with torch.no_grad():

            outputs_1_w=net1(inputs_w)
            outputs_2_w=net2(inputs_w)

        with autocast():
            outputs_s=net1(inputs_s)

            probs=torch.softmax(outputs_1_w,dim=-1)+torch.softmax(outputs_2_w,dim=-1)
            probs=probs/2
            probs_T = probs**args.T
            probs_T = probs_T  / probs_T.sum(dim=1, keepdim=True)
            probs_T = probs_T.detach()
            probs_T_x=probs_T[:batch_size]
            targets_x=one_hot_x*w_x+(1-w_x)*probs_T_x
            Lx=softCE(outputs_s, targets_x)

            prior = torch.ones(args.num_class)/args.num_class
            prior = prior.cuda() 
            pred_mean = torch.softmax(outputs_s, dim=1).mean(0) 
            penalty = torch.sum(prior*torch.log(prior/pred_mean))
            loss = args.lambda_l*Lx+args.lambda_p*penalty

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


def warmup(epoch,net,eval_models,ema,optimizer,dataloader,updataema=False):
    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):         
        inputs, labels = inputs.cuda(), labels.cuda() 
        optimizer.zero_grad()
        outputs = net(inputs)                
        loss = CEloss(outputs, labels)        
        if args.noise_mode=='asym':  
            penalty = conf_penalty(outputs)
            L = loss + penalty     
        elif args.noise_mode=='sym':   
            L = loss
        L.backward()  
        optimizer.step() 


@torch.no_grad()
def test(epoch,net1,net2,class_wise=False):
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    all_target=[]
    all_preds=[]
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1 = net1(inputs)
            outputs2 = net2(inputs)              
            outputs = outputs1+outputs2
            _, predicted = torch.max(outputs, 1)            
            all_target.append(targets)
            all_preds.append(predicted)
                       
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()                    
    acc = 100.*correct/total
    #print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" %(epoch,acc))    
    print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" %(epoch,acc))  
    if class_wise:
        class_wise_accs=[]
        all_preds=torch.cat(all_preds,dim=0).cpu().numpy()
        all_target=torch.cat(all_target,dim=0).cpu().numpy()
        # print(np.mean(all_preds==all_target))
        for c in range(args.num_class):
            class_wise_accs.append(np.mean(all_preds[all_target==c]==c))
        print(min(class_wise_accs),class_wise_accs)
        from sklearn.metrics import confusion_matrix
        print(confusion_matrix(all_target,all_preds))
    stats_log.write('Epoch:%d    Accuracy:%.2f\n'%(epoch,acc))
    stats_log.flush()  
    return acc
    

@torch.no_grad()
def eval_train(model,all_loss,noise,gmm,weight=None):
    model.eval()
    losses = torch.zeros(num_samples)    

    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda() 
            outputs = model(inputs) 

            if weight is None:
                loss = CE(outputs, targets)  
            else:
                weight=weight.view(-1)
                weight=torch.index_select(weight,0,targets).view(-1)
                loss = CE(outputs, targets)
                loss = loss/weight
            for b in range(inputs.size(0)):
                losses[index[b]]=loss[b]
                #predictions[index[b]]=outputs[b]

    losses=losses.cpu().numpy()
    losses = (losses-losses.min())/(losses.max()-losses.min())      
    all_loss.append(losses)
    #print(losses.tolist())

    input_loss = losses.reshape(-1,1)

    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss) 
    prob = prob[:,gmm.means_.argmin()]
    return prob,all_loss#,predictions
    
@torch.no_grad()
def eval_train_c(model,all_loss,noise,gmm):
    model.eval()
    losses = torch.zeros(num_samples)    
    predictions=[]
    labels = torch.zeros(num_samples)    
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda() 
            outputs = model(inputs) 
            loss = CE(outputs, targets)  
            _, predicted = torch.max(outputs, 1)            
            predictions.append(predicted)
            for b in range(targets.size(0)):
                losses[index[b]]=loss[b]         
                labels[index[b]]=targets[b]         

    predictions=torch.cat(predictions,dim=0).cpu().numpy()
    labels=labels.long().cpu().numpy()
    losses=losses.cpu().numpy()
    losses = (losses-losses.min())/(losses.max()-losses.min())    

    all_loss.append(losses)

    input_loss = losses.reshape(-1,1)
    input_loss = losses.reshape(-1,1)
    all_prob=np.zeros(num_samples)+0.5
    gmm_c = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=1e-4)
    for c in range(args.num_class):
        mask=predictions==c
        if mask.sum()<=1:
            continue
        gmm_c.fit(input_loss[mask])
        prob = gmm_c.predict_proba(input_loss[mask])
        all_prob[mask] = prob[:,gmm_c.means_.argmin()]

    return all_prob,all_loss

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

def create_model():
    model = ResNet18(num_classes=args.num_class)
    model = model.cuda()
    return model

def create_model_selfsup(net='resnet18', dataset='cifar10', num_classes=10,drop=0):
    chekpoint = torch.load('pretrained/ckpt_{}_{}.pth'.format(dataset, net),map_location=f'cuda:{args.gpuid}')
    sd = {}
    for ke in chekpoint['model']:
        nk = ke.replace('module.', '')
        sd[nk] = chekpoint['model'][ke]
    model = SupCEResNet(net, num_classes=num_classes)
    
    #model = ResNet18(num_classes=args.num_class)
    model.load_state_dict(sd, strict=False)
    model = model.cuda()
    return model


stats_log=open('./checkpoint/%s_%s_%.1f_%s'%(args.save_name,args.dataset,args.r,args.noise_mode)+'_stats.txt','w') 
#test_log=open('./checkpoint/%s_%s_%.1f_%s'%(args.save_name,args.dataset,args.r,args.noise_mode)+'_acc.txt','w')     

# if args.dataset=='cifar10':
#      warm_up = 10
# elif args.dataset=='cifar100':
#      warm_up = 30

loader = dataloader.cifar_dataloader(args.dataset,r=args.r,noise_mode=args.noise_mode,batch_size=args.batch_size,num_workers=5,root_dir=args.data_path,log=stats_log,noise_file='%s/%.1f_%s.json'%(args.data_path,args.r,args.noise_mode),imbalance=args.imbalance)
print('hard label',args.hard_label)
print('| Building net')
#net1 = create_model_selfsup()
#net2 = create_model_selfsup()

net1 = create_model()
net2 = create_model()
test_loader = loader.run('test')
acc=test(-1,net1,net2)    
#eval_models = []
#for i in range(3):
#     eval_models.append(create_model())
#     for param_k in eval_models[-1].parameters():
#         param_k.requires_grad = False
#emas=[0.999,0.998,0.9995]


cudnn.benchmark = True
optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
gmm4loss = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=1e-4)

CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
LSloss = LabelSmoothingLoss(10)
softCE=SoftCELoss()
#CEloss = SCELoss(1,0.1)
if args.noise_mode=='asym':
    conf_penalty = NegEntropy()
best_acc=0

#all_loss=[[] for i in range(len(eval_models))]
# save the history of losses from two networks
all_loss_1=[]
all_loss_2=[]

if len(args.resume)!=0:
    load_model(args.resume,net1,net2,optimizer1,optimizer2)
    test_loader = loader.run('test')
    acc=test(-1,net1,net2,class_wise=True)    
    exit()
    eval_loader = loader.run('eval_train')     

    print('ori')
    prob1,all_loss_1=eval_train(net1,all_loss_1,None,gmm4loss)
    from sklearn import metrics
    with open(f'cifar10_{args.r}r_LT_label',) as f:
        clean=eval(f.readline())
        noise=eval(f.readline())
        clean=np.asarray(clean)
        noise=np.asarray(noise)
        clean=clean==noise
    fpr, tpr, thresholds = metrics.roc_curve(clean, prob1)
    print(metrics.auc(fpr, tpr))

    print('classwise')
    prob2,_ =eval_train_c(net2,all_loss_2,None,gmm4loss)            
    from sklearn import metrics
    with open(f'cifar{num_class}_{args.r}r_LT_label',) as f:
        clean=eval(f.readline())
        noise=eval(f.readline())
        clean=np.asarray(clean)
        noise=np.asarray(noise)
        clean=clean==noise
    fpr, tpr, thresholds = metrics.roc_curve(clean, prob2)
    print(metrics.auc(fpr, tpr))
    exit()
    # print(prob2.tolist())

    p_epoch=150
else:
    p_epoch=0

for epoch in range(p_epoch,args.warm_up):    
    test_loader = loader.run('test')
    for param_group in optimizer1.param_groups:
        param_group['lr'] = 0.02
    for param_group in optimizer2.param_groups:
        param_group['lr'] = 0.02
    warmup_trainloader = loader.run('warmup')
    print('Warmup Net1')
    warmup(epoch,net1,None,None,optimizer1,warmup_trainloader)
    print('Warmup Net2')
    warmup(epoch,net2,None,None,optimizer2,warmup_trainloader)

    acc=test(epoch,net1,net2)
    
save_model(f"{args.save_name}_{args.dataset}_{args.r}_{args.noise_mode}_latest","./checkpoint/",net1,net2,optimizer1,optimizer2)
#fp=open('lc_predictions_0.9','wb')

for epoch in range(p_epoch,args.num_epochs+1):     
    #start_time = time.time()
    lr=args.lr
    if args.num_epochs-epoch<100:
        lr /= 10 
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr      
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr      
        
    test_loader = loader.run('test')
    eval_loader = loader.run('eval_train')     
    if args.conf_mode=='C':
        prob1,all_loss_1=eval_train_c(net1,all_loss_1,None,gmm4loss)
        prob2,all_loss_2=eval_train_c(net2,all_loss_2,None,gmm4loss)            
    elif args.conf_mode=='O':
        prob1,all_loss_1=eval_train(net1,all_loss_1,None,gmm4loss)
        prob2,all_loss_2=eval_train(net2,all_loss_2,None,gmm4loss)            
    elif args.conf_mode=='W':
        prob1,all_loss_1=eval_train(net1,all_loss_1,None,gmm4loss,weight=class_weight)
        prob2,all_loss_2=eval_train(net2,all_loss_2,None,gmm4loss,weight=class_weight)            
    else:
        raise Exception

    pred1 = (prob1 > args.p_threshold)        
    pred2 = (prob2 > args.p_threshold)        


    stats_log.write(f'selected {pred1.sum()/pred1.shape[0]} \n')
    stats_log.write(f'selected {pred2.sum()/pred2.shape[0]} \n')
    stats_log.flush()

    print('Train Net')
    labeled_trainloader = loader.run('train',pred1,prob1) # co-divide
    train(epoch,net2,net1,optimizer2,labeled_trainloader,num_iter=args.epochs) # train net2            

    print('\nTrain Net2')
    labeled_trainloader = loader.run('train',pred2,prob2) # co-divide
    train(epoch,net1,net2,optimizer1,labeled_trainloader,num_iter=args.epochs) # train net1  


    acc=test(epoch,net1,net2)
    if acc>best_acc:
        best_acc=acc
        save_model(f"{args.save_name}_{args.dataset}_{args.r}_{args.noise_mode}_best","./checkpoint/",net1,net2,optimizer1,optimizer2)
    
    save_model(f"{args.save_name}_{args.dataset}_{args.r}_{args.noise_mode}_latest","./checkpoint/",net1,net2,optimizer1,optimizer2)
