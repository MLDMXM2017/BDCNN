import argparse
import os
import shutil
import time
import errno
import math
import cv2
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torchvision import transforms
from models.BDCNN import BDCNN

from getdata import GetData_raw
from ISDA_im import EstimatorCV, ISDALoss


import numpy as np
import pandas as pd
import pickle
        
def to_pkl(content,pkl_file):
    """
    save data in pkl format
    @param content:data
    @param pkl_file: save path
    """
    with open(pkl_file, 'wb') as f:
        pickle.dump(content, f)
        f.close()
    

     
# Configurations adopted for training deep networks.
# (specialized for each type of models)
training_configurations = {
        'epochs': 100,
        'batch_size': 128,
        'initial_learning_rate': 0.01,
        'changing_lr': [80, 120],
        'lr_decay_rate': 0.1,
        'momentum': 0.9,
        'nesterov': True,
        'weight_decay': 1e-4,
}

def train_fold(fold,recod_path,size):
    record_file = record_path + '/training_process.txt'
    accuracy_file = record_path + '/accuracy_epoch.txt'
    conf_file=record_path+'/conf_matrix.csv'
    check_point = os.path.join(record_path, args.checkpoint)

    global best_prec1
    best_prec1 = 0

    global val_acc
    val_acc = []

    global class_num

    class_num = 5


    kwargs = {'num_workers': 0, 'pin_memory': True}
    #unify or separately transform inputs
    if args.is_unify_augment==True:
        trainset=GetData_unify('../loso_5type/'+fold+'/train',True,my_transform1)
        valset = GetData_unify('../loso_5type/'+fold+'/test',False,my_transform1)
    else:
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        transform_train = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        transform_test = transforms.Compose([
                transforms.ToTensor(),
                normalize
            ])
        dataset = GetData_raw('../loso_5type/'+fold+'/train',transform_train,5)
        testset = GetData_raw('../loso_5type/'+fold+'/test',transform_test,5)
    
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    trainset, valset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader =torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True,**kwargs)
    val_loader =torch.utils.data.DataLoader(valset, batch_size=128, shuffle=True,**kwargs)
    test_loader =torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True,**kwargs)
    
    # create model
    if args.model=='BDCNN':
        model=BDCNN(size)
    print(model)
    
    if not os.path.isdir(check_point):
        mkdir_p(check_point)
           
    if args.is_attention_feature==False:
        fc = Full_layer(int(model.feature_num), class_num)
    else:
        fc = Full_layer_attention(int(model.feature_num), class_num)

    print('Number of final features: {}'.format(
        int(model.feature_num))
    )

    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])
        + sum([p.data.nelement() for p in fc.parameters()])
    ))
    
    cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    isda_criterion = ISDALoss(int(model.feature_num), class_num,[0.21, 0.21,0.16,0.21,0.21]).cuda()
    ce_criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD([{'params': model.parameters()},
                                {'params': fc.parameters()}],
                                lr=training_configurations['initial_learning_rate'],
                                momentum=training_configurations['momentum'],
                                nesterov=training_configurations['nesterov'],
                                weight_decay=training_configurations['weight_decay'])

    model = torch.nn.DataParallel(model).cuda()
    fc = nn.DataParallel(fc).cuda()
    
    best_test_prec=0
    best_conf=[]
    for epoch in range(training_configurations['epochs']):

        adjust_learning_rate(optimizer, epoch + 1)

        # train for one epoch
        features,targets=train(train_loader, model, fc, isda_criterion, optimizer, epoch,record_file)
        
       
        # evaluate on validation set
        prec1,conf1,val_fea,val_tar = validate(val_loader, model, fc, ce_criterion, epoch,record_file)

        # remember best prec@1 and save checkpoint
        is_best = prec1 >= best_prec1
        best_prec1 = max(prec1, best_prec1)
        if is_best==True:
            test_prec,test_conf= test(test_loader,model, fc, ce_criterion, epoch,record_file)
            best_test_prec=test_prec
            best_test_conf=test_conf

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'fc': fc.state_dict(),
            'best_acc': best_prec1,
            'optimizer': optimizer.state_dict(),
            'isda_criterion': isda_criterion,
            'val_acc': val_acc,

        }, is_best, checkpoint=check_point)
        print('Best accuracy: ', best_test_prec)
        np.savetxt(accuracy_file, np.array(val_acc))
        
    
    last_conf=conf1
    print('Best accuracy: ', best_test_prec)

    m=pd.DataFrame(data=best_test_conf)
    m.to_csv(conf_file)
    np.savetxt(accuracy_file, np.array(val_acc))
    

def train(train_loader, model, fc, criterion, optimizer, epoch,record_file):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    train_batches_num = len(train_loader)

    ratio = args.lambda_0 * (epoch / (training_configurations['epochs']))
    # switch to train mode
    model.train()
    fc.train()
    
    end = time.time()
    for i, (x,x1,x2,x3,x4,target) in enumerate(train_loader):
        target = target.cuda()
        x=x.cuda()
        x2 = x2.cuda()
        x3 =x3.cuda()
        x4=x4.cuda()
        #print(x)
        input_var1 = torch.autograd.Variable(x2)
        input_var2 = torch.autograd.Variable(x3)
        input_var3 = torch.autograd.Variable(x)
        input_var4 = torch.autograd.Variable(x4)
        target_var = torch.autograd.Variable(target)

        # compute output
        loss, output,features = criterion(model, fc, input_var1, input_var2,input_var3,input_var4,target_var, ratio)
        if i==0:
            all_features=features.cpu().detach().numpy()
            all_targets=target.cpu().detach().numpy()
        else:
            all_features=np.concatenate((all_features,features.cpu().detach().numpy()),axis=0)
            all_targets=np.concatenate((all_targets,target.cpu().detach().numpy()),axis=0)
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), x2.size(0))
        top1.update(prec1.item(), x2.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()


        if (i+1) % args.print_freq == 0:
            # print(discriminate_weights)
            fd = open(record_file, 'a+')
            string = ('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
                      'Loss {loss.value:.4f} ({loss.ave:.4f})\t'
                      'Prec@1 {top1.value:.3f} ({top1.ave:.3f})\t'.format(
                       epoch, i+1, train_batches_num, batch_time=batch_time,
                       loss=losses, top1=top1))

            print(string)
            # print(weights)
            fd.write(string + '\n')
            fd.close()
 
    return all_features,all_targets

def validate(val_loader, model, fc, criterion, epoch,record_file):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    train_batches_num = len(val_loader)

    # switch to evaluate model
    model.eval()
    fc.eval()

    end = time.time()
    conf_matrix = torch.zeros(5,5)
    for i, (x,x1,x2,x3,x4,target) in enumerate(val_loader):
        target = target.cuda()
        x=x.cuda()
        x2 = x2.cuda()
        x3=x3.cuda()
        x4=x4.cuda()
        #print(input.shape)
        input_var1 = torch.autograd.Variable(x2)
        input_var2 = torch.autograd.Variable(x3)
        input_var3 = torch.autograd.Variable(x)
        input_var4 = torch.autograd.Variable(x4)
        target_var = torch.autograd.Variable(target)

        # compute output
        with torch.no_grad():
            features = model(input_var1,input_var2,input_var3,input_var4)
            if i==0:
                all_features=features.cpu().detach().numpy()
                all_targets=target.cpu().detach().numpy()
            else:
                all_features=np.concatenate((all_features,features.cpu().detach().numpy()),axis=0)
                all_targets=np.concatenate((all_targets,target.cpu().detach().numpy()),axis=0)
            output = fc(features)
        
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), x2.size(0))
        top1.update(prec1.item(), x2.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        prediction = torch.max(output, 1)[1]

        conf_matrix = confusion_matrix(prediction, labels=target, conf_matrix=conf_matrix)

        if (i+1) % args.print_freq == 0:
            fd = open(record_file, 'a+')
            string = ('Test: [{0}][{1}/{2}]\t'
                      'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
                      'Loss {loss.value:.4f} ({loss.ave:.4f})\t'
                      'Prec@1 {top1.value:.3f} ({top1.ave:.3f})\t'.format(
                       epoch, (i+1), train_batches_num, batch_time=batch_time,
                       loss=losses, top1=top1))
            print(string)
            fd.write(string + '\n')
            fd.close()

    fd = open(record_file, 'a+')
    string = ('Test: [{0}][{1}/{2}]\t'
              'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
              'Loss {loss.value:.4f} ({loss.ave:.4f})\t'
              'Prec@1 {top1.value:.3f} ({top1.ave:.3f})\t'.format(
        epoch, (i + 1), train_batches_num, batch_time=batch_time,
        loss=losses, top1=top1))
    print(string)
    
    fd.write(string + '\n')
    fd.close()
    val_acc.append(top1.ave)
    
    return top1.ave,conf_matrix.numpy(),all_features,all_targets
    
def test(test_loader,model, fc, criterion, epoch,record_file):
    """Perform test on the test set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    train_batches_num = len(test_loader)

    # switch to evaluate mode
    model.eval()
    fc.eval()

    end = time.time()
    conf_matrix = torch.zeros(5, 5)
    for i, (x,x1,x2,x3,x4,target) in enumerate(test_loader):
        target = target.cuda()
        x=x.cuda()
        x2 = x2.cuda()
        x3=x3.cuda()
        x4=x4.cuda()
        #print(input.shape)
        input_var1 = torch.autograd.Variable(x2)
        input_var2 = torch.autograd.Variable(x3)
        input_var3 = torch.autograd.Variable(x)
        input_var4 = torch.autograd.Variable(x4)
        target_var = torch.autograd.Variable(target)

        # compute output
        with torch.no_grad():
            features = model(input_var1,input_var2,input_var3,input_var4)
            if i==0:
                all_features=features.cpu().detach().numpy()
                all_targets=target.cpu().detach().numpy()
            else:
                all_features=np.concatenate((all_features,features.cpu().detach().numpy()),axis=0)
                all_targets=np.concatenate((all_targets,target.cpu().detach().numpy()),axis=0)
            output = fc(features)
        
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), x2.size(0))
        top1.update(prec1.item(), x2.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        prediction = torch.max(output, 1)[1]

        conf_matrix = confusion_matrix(prediction, labels=target, conf_matrix=conf_matrix)

        if (i+1) % args.print_freq == 0:
            fd = open(record_file, 'a+')
            string = ('Test: [{0}][{1}/{2}]\t'
                      'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
                      'Loss {loss.value:.4f} ({loss.ave:.4f})\t'
                      'Prec@1 {top1.value:.3f} ({top1.ave:.3f})\t'.format(
                       epoch, (i+1), train_batches_num, batch_time=batch_time,
                       loss=losses, top1=top1))
            print(string)
            fd.write(string + '\n')
            fd.close()

    fd = open(record_file, 'a+')
    string = ('Test: [{0}][{1}/{2}]\t'
              'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
              'Loss {loss.value:.4f} ({loss.ave:.4f})\t'
              'Prec@1 {top1.value:.3f} ({top1.ave:.3f})\t'.format(
        epoch, (i + 1), train_batches_num, batch_time=batch_time,
        loss=losses, top1=top1))
    print(string)
    
    fd.write(string + '\n')
    fd.close()
    val_acc.append(top1.ave)
    
    return top1.ave,conf_matrix.numpy()
    
def confusion_matrix(preds, labels, conf_matrix):
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix

class Full_layer(torch.nn.Module):
    '''explicitly define the full connected layer'''

    def __init__(self, feature_num, class_num):
        super(Full_layer, self).__init__()
        self.class_num = class_num
        self.fc = nn.Linear(feature_num, class_num)

    def forward(self, x):
        x = self.fc(x)
        return x
        
def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.ave = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.value = val
        self.sum += val * n
        self.count += n
        self.ave = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate"""
    if not args.cos_lr:
        if epoch in training_configurations['changing_lr']:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= training_configurations['lr_decay_rate']

    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.5 * training_configurations['initial_learning_rate']\
                                * (1 + math.cos(math.pi * epoch / training_configurations['epochs']))


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ME recognition using ISDA')

    parser.add_argument('--model', default='BDCNN', type=str,
                    help='deep networks to be trained')

    parser.add_argument('--print-freq', '-p', default=2, type=int,
                    help='print frequency (default: 2)')

    parser.add_argument('--is_split',default=False,
                    help='whether to use spliting data (default: False)')

    parser.add_argument('--is_augment',default=False,
                    help='whether to use data augment(default: False)')

    parser.add_argument('--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')

    parser.add_argument('--lambda_0', default=0.5, type=float,
                    help='hyper-patameter_\lambda for ISDA')

    parser.add_argument('--save_path', default='result/BDCNN', type=float,
                    help='the save path of results')

    # Cosine learning rate
    parser.add_argument('--cos_lr', dest='cos_lr', action='store_true',
                    help='whether to use cosine learning rate')
    parser.set_defaults(cos_lr=False)
    args = parser.parse_args()
        
        
    path=' '  #the path your data stores
    folds=os.listdir(path)
    
    #LOSO
    for fold in folds:
    	record_path = args.save_path+'/'+fold
        if fold!='total' and not os.path.exists(record_path):
                print(fold+' starts training!')
                train_fold(fold,record_path,args.size)
