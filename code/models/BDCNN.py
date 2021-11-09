import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

'''
#BDCNN network
#size8 conv2:7200,512  fc1:4608
#size10 conv2:4608,512  fc1:8192
#size20  conv2:1152,256 fc1:12544
#size30  conv2:512,128 fc1:12800
#size40  conv2:288,32 fc1:6272
'''
class BDCNN(nn.Module):
    def __init__(self,size):
        super(BDCNN, self).__init__()
        self.feature_num=128
        self.size=size
        self.net_dict={
            'conv2_0':{8:7200,10:4608,20:1152,30:512,40:288},
            'conv2_1':{8:512,10:512,20:256,30:128,40:32},
            'fc1_0':{8:4608,10:8192,20:12544,30:12800,40:6272},
        }
        self.pool=nn.MaxPool2d(3,3,1)
        self.conv1=nn.Conv2d(3,8,3,1,1)
        self.conv2=nn.Conv2d(self.net_dict['conv2_0'][self.size],self.net_dict['conv2_1'][self.size],1)
        self.fc1 = nn.Linear(self.net_dict['fc1_0'][self.size],1024)
        self.fc2=nn.Linear(1024,128)
        
        self.cbam=CBAM(512)
        
    
    def forward(self,input1,input2,input3,input4):
        blocks1=[]
        n=120//self.size
        for i in range(n):
            for j in range(n):
                blocks1.append(input1[:,:,self.size*i:self.size*(i+1),self.size*j:self.size*(j+1)])
        convs1=[]
        for i in range(n**2):
            convs1.append(self.pool(F.relu(self.conv1(blocks1[i]))))
        x1=torch.cat((convs1[0],convs1[1]),dim=1)
        for i in range(2,n**2):
            x1=torch.cat((x1,convs1[i]),dim=1)
        
        blocks2=[]
        for i in range(n):
            for j in range(n):
                blocks2.append(input2[:,:,self.size*i:self.size*(i+1),self.size*j:self.size*(j+1)])
        convs2=[]
        for i in range(n**2):
            convs2.append(self.pool(F.relu(self.conv1(blocks2[i]))))
        x2=torch.cat((convs2[0],convs2[1]),dim=1)
        for i in range(2,n**2):
            x2=torch.cat((x2,convs2[i]),dim=1)
        
        blocks3=[]
        for i in range(n):
            for j in range(n):
                blocks3.append(input3[:,:,self.size*i:self.size*(i+1),self.size*j:self.size*(j+1)])
        convs3=[]
        for i in range(n**2):
            convs3.append(self.pool(F.relu(self.conv1(blocks3[i]))))
        x3=torch.cat((convs3[0],convs3[1]),dim=1)
        for i in range(2,n**2):
            x3=torch.cat((x3,convs3[i]),dim=1)
            
        blocks4=[]
        for i in range(n):
            for j in range(n):
                blocks4.append(input4[:,:,self.size*i:self.size*(i+1),self.size*j:self.size*(j+1)])
        convs4=[]
        for i in range(n**2):
            convs4.append(self.pool(F.relu(self.conv1(blocks4[i]))))
        x4=torch.cat((convs4[0],convs4[1]),dim=1)
        for i in range(2,n**2):
            x4=torch.cat((x4,convs4[i]),dim=1)
        
        x=torch.cat((x1,x2,x3,x4),dim=1)
        x=F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

