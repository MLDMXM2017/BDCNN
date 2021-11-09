#get data and data augmentation

from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from PIL import Image
import os
import cv2
from sklearn.model_selection import StratifiedKFold

'''
get data and then split data randomly
'''
class GetData_raw(Dataset):
    #separately transform to four inputs
    def __init__(self,path1,transform,count,is_reshape=True):
        super(GetData_raw,self).__init__()
        self.path = path1
        self.transform=transform
        self.is_reshape=is_reshape
        self.count=count
        self.dataset = []
        self.dataset.extend(open(os.path.join(self.path,'label.txt')).readlines())
 
    def __getitem__(self, index): 
        str1 = self.dataset[index].strip()
        imgdata=[]
        for i in range(self.count):
            imgpath=os.path.join(self.path,str1.split(',')[i])
            im = cv2.imread(imgpath)
            if self.is_reshape==True:
                im=cv2.resize(im,(120,120))
            imgdata.append(self.transform(im))
        label=int(str1.split(',')[self.count])   
        return [imgdata[i] for i in range(self.count)]+[label]
 
    def __len__(self):
        return len(self.dataset)
 
'''
get data by spliting data
'''
class GetData_split(Dataset):
    def __init__(self,path1,txt_name,transform,count,is_reshape=True):
        super(GetData_split,self).__init__()
        self.path = path1
        self.transform=transform
        self.is_reshape=is_reshape
        self.count=count
        self.dataset = []
        self.dataset.extend(open(os.path.join(self.path,txt_name)).readlines())
 
    def __getitem__(self, index): 
        str1 = self.dataset[index].strip()
        imgdata=[]
        for i in range(self.count):
            imgpath=os.path.join(self.path,str1.split(',')[i])
            im = cv2.imread(imgpath)
            if self.is_reshape==True:
                im=cv2.resize(im,(120,120))
            imgdata.append(self.transform(im))
        label=int(str1.split(',')[self.count])   
        return [imgdata[i] for i in range(self.count)]+[label]
 
    def __len__(self):
        return len(self.dataset)