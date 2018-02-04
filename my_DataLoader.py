import os
import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import cv2 as cv2
import torchvision.transforms as transforms
lab = {'HTC-1-M7':0, 'iPhone-4s':1, 'iPhone-6':2, 'LG-Nexus-5x':3, 'Motorola-Droid-Maxx':4, 'Motorola-Nexus-6':5, 'Motorola-X':6, 'Samsung-Galaxy-Note3':7, 'Samsung-Galaxy-S4':8, 'Sony-NEX-7':9}



def default_loader(path):
    return Image.open(path)
class myImageFloder(data.Dataset):
    def __init__(self, root,train , label, transform, target_transform=None, loader=default_loader):
        ft = train
        fh = label
        c=0
        imgs=[]
        class_names=[]
        for line,name in  zip(ft,fh):
            if c==0:
                class_names=[n.strip() for n in line.rstrip().split('	')]
            else:
                cls = line.split()
                fn = cls.pop(0)
                #print(os.path.join(root,name,fn))
                #print(os.path.isfile(os.path.join(root,name,fn)))
                if os.path.isfile(os.path.join(root,name,fn)):
                    #imgs.append((fn, tuple([float(v) for v in cls])))
                    imgs.append((name,os.path.join(root,name,fn)))
                #print((name,os.path.join(root,name,fn)))
            c=c+1
        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        label,fn = self.imgs[index]
        img = self.loader(fn)
        #img = np.transpose(img, (1,0,2))
        #img = np.array(img).astype('uint8')
        #print(label)
        if label in lab:
            label2 = np.zeros(1,dtype=int)
            #label2[lab[label]] = 1
            label2 = int(lab[label])
            #print(label2)
        if self.transform is not None:
            #print(img.shape,img)
            img = self.transform(img)
            #print(img.shape,img)
        return img, label2

    def __len__(self):
        return len(self.imgs)
