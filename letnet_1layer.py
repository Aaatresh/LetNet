import torch as tor
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.utils.data
import cv2
import os
from scipy.fftpack import dct
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F

def preprocess(img):
    
    scaler=StandardScaler()
    img=scaler.fit_transform(img)
    
    return img

def DoG(u,k,tau):
    
    DoG=u*tor.exp(-((k-1)*tor.pow(u,2))/(2*tau**2))
    
    return DoG
    
def rho(u,param):
    
    c=param[0]
    tau=param[1]
    rho=0
    
    K=len(c)
    
    for k in range(1,K+1):
        rho=rho+c[k-1,0]*DoG(u,k,tau)
    
    return rho
    

def gen_dct(size):

    D=tor.from_numpy(dct(np.eye(size),norm="ortho",axis=0))   
    D=D.view(1,1,D.size(0),-1).type(tor.FloatTensor)
    
    return D
 
def get_psf(psf_path,width,height):
    
    psf=cv2.resize(cv2.imread(psf_path,0),(width,height))
    psf=tor.from_numpy(psf).type(tor.FloatTensor)
    psf=psf.view(1,1,psf.size(0),psf.size(1))
    
    return psf
 
def conv(h,x):

    final_conv_dim=(512,512)
    x_dim=(x.size(2),x.size(3))
    h_dim=(h.size(2),h.size(3))
    crop_dim=x_dim

    padding=(final_conv_dim[0]-(x_dim[0]-h_dim[0]+1),final_conv_dim[1]-(x_dim[1]-h_dim[1]+1))

    x_pad=F.pad(x,(padding[0]//2,padding[0]//2+1,padding[1]//2,padding[1]//2+1))
    y=F.conv2d(x_pad,h.flip(2,3),padding=0)
    
    starti=(final_conv_dim[0]-crop_dim[0])//2
    endi=crop_dim[0]+starti
    startj=(final_conv_dim[1]-crop_dim[1])//2
    endj=crop_dim[1]+startj

    y=y[:,:,starti:endi,startj:endj]
    #y=y[:,:,181:331,156:356]
    
    return y

class t_layer(nn.Module):
    
    def __init__(self,learning_rate):
        
        super().__init__()
        
        self.learning_rate = learning_rate
        
    def forward(self,y,xt,D,h,param):
        
        h_flip=h.flip(2,3)
        a=conv(h,xt)-y
        b=xt-self.learning_rate*conv(h_flip,a)
        u=tor.matmul(D,tor.matmul(b,D.transpose(2,3)))
        u_=rho(u,param)
        xt1=tor.matmul(D.transpose(2,3),tor.matmul(u_,D))
        
        return xt1
        

class LetNet(nn.Module):

    def __init__(self,c,tau,psf,learning_rate):
        
        super().__init__()
        
        self.c=nn.Parameter(c,requires_grad=True)
        self.tau=nn.Parameter(tau,requires_grad=True)
        self.param=[self.c,self.tau]
        
        self.D=gen_dct(psf.size(2))
        self.h=psf
        self.learning_rate = learning_rate
        
        self.layer1=t_layer(learning_rate)
        
    def forward(self,y):
    
        x0=tor.randn(y.shape)
        out=self.layer1(y,x0,self.D,self.h,self.param)
        
        return out
        
        
class dataset(torch.utils.data.Dataset):

    def __init__(self,gt_dir,sensor_dir,datasize=0,transform=None):
        
        super().__init__()
        
        gts=os.listdir(gt_dir)
        gts=[os.path.join(gt_dir,x) for x in gts]
        
        sensor=os.listdir(sensor_dir)
        sensor=[os.path.join(sensor_dir,x) for x in sensor]
        
        if(datasize<0 or datasize>len(gts)):
            assert("datasize should be >=0 or <= max data size")
        elif(datasize==0):
            datasize=len(gts)
            
        self.gts=gts
        self.sensor=sensor
        self.datasize=datasize
        self.transform=transform
        
    def __len__(self):
        
        return self.datasize
        
    def __getitem__(self,idx):
        
        gt_addr=self.gts[idx]
        sensor_addr=self.sensor[idx]
        
        X=np.float32(cv2.imread(gt_addr,0))
        Y=np.float32(cv2.imread(sensor_addr,0))
        
        if(self.transform is not None):
            X=self.transform(X)
            Y=self.transform(Y)
        
        X=tor.from_numpy(X).type(tor.FloatTensor).unsqueeze(0)
        Y=tor.from_numpy(Y).type(tor.FloatTensor).unsqueeze(0)
            
        return Y,X
        
        
if __name__=="__main__":

    gts_dir="./gts_256_256/"
    sensor_dir="./sensor_readings_256_256/"
    psf_path="./psf/psf_sample.tif"
    
    width=256           ## sensor width
    height=256          ## sensor height
    
    psf=get_psf(psf_path,width,height)
    
    trainset=dataset(gts_dir,sensor_dir,transform=preprocess)
    trainloader=torch.utils.data.DataLoader(trainset,batch_size=2,shuffle=True)
    
    c=tor.randn(10,1)*0.001
    tau=tor.randn(1)*0.001
    
    ## set the learning rate for each layer
    learning_rate = 1e-5
    
    clf=LetNet(c,tau,psf,learning_rate)
    
    criterion=nn.MSELoss()
    optimizer=tor.optim.Adam(clf.parameters(),lr=1e-3)
    
    losses=[]
    epochs=10
    
    
    for epoch in range(epochs):
    
        l=[]
        for batch_idx,(Y,X) in enumerate(trainloader):
            
            X_=clf(Y)
            
            """
            print("X.shape: ",X.shape)
            print("Y.shape: ",Y.shape)
            print("X_.shape: ",X_.shape)
            input("")
            """
            
            loss=criterion(X_,X)
            loss.backward()
            optimizer.step()
            
            l.append(loss.item())
        
            print("Batch Index: %d \t Loss in batch: %0.4f" % (batch_idx,l[-1]))
            
        losses.append(np.mean(l))       
        
        print("Epoch: ",epoch,"\tCost: ",losses[-1])
        
      
