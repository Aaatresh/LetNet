"""

    Script to apply LET approach to image reconstruction on diffusercam lensless cameras.
    The parameters C and tau, (the linear coeffecients and standard deviation respectively) are fixed across all layers. These parameters
    can be made a parameter for each layer by extending this program.
    
    This based on the paper: "Bayesian Deep Deconvolutional Neural Networks"

"""

########################################################################
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
########################################################################

def preprocess(img):

    """
        Function to preprocess the image by zero-centering the mean and normalizing it with it's standard deviation
        
        Arguments: 
        
            img: The image to be preprocessed
            
        Returns: 
        
            image after preprocessing
    """
    
    scaler=StandardScaler()                 ## scaler object to perform preprocessing
    img=scaler.fit_transform(img)           ## zero-center and normalize
    
    return img

def DoG(u,k,tau):
    
    """
        Function to calculate the kth Derivative of Gaussian (DoG) 

        Arguments:
        
            u: input element
            k: kth derivative of gaussian
            tau: standard deviation

        Returns: 
        
            The kth Derivative of Gaussian            
    """
    
    DoG=u*tor.exp(-((k-1)*tor.pow(u,2))/(2*tau**2))
    
    return DoG
    
def rho(u,param):
    
    """
        Function to find the linear combination of K Derivatives of Gaussian
        
        Arguments: 
            
            u: input element
            param: list containing the 'K' linear coeffecients and the standard deviation
    
        Returns: 
        
            The linear combination of K Derivatives of Gaussian
    
    """
    
    c=param[0]
    tau=param[1]
    rho=0
    
    K=len(c)
    
    for k in range(1,K+1):              ## Sum up the K Derivatives of Gaussian
        rho=rho+c[k-1,0]*DoG(u,k,tau)
    
    return rho
    

def gen_dct(size):

    """
        Function to generate a DCT basis. It is used to enforce sparsity.
        
        Arguments: 
        
            size: The size of one of the dimensions (the matrix is a square matrix) of the DCT matrix
            
        Returns: 
        
            DCT basis of required size
    
    
    """

    D=tor.from_numpy(dct(np.eye(size),norm="ortho",axis=0))         ## Generate DCT basis
    D=D.view(1,1,D.size(0),-1).type(tor.FloatTensor)                ## Resize it and convert it to float datatype
    
    return D
 
def get_psf(psf_path,width,height):

    """
        Function to read a point-spread-function and return it after resizing and converting it to the required datatype
       
        Arguments: 
        
            psf_path:   The location of the directory that contains the psf image
            width:      The width of the sensor
            height:     The height of the sensor
            
        Returns:
        
            The psf after resizing and converting it to float datatype  
        
    """
    
    psf=cv2.resize(cv2.imread(psf_path,0),(width,height))       ## Read image and resize it
    psf=tor.from_numpy(psf).type(tor.FloatTensor)               ## Convert into a float dtype tensor
    psf=psf.view(1,1,psf.size(0),psf.size(1))                   ## reshaping the tensor 
    
    return psf
 
def conv(h,x):

    """
        Function to perform 2D-convolution in time domain.
        
        Arguments: 
        
            h:  signal1
            x:  signal2
            
        Returns: 
        
            The 2D convolution result
        
    """

    final_conv_dim=(512,512)                ## dimension of the convolution result before cropping
    x_dim=(x.size(2),x.size(3))             ## dimension of x
    h_dim=(h.size(2),h.size(3))             ## dimension of h
    crop_dim=x_dim                          ## image obtained after cropping is the same dimension as the image x

    padding=(final_conv_dim[0]-(x_dim[0]-h_dim[0]+1),final_conv_dim[1]-(x_dim[1]-h_dim[1]+1))   ## calculate the amount of padding required given final_conv_dim, x_dim and h_dim

    x_pad=F.pad(x,(padding[0]//2,padding[0]//2+1,padding[1]//2,padding[1]//2+1))    ## pad x
    y=F.conv2d(x_pad,h.flip(2,3),padding=0)                                         ## convolve x_pad with h
    
    ## starting and ending values along the column and the rows for cropping
    starti=(final_conv_dim[0]-crop_dim[0])//2                                       
    endi=crop_dim[0]+starti
    startj=(final_conv_dim[1]-crop_dim[1])//2
    endj=crop_dim[1]+startj

    ## Cropping
    y=y[:,:,starti:endi,startj:endj]
    
    return y

class t_layer(nn.Module):
    """
        Class that defines the tth-layer of the network
    
    """
    
    def __init__(self,learning_rate):
        
        super().__init__()
        
        self.learning_rate = learning_rate
        
    def forward(self,y,xt,D,h,param):
        """
            Compute the forward prop in the tth-layer
            
            
            xt1 = D.T ( rho ( D ( xt - (learning_rate)h_flip*( h*xt - y ) ) D.T ) ) D
                
                here '*' denotes linear convolution
        """
    
        
        h_flip=h.flip(2,3)
        a=conv(h,xt)-y                      
        b=xt-self.learning_rate*conv(h_flip,a)
        u=tor.matmul(D,tor.matmul(b,D.transpose(2,3)))
        u_=rho(u,param)
        xt1=tor.matmul(D.transpose(2,3),tor.matmul(u_,D))
        
        return xt1
        

class LetNet(nn.Module):

    """
        Class that defines the whole network called, "LetNet". This network applies the concept of LET for image reconstruction on lensless
        cameras
    """

    def __init__(self,c,tau,psf,num_layers,learning_rate):
        
        super().__init__()
        
        self.c=nn.Parameter(c,requires_grad=True)                   ## The linear coeffecients ; the are the paremeters of the network
        self.tau=nn.Parameter(tau,requires_grad=True)               ## The standard deviation ; it is the parameter of the network
        self.param=[self.c,self.tau]                                
        
        self.D=gen_dct(psf.size(2))                                 
        self.h=psf
        self.num_layers=num_layers                                  ## number of layers required
        
        ## uncomment to test a single layer
        ## self.layer1=t_layer()
        
        ## comment out this snippet to test just a single layer
        self.layers=[]
        for l in range(num_layers):                                 ## instantiate 'num_layers' number of layers and store each layer as an element of the list
            self.layers.append(t_layer(learning_rate))
        ## comment
        
    def forward(self,y):
        
        """
            Forward prop of the network
        """
        
        x0=tor.randn(y.shape)                                       ## Initialize the reconstructed image to half the psf pixel values (This method has been followed in the iterative appoach as well.
        
        ## uncomment to test a single layer
        ## out = self.layer1(y,x0,self.D,self.h,self.param)
        
        ## comment this snippet to test a single layer
        out=self.layers[0](y,x0,self.D,self.h,self.param)           ## Compute the output of the first layer
        for l in range(1,self.num_layers):
            out=self.layers[l](y,out,self.D,self.h,self.param)      ## Compute output of other 'num_layers-1' layers
        ## comment
        
        return out
        
        
class dataset(torch.utils.data.Dataset):

    """
        Class that defines the properties of a dataset object

        The functions look into gt_dir for the ground truth images and sensor_dir for the blurred images(sensor readings)
    """

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
        
        ## Convert the images into float dtype
        X=np.float32(cv2.imread(gt_addr,0))             
        Y=np.float32(cv2.imread(sensor_addr,0))
        
        ## Preprocess the images
        if(self.transform is not None):
            X=self.transform(X)
            Y=self.transform(Y)
        
        ## Convert the images to tensors and reshape 
        X=tor.from_numpy(X).type(tor.FloatTensor).unsqueeze(0)
        Y=tor.from_numpy(Y).type(tor.FloatTensor).unsqueeze(0)
            
        return Y,X
        
        
if __name__=="__main__":

    gts_dir="./gts_256_256/"                        ## The directory where the ground truth images are stored
    sensor_dir="./sensor_readings_256_256/"         ## The directory where the blurred images(sensor readings) are stored
    psf_path="./psf/psf_sample.tif"                 ## The directory where the psf is stored
    
    width=256                                       ## sensor width
    height=256                                      ## sensor height
    
    psf=get_psf(psf_path,width,height)              ## read the psf
    
    trainset=dataset(gts_dir,sensor_dir,transform=preprocess)                       ## instantiate a training set object
    trainloader=torch.utils.data.DataLoader(trainset,batch_size=2,shuffle=True)     ## instantiate a dataloader
    
    """
    testset=dataset(gts_dir,sensor_dir,transform=preprocess)                        ## instantiate a test set object
    testloader=torch.utils.data.DataLoader(testset,batch_size=2,shuffle=True)       ## instantiate a dataloader
    """
    
    ## K DoGs
    K=10
    ## randomly initialize the parameters 
    c=tor.randn(K,1)*0.001
    tau=tor.randn(1)*0.001
    
    ## Number of layers
    num_layers=1
    ## learning rate for each layer
    learning_rate = 1e-5
    
    ## The network is instantiated    
    clf=LetNet(c,tau,psf,num_layers,learning_rate)
    
    ## The mean-square error loss function 
    criterion=nn.MSELoss()
    
    ## Adam optimizer is used to optimize this problem
    optimizer=tor.optim.Adam(clf.parameters(),lr=1e-3)
    
    losses=[]
    epochs=10
    
    for epoch in range(epochs):
    
        l=[]
        for batch_idx,(Y,X) in enumerate(trainloader):
            
            X_=clf(Y)                       ## Forward Prop
            
            loss=criterion(X_,X)            ## Compute cost
            
            loss.backward()                 ## Back prop
            
            optimizer.step()                ## Update network parameters
            
            l.append(loss.item())           ## append the loss after processing each batch
        
            print("Batch Index: %d \t Loss in batch: %0.4f" % (batch_idx,l[-1]))
            
        losses.append(np.mean(l))           ## append the mean loss on that epoch
        
        print("Epoch: ",epoch,"\tCost: ",losses[-1])
        
      
