## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        # Assume input shape = (1,224, 224)
        # Use kernel_size 5
        # Output width/height = (W-F)/S+1 = ((224-5)/1)+1 = 220
        # Output shape = (32, 220, 220)
        self.conv1 = nn.Conv2d(1, 32, 5)
        
        # Pool with kernel_size=2, stride=2
        # Output width/height of 1st max pooling: (W-F)/S+1 = ((220-2)/2)+1 = 110
        # Output shape of first max pooling: (32, 110, 110)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Use kernel_size 4
        # Output width/height of 2nd Conv layer: (W-F)/S+1 = ((110-4)/1)+1 = 107
        # Output shape of second conv layer: (64, 107, 107)
        self.conv2 = nn.Conv2d(32, 64, 4)
        
        # Output width/height of 2nd max pooling layer: (W-F)/S+1 = ((107-2)/2)+1 = 53
        # Output shape of second max pooling layer: (64, 53, 53)
        
        # Use kernel_size 3
        # Output width/height of 3rd Conv layer: (W-F)/S+1 = ((53-3)/1)+1 = 51
        # Output shape of second conv layer: (128, 51, 51)
        self.conv3 = nn.Conv2d(64, 128, 3)
        
        # Output width/height of 3rd max pooling layer: (W-F)/S+1 = ((51-2)/2)+1 = 25
        # Output shape of second max pooling layer: (128, 25, 25)
        
        # Use kernel_size 2
        # Output width/height of 4th Conv layer: (W-F)/S+1 = ((25-2)/1)+1 = 24
        # Output shape of second conv layer: (256, 24, 24)
        self.conv4 = nn.Conv2d(128, 256, 2)
        
        # Output width/height of 4th max pooling layer: (W-F)/S+1 = ((24-2)/2)+1 = 12
        # Output shape of second max pooling layer: (256, 12, 12)
        
        # Use kernel_size 1
        # Output width/height of 5th Conv layer: (W-F)/S+1 = ((12-1)/1)+1 = 12
        # Output shape of second conv layer: (512, 12, 12)
        self.conv5 = nn.Conv2d(256, 512, 1)
        
        # Output width/height of 5th max pooling layer: (W-F)/S+1 = ((12-2)/2)+1 = 6
        # Output shape of second max pooling layer: (512, 6, 6)
        
        # 512 outputs * the 6*6 filtered/pooled map size
        self.fc1 = nn.Linear(512*6*6, 1024)
        
        # finally, create 136 output channels (for the 136 classes)
        self.fc2 = nn.Linear(1024, 136)
        
        # dropout between layers: p=0.3
        self.drop = nn.Dropout(p=0.3)
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.selu(self.conv1(x)))
        x = self.drop(x)    
        
        x = self.pool(F.selu(self.conv2(x)))
        x = self.drop(x)
        
        x = self.pool(F.selu(self.conv3(x)))
        x = self.drop(x)
        
        x = self.pool(F.selu(self.conv4(x)))
        x = self.drop(x)
        
        x = self.pool(F.selu(self.conv5(x)))
        x = self.drop(x)
        
        # prep for linear layer
        # this line of code is the equivalent of Flatten in Keras
        x = x.view(x.size(0), -1)
        
        # two linear layers with dropout in between
        x = F.selu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        
        # final output
        # a modified x, having gone through all the layers of your model, should be returned
        return x
