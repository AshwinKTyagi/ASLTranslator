import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        ''' init the nerual network '''
        super(Net, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 8, 3),
            nn.BatchNorm2d(8),
            nn.AvgPool2d(2),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, 3),
            nn.BatchNorm2d(16),
            nn.AvgPool2d(2),
            nn.ReLU()
        )
        
        self.fc1 = nn.Linear(16*5*5, 100)
        self.fc2 = nn.Linear(100, 64)
        self.leak = nn.LeakyReLU()
        self.drop = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(64, 24)

    def forward(self, x):
        '''defines the forward prop algorithm'''
        #apply Convolution on the relu'd results from the convolution layers
        x = self.layer1(x)
        x = self.layer2(x)

        x = x.view((x.shape[0], -1))

        #run the fcs
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.leak(x)
        x = self.drop(x)

        #fc4 will give us the final **24** layers
        x = self.fc3(x)

        return x
