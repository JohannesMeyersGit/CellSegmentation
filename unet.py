import torch 
import torch.nn as nn 

# original paper https://arxiv.org/pdf/1505.04597.pdf 


class DoubleConv(nn.Module):
    # double convolution block with same padding! --> leads to same input and output size which eases working with label masks
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.dconv = nn.Sequential(
            nn.Conv2d(channels_in, channels_out,3,1,1,bias=False),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels_out, channels_out,3,1,1,bias=False),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True)   
        )
    
    def forward(self,x):
        return self.dconv(x)


#example = torch.rand(1,1,704,520) # no of ims, RGB channels, im widths, im heights
#mydconv = DoubleConv(1,64)
#out = mydconv.forward(example)
#print(out.shape) 



class Unet(nn.Module):
    def __init__(self, in_channels=3, segm_channels=1, features = [64, 128, 256, 512]):  # RGB in binary segmentation out
        super(Unet,self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pooling = nn.MaxPool2d(kernel_size=2,stride=2) 
        
        for f in features:
            self.encoder.append(DoubleConv(in_channels,f))
            in_channels = f # new in channels equals old out channels 
        
        for f in reversed(features):
            self.decoder.append(nn.ConvTranspose2d(f*2, f, kernel_size=2, stride=2)) # check Conv Transposed 2D vs bilin. upsampling
            self.decoder.append(DoubleConv(f*2, f))
        
        self.bottleneck = DoubleConv(features[-1],features[-1]*2)
        self.out = nn.Conv2d(features[0], segm_channels, kernel_size=1, stride=1)


    def forward(self,x):
        skip_connections = []
        # decoder 
        for enc_layer in self.encoder:
            x = enc_layer(x)
            skip_connections.append(x)
            x = self.pooling(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1] # remove the last one as bottelneck has no skip connetction
        for i in range(0,len(self.decoder),2):
            x = self.decoder[i](x)
            skip_connection = skip_connections[i//2] # int div 2 
            if x.shape != skip_connection.shape: # prevent errors due shape errors from floor operation in pooling
                m = skip_connection.shape[2] - x.shape[2]
                n = skip_connection.shape[3] - x.shape[3]
                fun = nn.ZeroPad2d((0,n,0,m)) # correct by 0 padding
                x = fun(x)
            

            concat = torch.concat((skip_connection,x),dim=1) 

            x = self.decoder[i+1](concat)
        
        return self.out(x)


#example = torch.rand(8,1,704,520) # no of ims, RGB channels, im widths, im heights
#print(example.shape) 
#myunet = Unet(1,1)
#out = myunet.forward(example)
#print(out.shape) 

