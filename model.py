import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_layer(nn.Module):
    def __init__(self,in_cn,mid_cn,out_cm):
        super(CNN_layer, self).__init__()
        
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_cn, out_channels=mid_cn, kernel_size=3,stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_cn, out_channels=out_cm, kernel_size=3,stride = 1, padding = 1),
            nn.ReLU()
        )
        self.pool = nn.MaxPool2d(kernel_size=2)
        
    def forward(self,x):
        
        x = self.conv_layer1(x)
        
        return x



class Self_Attention_layer(nn.Module):
    def __init__(self, in_channels):
        """
        :param in_channels: Number of input channels
        """
        super(Self_Attention_layer, self).__init__()

        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)

        # Output transformation
        self.fc_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape

        # Compute Q, K, V
        Q = self.query_conv(x) 
        K = self.key_conv(x)    
        V = self.value_conv(x) 

        # Flatten spatial dimensions (h, w) for attention
        Q = Q.view(b, c, -1)  
        K = K.view(b, c, -1)  
        V = V.view(b, c, -1) 

        # Compute attention scores
        attention = torch.softmax(torch.bmm(Q.permute(0, 2, 1), K) / (c ** 0.5), dim=-1)  # (batch_size, h*w, h*w)

        # Apply attention to V
        out = torch.bmm(V, attention.permute(0, 2, 1)) 
        out = out.view(b, c, h, w)  

        # Final transformation
        out = self.fc_out(out) 
        return out
    
class CNN_attention_block(nn.Module):
    
    def __init__(self, in_cn,mid_cn,out_cn):
        super(CNN_attention_block, self).__init__()
        self.cnn_11 = CNN_layer(in_cn,mid_cn,mid_cn)
        self.cnn_12 = CNN_layer(mid_cn,mid_cn,out_cn)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.attention = Self_Attention_layer(out_cn)
    
    def forward(self, x):
        
        x = self.cnn_11(x)
        
        x = self.cnn_12(x)
        
        x = self.attention(x)
        
        x = self.pool(x)
        
        return x
        
        
class CNN_attention_model(nn.Module):
    def __init__(self,depth,inter_channels):
        """
        inter_channels: Number of intermediate channels
        depth: number of cnn_attention_blocks
        assume image is 3 * 128 * 128, target is 7 labels
        """
        super(CNN_attention_model,self).__init__()
        self.layers = nn.ModuleList()
        self.fcs = nn.ModuleList()
        
        self.layers.append(CNN_attention_block(3,inter_channels,inter_channels))
        for i in range(1,depth):
            self.layers.append(CNN_attention_block(inter_channels*(2**i),inter_channels*(2**i),inter_channels*(2**(i+1))))
        for j in range(1,depth+1):
            feature_size = inter_channels*(2**(i+1))*(128/(2**depth))**2 / 2**j
            self.fcs.append(nn.Linear(feature_size,feature_size//2))
        self.fcs.append(nn.Linear(feature_size//2,7))
            
        
        
    def forward(self,x):
        for layer in self.layers:
            x = layer(x)  
        
        x = torch.flatten(x, start_dim=1)
        
        for fc in self.fcs:
            x = fc(x)  
            
        return x
            
    
