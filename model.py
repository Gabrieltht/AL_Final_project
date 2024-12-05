import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_layer_2(nn.Module):
    def __init__(self,in_cn,mid_cn,out_cm):
        super(CNN_layer_2, self).__init__()
        
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_cn, out_channels=mid_cn, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_cn),  
            nn.ReLU(),
            nn.Conv2d(in_channels=mid_cn, out_channels=out_cm, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_cm),  
            nn.ReLU()
        )
    def forward(self,x):
        
        x = self.conv_layer1(x)
        
        return x

class CNN_layer_3(nn.Module):
    def __init__(self,in_cn,mid_cn,out_cm):
        super(CNN_layer_3, self).__init__()
        
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_cn, out_channels=mid_cn, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_cn),  
            nn.ReLU(),
            nn.Conv2d(in_channels=mid_cn, out_channels=mid_cn, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_cn),  
            nn.ReLU(),
            nn.Conv2d(in_channels=mid_cn, out_channels=out_cm, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_cm),  
            nn.ReLU()
        )

        
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
        self.cnn_11 = CNN_layer_2(in_cn,mid_cn,out_cn)
        # self.cnn_12 = CNN_layer_2(mid_cn,mid_cn,out_cn)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.attention = Self_Attention_layer(out_cn)
    
    def forward(self, x):
        
        x = self.cnn_11(x)
        
        # x = self.cnn_12(x)
        
        x = self.attention(x)
        
        x = self.pool(x)
        
        return x
        
        
class Advanced_CNN_Attention_Model(nn.Module):
    def __init__(self, in_channel, target_size):
        super(Advanced_CNN_Attention_Model, self).__init__()
        self.cnn1 = CNN_layer_2(in_channel, 64, 128)
        self.cnn2 = CNN_layer_2(128, 256, 256)
        self.cnn3 = CNN_layer_3(256, 512, 512)
        self.cnn4 = CNN_layer_3(512, 512, 1024)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.att1 = Self_Attention_layer(256)
        self.att2 = Self_Attention_layer(512)
        self.att3 = Self_Attention_layer(1024)
        self.reduce_channels_1 = nn.Conv2d(256, 128, kernel_size=1)
        self.reduce_channels_2 = nn.Conv2d(512, 256, kernel_size=1)
        self.reduce_channels_3 = nn.Conv2d(1024, 512, kernel_size=1)

        self.fcs = nn.Sequential(
            nn.Linear(128 + 256 + 512 + 1024, 512, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 128, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, target_size, bias=True)
        )

    def _forward_features(self, x):
        x = self.cnn1(x)
        x = self.pool(x)
        x = self.cnn2(x)
        att1_x = self.reduce_channels_1(self.att1(x))  # Apply reduce_channels_1
        att1_x = self.global_pool(att1_x).view(att1_x.size(0), -1)  # Flatten after pooling
        x = self.pool(x)
        x = self.cnn3(x)
        att2_x = self.reduce_channels_2(self.att2(x))  # Apply reduce_channels_2
        att2_x = self.global_pool(att2_x).view(att2_x.size(0), -1)  # Flatten after pooling
        x = self.pool(x)
        x = self.cnn4(x)
        att3_x = self.reduce_channels_3(self.att3(x))  # Apply reduce_channels_3
        att3_x = self.global_pool(att3_x).view(att3_x.size(0), -1)  # Flatten after pooling
        x = self.global_pool(x).view(x.size(0), -1)  # Final global pooling and flatten

        # Concatenate outputs from all levels
        x = torch.cat([x, att1_x, att2_x, att3_x], dim=1)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = self.fcs(x)
        return x

    def predict(self, logits):
        probs = F.softmax(logits, dim=1)
        labels = torch.argmax(probs, dim=1)
        return labels


    
class CNN_attention_model_S(nn.Module):
    def __init__(self):
        """
        inter_channels: Number of intermediate channels
        depth: number of cnn_attention_blocks
        assume image is 3 * 128 * 128, target is 7 labels
        """
        super(CNN_attention_model_S,self).__init__()
        self.layers = nn.ModuleList()
        self.fcs = nn.ModuleList()
        
        self.layers.append(CNN_attention_block(3,16,32))
        self.layers.append(CNN_attention_block(32,64,64))
        self.layers.append(CNN_attention_block(64,128,128))
        self.layers.append(CNN_attention_block(128,128,256))
        self.fcs.append(nn.Linear(16384,8192,bias=True))
        self.fcs.append(nn.Dropout(p=0.2))  
        self.fcs.append(nn.ReLU())
        self.fcs.append(nn.Linear(8192,2048,bias=True))
        self.fcs.append(nn.Dropout(p=0.2))
        self.fcs.append(nn.ReLU())  
        self.fcs.append(nn.Linear(2048,512,bias=True))
        self.fcs.append(nn.Dropout(p=0.2))  
        self.fcs.append(nn.ReLU())
        self.fcs.append(nn.Linear(512,7,bias=True))
            
        
        
    def forward(self,x):
        for layer in self.layers:
            x = layer(x)  
        
        x = torch.flatten(x, start_dim=1)
        
        for fc in self.fcs:
            x = fc(x)  
            
        return x
    
    def predict(self,logits):
        
        probs = F.softmax(logits,dim=1)
        labels = torch.argmax(probs,dim=1)
        
        return labels

# class CNN_attention_res_model(nn.Module):
#     def __init__(self,in_channel,target_size):
#         super(CNN_attention_res_model,self).__init__()
#         self.cnn1 = CNN_layer_2(in_channel,16,16)
#         self.cnn2 = CNN_layer_2(16,32,32)
#         self.cnn3 = CNN_layer_3(32,64,64)
#         self.cnn4 = CNN_layer_3(64,128,128)
#         self.cnn5 = CNN_layer_3(128,128,128)
#         self.pool2 = nn.MaxPool2d(kernel_size=2)
#         self.att_pool_4 = nn.MaxPool2d(kernel_size=4)
        
#         self.att1 = Self_Attention_layer(in_channels=16)
        
#         self.att2 = Self_Attention_layer(in_channels=32)
        
#         self.att3 = Self_Attention_layer(in_channels=64)
        
#         self.fcs = nn.Sequential(
#         nn.Linear(in_features=43008,out_features=10752,bias = True),
#         nn.Dropout(p=0.2),
#         nn.ReLU(),    
#         nn.Linear(in_features=10752,out_features=2688,bias = True),
#         nn.Dropout(p=0.2),
#         nn.ReLU(),
#         nn.Linear(in_features=2688, out_features=672,bias = True),
#         nn.Dropout(p=0.2),
#         nn.ReLU(),
#         nn.Linear(in_features=672, out_features=target_size,bias = True)     
#         )
#     def forward(self,x):
        
#         x = self.cnn1(x) 
        
#         att1_x = self.att1(self.att_pool_4(x))
#         x = self.pool2(x)
        
#         x = self.cnn2(x)
#         att2_x = self.att2(self.att_pool_4(x))
        
#         x = self.pool2(x)
        
#         x = self.cnn3(x)
#         att3_x = self.att3(self.pool2(x))
        
#         x = self.pool2(x)
        
#         x = self.pool2(self.cnn4(x))
        
#         x = self.pool2(self.cnn5(x))
        
#         x = x.view(x.size(0), -1)

#         att1_x = att1_x.view(att1_x.size(0), -1) 
#         att2_x = att2_x.view(att2_x.size(0), -1) 
#         att3_x = att3_x.view(att3_x.size(0), -1) 
        
#         x = torch.cat([x, att1_x, att2_x, att3_x], dim=1)
        
#         x = self.fcs(x)
        
#         return x
    
#     def predict(self,logits):
        
#         probs = F.softmax(logits,dim=1)
#         labels = torch.argmax(probs,dim=1)
        
#         return labels
        
        
class CNN_attention_res_model(nn.Module):
    def __init__(self,in_channel,target_size):
        super(CNN_attention_res_model,self).__init__()
        self.cnn1 = CNN_layer_2(in_channel, 32, 64)
        self.cnn2 = CNN_layer_2(64, 128, 128)
        self.cnn3 = CNN_layer_3(128, 256, 256)
        self.cnn4 = CNN_layer_3(256, 512, 512)
        self.cnn5 = CNN_layer_3(512,512,512)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.att_pool_4 = nn.MaxPool2d(kernel_size=4)
        
        self.att1 = Self_Attention_layer(in_channels=128)
        
        self.att2 = Self_Attention_layer(in_channels=256)
        
        self.att3 = Self_Attention_layer(in_channels=512)
        
        
        self.global_pool = nn.AdaptiveAvgPool2d(1) 
        
        self.reduce_channels_1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)
        self.reduce_channels_2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1)
        self.reduce_channels_3 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)
        
        self.fcs = nn.Sequential(
        nn.Linear(in_features=41472,out_features=10368,bias = True),
        nn.Dropout(p=0.5),
        nn.ReLU(),    
        nn.Linear(in_features=10368,out_features=2592,bias = True),
        nn.Dropout(p=0.5),
        nn.ReLU(),
        nn.Linear(in_features=2592, out_features=648,bias = True),
        nn.Dropout(p=0.5),
        nn.ReLU(),
        nn.Linear(in_features=648, out_features=target_size,bias = True)     
        )
    def forward(self,x):
        
        x = self.cnn1(x) 
        x = self.pool2(x)
        
        x = self.cnn2(x)
        att1_x = self.reduce_channels_1(self.att1(self.att_pool_4(x)))
        x = self.pool2(x)
        
        x = self.cnn3(x)
        att2_x = self.reduce_channels_2(self.att2(self.att_pool_4(x)))
        x = self.pool2(x)
        
        x = self.cnn4(x)
        att3_x = self.reduce_channels_3(self.att3(self.pool2(x)))
        x = self.pool2(x)
        
        x = self.global_pool(self.cnn5(x))

        x = x.view(x.size(0), -1)

        att1_x = att1_x.view(att1_x.size(0), -1) 
        att2_x = att2_x.view(att2_x.size(0), -1) 
        att3_x = att3_x.view(att3_x.size(0), -1) 
        
        x = torch.cat([x, att1_x, att2_x, att3_x], dim=1)
        
        x = self.fcs(x)
        
        return x
    
    def predict(self,logits):
        
        probs = F.softmax(logits,dim=1)
        labels = torch.argmax(probs,dim=1)
        
        return labels
        
        