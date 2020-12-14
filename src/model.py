import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models



class OcrModel_v0(nn.Module):
    def __init__(self, num_characters):
        super(OcrModel_v0, self).__init__()
        mobilenet = models.mobilenet_v2()
        self.mobilenet_feature_extractor = nn.Sequential(*list(mobilenet.children())[:-1])
        self.linear1 = nn.Linear(1280, 64)
        self.dropout1  = nn.Dropout(0.2)
        self.gru = nn.GRU(64, 32 , bidirectional=True,
                          num_layers=2,
                          dropout=0.25,
                          batch_first=True)
        self.output = nn.Linear(64, num_characters + 4)
        
    def forward(self, images, labels=None):
        bs, c, h, w = images.size()
        out = F.relu(self.mobilenet_feature_extractor(images))# 32 1280 7 7
        #print(out.size())

        x = out.permute(0,3,1,2) # 32 7 1280  7 
        #print(x.size())
        x = torch.reshape(x,(bs, 7*7, 1280))#32 49 1280
        #print(x.size())
        x = self.linear1(x)
        x = self.dropout1(x)
        #print(x.size())


        x, _ = self.gru(x)
        #print(x.size())
        x = self.output(x)
        #print(x.size())
        # permute again 
        x = x.permute(1,0,2)
        #print(x.size())
        if labels is not None: 
            log_softmax_values =  F.log_softmax(x,2)   
            input_lenghts = torch.full(size=(bs,),
                                       fill_value=log_softmax_values.size(0), 
                                       dtype = torch.int32
                                       )
             #print(input_lenghts)
            
            output_lenghts = torch.full(size=(bs,),
                                        fill_value=labels.size(1), 
                                        dtype = torch.int32)
             #print(output_lenghts)
            
            loss = nn.CTCLoss(blank=0, zero_infinity = True)(
                log_softmax_values,
                labels,
                input_lenghts,
                output_lenghts)
            

            return x, loss