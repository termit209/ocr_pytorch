import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models


class OcrModel_mobilenet(nn.Module):
    def __init__(self, num_characters):
        super(OcrModel_mobilenet, self).__init__()
        mobilenet = models.mobilenet_v2()
        self.mobilenet_feature_extractor = nn.Sequential(*list(mobilenet.children())[:-1])
        self.linear1 = nn.Linear(int(1280*10*3/64), 64)
        self.dropout1  = nn.Dropout(0.2)
        self.gru = nn.GRU(64, 32 , bidirectional=True,
                          num_layers=2,
                          dropout=0.25,
                          batch_first=True)
        self.output = nn.Linear(64, num_characters + 4)
        
    def forward(self, images, label_size, labels=None):
        bs, c, h, w = images.size()
        out = F.relu(self.mobilenet_feature_extractor(images))# 32 1280 7 7
        #print(out.size())

        x = out.permute(0,3,1,2) # 32 10, 1280, 3
        #print(x.size())
        x = torch.reshape(x,(bs, 64, int(1280*10*3/64)))#32 49 1280
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
            
            output_lenghts = label_size

            #print(output_lenghts)
            
            loss = nn.CTCLoss(blank=0, zero_infinity = True)(
                log_softmax_values,
                labels,
                input_lenghts,
                output_lenghts)
            
            return x, loss


class OcrModel_vgg16(nn.Module):
    def __init__(self, num_characters):
        super(OcrModel_vgg16, self).__init__()
        vgg = models.vgg16_bn(pretrained=True)
        self.vgg_feature_extractor = nn.Sequential(*list(vgg.children())[:-1])
        self.linear1 = nn.Linear(256, 64)
        self.dropout1  = nn.Dropout(0.2)
        self.gru = nn.GRU(64, 32 , bidirectional=True,
                          num_layers=2,
                          dropout=0.25,
                          batch_first=True)
        self.output = nn.Linear(64, num_characters + 4)
        
    def forward(self, images, label_size, labels=None):
        bs, c, h, w = images.size()
        out = F.relu(self.vgg_feature_extractor(images))
        #print(out.size())

        x = out.permute(0,3,1,2) # 32 7 512  7 
        # print(x.size()) 512 7 7 
        x = torch.reshape(x,(bs, 7*7*2, 256))#view(bs, 7*7*2, 256)  # 32 7 512*7 
        # print(x.size()[2])
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
            
            output_lenghts = label_size

             #print(output_lenghts)
            
            loss = nn.CTCLoss(blank=0, zero_infinity = True)(
                log_softmax_values,
                labels,
                input_lenghts,
                output_lenghts)
            
            return x, loss


class OcrModel_v0(nn.Module):
    def __init__(self, num_characters):
        super(OcrModel_v0, self).__init__()
        self.conv1 = nn.Conv2d(3,128, kernel_size=(3,3), padding=(1,1))
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2,2))
        self.conv2 = nn.Conv2d(128,64, kernel_size=(3,3), padding=(1,1))
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2,2))
        self.linear1 = nn.Linear(1152, 64)
        self.dropout1  = nn.Dropout(0.2)
        self.gru = nn.GRU(64, 32 , bidirectional=True,
                          num_layers=2,
                          dropout=0.25,
                          batch_first=True)
        self.output = nn.Linear(64, num_characters + 1)
        
    def forward(self, images, label_size, labels=None):
        bs, c, h, w = images.size()
        #print(bs, c, h, w)
        x = F.relu(self.conv1(images))
        #print(x.size())
        x = self.maxpool1(x)
        #print(x.size()) 
        x = F.relu(self.conv2(x))
        #print(x.size())
        x = self.maxpool2(x) # 32 512 7 7 
        # need to change channels for rnn bs, f, h, w --> bs, w, f, h


        x = x.permute(0,3,1,2) # 32 7 512  7 
        #print(x.size()) # 512 7 7 
        x = x.view(bs, x.size(1),-1)  # 32 7 512*7  x.view(bs, 7*7*2, 256) -1
        #print(x.size()[2], x.size())
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
            # print(input_lenghts)
            
            output_lenghts = label_size

            # print(output_lenghts)
            
            loss = nn.CTCLoss(blank=0, zero_infinity = True)(
                log_softmax_values,
                labels,
                input_lenghts,
                output_lenghts
            )  
            
            return x, loss
        

if __name__ == '__main__':
    # model = OcrModel_v0(model_arch='gluon_seresnext50_32x4d', num_characters=19, pretrained=True)
    model = OcrModel_v0(num_characters=91)
    img = torch.rand(5,3,75,300)
    label = torch.randint(1,20,(5,5))
    x, loss = model(img, label)