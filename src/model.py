from torch import nn
from torch import functional as F


class OcrModel(nn.Module):
    def __init__(self, num_characters):
        super(OcrModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=(3, 3), padding=(1, 1))
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(128, 64, kernel_size=(3, 3), padding=(1, 1))
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.linear1 = nn.Linear(1152, 64)
        self.dropout1 = nn.Dropout(0.2)
        self.gru = nn.GRU(64, 32, bidirectional=True,
                          num_layers=2,
                          dropout=0.25,
                          batch_first=True)
        self.output = nn.Linear(64, num_characters + 1)

    def forward(self, images):
        bs, c, h, w = images.size()
        x = F.relu(self.conv1(images))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = x.permute(0, 3, 1, 2)
        x = x.view(bs, x.size(1), -1)
        x = self.linear1(x)
        x = self.dropout1(x)
        x, _ = self.gru(x)
        x = self.output(x)
        x = x.permute(1, 0, 2)
        return x


