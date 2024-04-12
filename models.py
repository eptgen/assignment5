import torch
import torch.nn as nn
import torch.nn.functional as F

# ------ TO DO ------
class cls_model(nn.Module):
    def __init__(self, num_classes=3):
        super(cls_model, self).__init__()
        N = 10000
        self.tnet1 = nn.Linear(3, 3, device = "cuda")

        self.mlp1_linear1 = nn.Linear(3, 64, device = "cuda")
        self.mlp1_batchnorm1 = nn.BatchNorm1d(N, device = "cuda")
        self.mlp1_relu1 = nn.ReLU()
        self.mlp1_linear2 = nn.Linear(64, 64, device = "cuda")
        self.mlp1_batchnorm2 = nn.BatchNorm1d(N, device = "cuda")
        self.mlp1_relu2 = nn.ReLU()

        self.tnet2 = nn.Linear(64, 64, device = "cuda")

        self.mlp2_linear1 = nn.Linear(64, 64, device = "cuda")
        self.mlp2_batchnorm1 = nn.BatchNorm1d(N, device = "cuda")
        self.mlp2_relu1 = nn.ReLU()
        self.mlp2_linear2 = nn.Linear(64, 128, device = "cuda")
        self.mlp2_batchnorm2 = nn.BatchNorm1d(N, device = "cuda")
        self.mlp2_relu2 = nn.ReLU()
        self.mlp2_linear3 = nn.Linear(128, 1024, device = "cuda")
        self.mlp2_batchnorm3 = nn.BatchNorm1d(N, device = "cuda")
        self.mlp2_relu3 = nn.ReLU()

        self.mlp3_linear1 = nn.Linear(1024, 512, device = "cuda")
        self.mlp3_batchnorm1 = nn.BatchNorm1d(512, device = "cuda")
        self.mlp3_relu1 = nn.ReLU()
        self.mlp3_dropout1 = nn.Dropout(0.3)
        self.mlp3_linear2 = nn.Linear(512, 256, device = "cuda")
        self.mlp3_batchnorm2 = nn.BatchNorm1d(256, device = "cuda")
        self.mlp3_relu2 = nn.ReLU()
        self.mlp3_dropout2 = nn.Dropout(0.3)
        self.mlp3_linear3 = nn.Linear(256, num_classes, device = "cuda")
        
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, num_classes)
        '''
        (B, N, _) = points.shape
        res = points # (B, N, 3)
        res = self.tnet1(res) # (B, N, 3)

        res = self.mlp1_linear1(res) # (B, N, 64)
        res = self.mlp1_batchnorm1(res) # (B, N, 64)
        res = self.mlp1_relu1(res) # (B, N, 64)
        res = self.mlp1_linear2(res) # (B, N, 64)
        res = self.mlp1_batchnorm2(res) # (B, N, 64)
        res = self.mlp1_relu2(res) # (B, N, 64)

        res = self.tnet2(res) # (B, N, 64)

        res = self.mlp2_linear1(res) # (B, N, 64)
        res = self.mlp2_batchnorm1(res) # (B, N, 64)
        res = self.mlp2_relu1(res) # (B, N, 64)
        res = self.mlp2_linear2(res) # (B, N, 128)
        res = self.mlp2_batchnorm2(res) # (B, N, 128)
        res = self.mlp2_relu2(res) # (B, N, 128)
        res = self.mlp2_linear3(res) # (B, N, 1024)
        res = self.mlp2_batchnorm3(res) # (B, N, 1024)
        res = self.mlp2_relu3(res) # (B, N, 1024)

        res, _ = torch.max(res, dim = 1) # (B, 1024)
        res = self.mlp3_linear1(res) # (B, 512)
        res = self.mlp3_batchnorm1(res) # (B, 512)
        res = self.mlp3_relu1(res) # (B, 512)
        res = self.mlp3_dropout1(res) # (B, 512)
        res = self.mlp3_linear2(res) # (B, 256)
        res = self.mlp3_batchnorm2(res) # (B, 256)
        res = self.mlp3_relu2(res) # (B, 256)
        res = self.mlp3_dropout2(res) # (B, 256)
        res = self.mlp3_linear3(res) # (B, num_classes)

        res = self.sigmoid(res) # (B, num_classes)
        res = self.softmax(res) # (B, num_classes)
        return res

# ------ TO DO ------
class seg_model(nn.Module):
    def __init__(self, num_seg_classes = 6):
        super(seg_model, self).__init__()
        N = 10000
        self.tnet1 = nn.Linear(3, 3, device = "cuda")

        self.mlp1_linear1 = nn.Linear(3, 64, device = "cuda")
        self.mlp1_batchnorm1 = nn.BatchNorm1d(N, device = "cuda")
        self.mlp1_relu1 = nn.ReLU()
        self.mlp1_linear2 = nn.Linear(64, 64, device = "cuda")
        self.mlp1_batchnorm2 = nn.BatchNorm1d(N, device = "cuda")
        self.mlp1_relu2 = nn.ReLU()

        self.tnet2 = nn.Linear(64, 64, device = "cuda")

        self.mlp2_linear1 = nn.Linear(64, 64, device = "cuda")
        self.mlp2_batchnorm1 = nn.BatchNorm1d(N, device = "cuda")
        self.mlp2_relu1 = nn.ReLU()
        self.mlp2_linear2 = nn.Linear(64, 128, device = "cuda")
        self.mlp2_batchnorm2 = nn.BatchNorm1d(N, device = "cuda")
        self.mlp2_relu2 = nn.ReLU()
        self.mlp2_linear3 = nn.Linear(128, 1024, device = "cuda")
        self.mlp2_batchnorm3 = nn.BatchNorm1d(N, device = "cuda")
        self.mlp2_relu3 = nn.ReLU()

        self.mlp3_linear1 = nn.Linear(1088, 512, device = "cuda")
        self.mlp3_batchnorm1 = nn.BatchNorm1d(N, device = "cuda")
        self.mlp3_relu1 = nn.ReLU()
        self.mlp3_linear2 = nn.Linear(512, 256, device = "cuda")
        self.mlp3_batchnorm2 = nn.BatchNorm1d(N, device = "cuda")
        self.mlp3_relu2 = nn.ReLU()
        self.mlp3_linear3 = nn.Linear(256, 128, device = "cuda")
        self.mlp3_batchnorm3 = nn.BatchNorm1d(N, device = "cuda")
        self.mlp3_relu3 = nn.ReLU()
        self.mlp3_linear4 = nn.Linear(128, 128, device = "cuda")
        self.mlp3_batchnorm4 = nn.BatchNorm1d(N, device = "cuda")
        self.mlp3_relu4 = nn.ReLU()
        self.mlp3_linear5 = nn.Linear(128, num_seg_classes, device = "cuda")

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, N, num_seg_classes)
        '''
        (B, N, _) = points.shape
        res = points # (B, N, 3)
        res = self.tnet1(res) # (B, N, 3)

        res = self.mlp1_linear1(res) # (B, N, 64)
        res = self.mlp1_batchnorm1(res) # (B, N, 64)
        res = self.mlp1_relu1(res) # (B, N, 64)
        res = self.mlp1_linear2(res) # (B, N, 64)
        res = self.mlp1_batchnorm2(res) # (B, N, 64)
        res = self.mlp1_relu2(res) # (B, N, 64)

        res = self.tnet2(res) # (B, N, 64)
        tmp = res

        res = self.mlp2_linear1(res) # (B, N, 64)
        res = self.mlp2_batchnorm1(res) # (B, N, 64)
        res = self.mlp2_relu1(res) # (B, N, 64)
        res = self.mlp2_linear2(res) # (B, N, 128)
        res = self.mlp2_batchnorm2(res) # (B, N, 128)
        res = self.mlp2_relu2(res) # (B, N, 128)
        res = self.mlp2_linear3(res) # (B, N, 1024)
        res = self.mlp2_batchnorm3(res) # (B, N, 1024)
        res = self.mlp2_relu3(res) # (B, N, 1024)

        res, _ = torch.max(res, dim = 1) # (B, 1024)
        res = res.unsqueeze(1) # (B, 1, 1024)
        res = torch.cat([res] * N, dim = 1) # (B, N, 1024)
        res = torch.cat((tmp, res), dim = 2) # (B, N, 1088)
        res = self.mlp3_linear1(res) # (B, N, 512)
        res = self.mlp3_batchnorm1(res) # (B, N, 512)
        res = self.mlp3_relu1(res) # (B, N, 512)
        res = self.mlp3_linear2(res) # (B, N, 256)
        res = self.mlp3_batchnorm2(res) # (B, N, 256)
        res = self.mlp3_relu2(res) # (B, N, 256)
        res = self.mlp3_linear3(res) # (B, N, 128)
        res = self.mlp3_batchnorm3(res) # (B, N, 128)
        res = self.mlp3_relu3(res) # (B, N, 128)
        res = self.mlp3_linear4(res) # (B, N, 128)
        res = self.mlp3_batchnorm4(res) # (B, N, 128)
        res = self.mlp3_relu4(res) # (B, N, 128)
        res = self.mlp3_linear5(res) # (B, N, num_seg_classes)
        return res



