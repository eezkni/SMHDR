import torch
import torch.nn as nn
import torch.nn.functional as F 
from loss.vgg19 import VGG19_relu
from utils.utils import *

class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, pred, gt, loss_map=None):
        if loss_map is not None:
            loss_map = loss_map.repeat(1, pred.shape[1], 1, 1)
            pred = pred * loss_map
            gt = gt * loss_map
            return nn.L1Loss()(pred, gt)
        else:
            return nn.L1Loss()(pred, gt)
    
class LocalEnhancedLoss(nn.Module):
    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0], resize=False, criterion='l1'):
        super(LocalEnhancedLoss, self).__init__()
        if criterion == 'l1':
            self.criterion = nn.L1Loss()
        elif criterion == 'sl1':
            self.criterion = nn.SmoothL1Loss()
        elif criterion == 'l2':
            self.criterion = nn.MSELoss()
        else:
            raise NotImplementedError('Loss [{}] is not implemented'.format(criterion))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.add_module('vgg', VGG19_relu())
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
        self.weights = weights
        self.resize = resize
        self.transformer = torch.nn.functional.interpolate

    def __call__(self, x, y, loss_map):
        if self.resize:
            x = self.transformer(x, mode='bicubic', size=(224, 224), align_corners=True)
            y = self.transformer(y, mode='bicubic', size=(224, 224), align_corners=True)
        
        if x.shape[1] != 3:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)
            
        x = (x - self.mean.to(x)) / self.std.to(x)
        y = (y - self.mean.to(y)) / self.std.to(y)
        x_vgg, y_vgg = self.vgg(x, loss_map), self.vgg(y, loss_map)

        loss = 0.0
        loss += self.weights[0] * self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])
        loss += self.weights[1] * self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
        loss += self.weights[2] * self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
        loss += self.weights[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
        loss += self.weights[4] * self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])

        return loss

class Loss(nn.Module):
    def __init__(self, le_lambda=0.005, mu=5000):
        super(Loss, self).__init__()
        self.mu = mu
        self.le_lambda = le_lambda
        self.loss_recon = L1Loss()
        if self.le_lambda > 0:
            self.loss_le = LocalEnhancedLoss(weights=[1,1,1,1,1], resize=False, criterion='l1')

    def forward(self, pred, gt, loss_map = None):
        loss = 0
        loss_dict = {}
        pred_mu = range_compressor(pred, self.mu)
        gt_mu = range_compressor(gt, self.mu)
        loss_recon = self.loss_recon(pred_mu, gt_mu, loss_map)
        loss_dict['loss_recon'] = loss_recon
        loss = loss + loss_recon
        if self.le_lambda > 0:
            loss_le = self.loss_le(pred_mu, gt_mu, loss_map) * self.le_lambda
            loss_dict['loss_le'] = loss_le
            loss = loss + loss_le
        return loss, loss_dict