import torch
import torch.nn as nn
import torch.nn.functional as F

from .segformer_head import SegFormerHead
from . import mix_transformer
from .pre import Class_Predictor
from misc import torchutils
import numpy as np



class CGM(nn.Module):
    def __init__(self, backbone, num_classes=None, embedding_dim=256, stride=None, pretrained=None, pooling=None,):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.feature_strides = [4, 8, 16, 32]
        self.stride = stride

        self.encoder = getattr(mix_transformer, backbone)(stride=self.stride)
        self.in_channels = self.encoder.embed_dims
        self.newcam_predictor = Class_Predictor(20, 512)

        ## initilize encoder
        if pretrained:
            state_dict = torch.load('pretrained/'+backbone+'.pth')
            state_dict.pop('head.weight')
            state_dict.pop('head.bias')
            self.encoder.load_state_dict(state_dict,)

        if pooling=="gmp":
            self.pooling = F.adaptive_max_pool2d
        elif pooling=="gap":
            self.pooling = F.adaptive_avg_pool2d

        self.dropout = torch.nn.Dropout2d(0.5)
        self.decoder = SegFormerHead(feature_strides=self.feature_strides, in_channels=self.in_channels, embedding_dim=self.embedding_dim, num_classes=self.num_classes)
        #self.decoder = conv_head.LargeFOV(self.in_channels[-1], out_planes=self.num_classes)

        self.attn_proj = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, bias=True)
        nn.init.kaiming_normal_(self.attn_proj.weight, a=np.sqrt(5), mode="fan_out")

        self.classifier = nn.Conv2d(in_channels=self.in_channels[3], out_channels=self.num_classes-1, kernel_size=1, bias=False)


    def get_param_groups(self):

        param_groups = [[], [], [], [], []] # backbone; backbone_norm; cls_head; seg_head;
        
        for name, param in list(self.encoder.named_parameters()):

            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

        param_groups[2].append(self.classifier.weight)
        param_groups[2].append(self.attn_proj.weight)
        param_groups[2].append(self.attn_proj.bias)

        for param in list(self.decoder.parameters()):
            param_groups[3].append(param)
        #加入计算sce损失的参数
        for param in list(self.newcam_predictor.parameters()):
            param_groups[4].append(param)

        return param_groups


    def forward(self, x, labels=None, cam_only=False, seg_detach=True,):

        _x, _attns = self.encoder(x)
        _x1, _x2, _x3, _x4 = _x

        seg = self.decoder(_x)
        #seg = self.decoder(_x4)

        attn_cat = torch.cat(_attns[-2:], dim=1)#.detach()
        attn_cat = attn_cat + attn_cat.permute(0, 1, 3, 2)
        attn_pred = self.attn_proj(attn_cat)
        attn_pred = torch.sigmoid(attn_pred)[:,0,...]

        cls_x4 = self.pooling(_x4,(1,1))
        cls_x4 = self.classifier(cls_x4)
        cls_x4 = cls_x4.view(-1, self.num_classes-1)
        
        cam_x4 = F.conv2d(_x4, self.classifier.weight)  #[4 20 32 32]即原来的cam
        cam_s4 = F.relu(cam_x4)  #[4 20 32 32]  
        cam_s4 = cam_s4/(F.adaptive_max_pool2d(cam_s4, (1, 1)) + 1e-5)  #[2 20 32 32]
        c_b, c_c, c_h, c_w = cam_s4.shape
        re_seg = F.interpolate(seg, size=(c_h, c_w), mode='bilinear', align_corners=False)
        re_seg = F.softmax(re_seg, dim=1)[:,1:,:,:]
        re_seg = re_seg/(F.adaptive_max_pool2d(re_seg, (1, 1)) + 1e-5)
        cam_s4 = cam_s4*re_seg
        #cam_s4 = cam_s4/(F.adaptive_max_pool2d(cam_s4, (1, 1)) + 1e-5)  #[2 20 32 32]
        cam_s4 = cam_s4.detach()        

        feature = cam_s4.unsqueeze(2) * _x4.unsqueeze(1)    #[2 20 512 32 32]
        mid_feature = feature.view(feature.size(0), feature.size(1), feature.size(2),-1)  # [2 20 512 1024]
        pre_feature = torch.mean(mid_feature, -1)  #[2 20 512 32]即用于sce损失的输入
        
        #mid_cam_s4 = cam_s4.unsqueeze(2) * torch.ones_like(_x4.unsqueeze(1))
        #mid_cam_s4 = mid_cam_s4.view(mid_cam_s4.size(0), mid_cam_s4.size(1), mid_cam_s4.size(2), -1)
        #pre_feature = torch.sum(mid_feature, dim=-1)/(torch.sum(mid_cam_s4, dim=-1)+1e-5)        
        
        if labels is not None:
            loss_ce, acc = self.newcam_predictor(pre_feature, labels)
        else:
            loss_ce = 0.0
            acc = 0.0

        if cam_only:
            newcam_s4 = (1.0*cam_x4 + 1.0*F.conv2d(_x4, self.newcam_predictor.classifier.weight)).detach()#
            # cam_s4 = F.conv2d(_x4, self.classifier.weight).detach()
            return newcam_s4, attn_pred

        return cls_x4, seg, _attns, attn_pred, loss_ce
    

if __name__=="__main__":

    pretrained_weights = torch.load('pretrained/mit_b1.pth')
    cgm = CGM('mit_b1', num_classes=20, embedding_dim=256, pretrained=True)
    cgm._param_groups()
    dummy_input = torch.rand(2,3,512,512)
    cgm(dummy_input)
