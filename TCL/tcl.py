import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.init as init
import functools
import torch.nn.functional as F

from census_transform import CensusTransform

torch.backends.cudnn.enabled = False
class PatchMatching(nn.Module):
    def __init__(self, kSize = 3,nsize=7,scale = 4,alpha=1):
        super(PatchMatching, self).__init__()
        self.scale = scale
        self.kSize = kSize
        self.nsize = nsize
        self.alpha = alpha

        self.ct = CensusTransform()

    def _unfold(self,data,with_unfold=False):
        
        if self.scale !=1:
            data = torch.nn.functional.interpolate(data,scale_factor=1.0/self.scale,mode='bicubic',align_corners=False)
        pad = self.kSize // 2

        data_pad = torch.nn.functional.pad(data,(pad,pad,pad,pad),mode='reflect')
        d1 = torch.nn.functional.unfold(data_pad,kernel_size=self.kSize)#.permute(0,2,1)
        if not with_unfold:
            return d1.permute(0,2,1).unsqueeze(-2)
        else:
            b,c,h,w = data.size()
            # print('d1',d1.shape,data.shape)
            d1 = d1.view(b,-1,h,w)
            c1 = d1.size()[1]
            pad = self.nsize // 2
            d1_pad = torch.nn.functional.pad(d1,(pad,pad,pad,pad),mode='reflect')
            d1_pad_unflod = torch.nn.functional.unfold(d1_pad,kernel_size=self.nsize)#.permute(0,2,1)
            d1_pad_unflod = d1_pad_unflod.view(b,c1,-1,h*w).permute(0,3,2,1)
            # print(d1_pad_unflod.shape)
            return d1_pad_unflod
    def _match(self,pred,ref_d0,ref_d1):
        # b
        b,n,c = pred.size()
        print('--',pred.shape)
        pred_2 = (pred**2).sum(-1).view(b,n,-1)
        ref_d0_2 = (ref_d0**2).sum(-1).view(b,-1,n)
        ref_d1_2  = (ref_d1**2).sum(-1).view(b,-1,n)
        # gt_2 = (gt**2).sum(-1).view(b,-1,n)

        error_d0 = pred_2 + ref_d0_2 - 2.0 * torch.matmul(pred,ref_d0.permute(0,2,1))
        error_d1 = pred_2 + ref_d1_2 - 2.0 * torch.matmul(pred,ref_d1.permute(0,2,1))

        score_d0 = torch.exp(self.alpha * error_d0)
        score_d1 = torch.exp(self.alpha * error_d1)
        # print('score_d0',score_d0.shape,score_d1.shape)

        weight,ind = torch.min(score_d0,dim=2)
        index_d0 = ind.unsqueeze(-1).expand([-1,-1,c])
        print(ref_d0.shape,index_d0.shape)
        matched_d0 = torch.gather(ref_d0,dim=1,index=index_d0)
        

        weight,ind = torch.min(score_d1,dim=2)
        index_d1 = ind.unsqueeze(-1).expand([-1,-1,c])
        matched_d1 = torch.gather(ref_d1,dim=1,index=index_d1)
        # print('matched_d1',matched_d1.shape)

        # error_gt_d0 = gt_2 + ref_d0_2 - 2.0 * torch.matmul(ref_d0,gt.permute(0,2,1))
        # score_gt_d0 = torch.exp(self.alpha * error_gt_d0)
        # weight,ind = torch.min(score_gt_d0,dim=2)
        # index_d0 = ind.unsqueeze(-1).expand([-1,-1,c])
        # matched_d0 = torch.gather(ref_d0,dim=1,index=index_d0)

        loss = ((pred - matched_d0)**2).mean() + ((pred - matched_d1)**2).mean()
        return loss

        # error_d1 = pred_2 + ref_d0_2 - 2.0 * torch.matmul(pred,ref_d0.permute(0,2,1))
    def forward(self,pred,I):

        

        pred_ct = self.ct(pred)
        I_0_ct = self.ct(I[:,0])
        I_1_ct = self.ct(I[:,1])
        


        pred_ct = self._unfold(pred_ct)
        I_0_ct   = self._unfold(I_0_ct,with_unfold=True)
        I_1_ct   = self._unfold(I_1_ct,with_unfold=True)
        
        cat_nbr_ct = torch.cat([I_0_ct,I_1_ct],2)

        pred_ct = pred_ct.repeat(1,1,self.nsize**2*2,1)

        dis_I_ct = ((pred_ct - cat_nbr_ct)**2).sum(-1)
        weight,ind = torch.min(dis_I_ct,dim=2)
        index_d = ind.unsqueeze(-1).unsqueeze(-1).repeat(1,1,self.nsize**2*2,3*self.kSize**2)


        pred = self._unfold(pred)
        I_0   = self._unfold(I[:,0],with_unfold=True)
        I_1   = self._unfold(I[:,1],with_unfold=True)
        
        cat_nbr = torch.cat([I_0,I_1],2)
        pred = pred.repeat(1,1,self.nsize**2*2,1)


        matched_d = torch.gather(cat_nbr,dim=2,index=index_d)

        # print(pred.shape,matched_d.shape)

      
        loss = ((pred[:,:,0] - matched_d[:,:,0])**2) *0.5

        return loss.mean()


        # print('pred_pad',pred.shape,I0.shape)

b,c,h,w = 2,3,64,64
pred = torch.randn(b,c,h,w, requires_grad=True)
I = torch.randn(b,2,c,h,w)
gt = torch.randn(b,c,h,w)

pm_func = PatchMatching()
# loss = pm_func(pred,torch.stack([pred,pred],1))
loss = pm_func(pred,I)

grad = torch.autograd.grad(inputs=pred, outputs=loss, allow_unused=True, retain_graph=True)
print('loss',loss,grad)