import torch
import torch.nn as nn

class MaskBlock(nn.Module):
    
    def __init__(self, adl_drop_rate):
        super(MaskBlock, self).__init__()
        self.adl_drop_rate = adl_drop_rate

    def forward(self, x, mask_map):
        # x: [B,C,H,W]
        x0 = x.mean(dim=1, keepdim=True) # shape: [B,1,H,W]
        importance_map = torch.sigmoid(x0) # shape: [B,1,H,W]
        random_tensor = torch.rand([], dtype=torch.float32) + self.adl_drop_rate
        binary_tensor = random_tensor.floor()        
        final_mask = (1. - binary_tensor) * importance_map + binary_tensor * mask_map
        x = x * final_mask # shape: [B,C,H,W]
        return x