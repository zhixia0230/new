import torch
from torch import nn
import torch.nn.functional as F
from radiomap.models.backbone import group_norm


class ConditionEncoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.blocks, self.tx_projection = nn.ModuleList(), nn.ModuleList()
        previous = 2
        for index, width in enumerate(channels):
            self.blocks.append(nn.Sequential(nn.Conv2d(previous,width,3,stride=1 if index==0 else 2,padding=1),
                                             group_norm(width),nn.SiLU(),nn.Conv2d(width,width,3,padding=1),
                                             group_norm(width),nn.SiLU()))
            self.tx_projection.append(nn.Conv2d(1,width,1))
            previous = width

    def forward(self, building, tx_map):
        hidden = torch.cat((building.float(),tx_map),dim=1)
        result = []
        for block, project in zip(self.blocks,self.tx_projection):
            hidden = block(hidden)
            tx = F.adaptive_max_pool2d(tx_map,hidden.shape[-2:])
            hidden = hidden+project(tx)
            result.append(hidden)
        return result
