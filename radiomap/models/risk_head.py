import torch
from torch import nn
import torch.nn.functional as F
from radiomap.models.backbone import sinusoidal
from radiomap.data.geometry import region_mean


class RiskHead(nn.Module):
    def __init__(self, state_channels, condition_channels, region_size=16, positive_floor=1e-8, static=False):
        super().__init__()
        self.region_size, self.positive_floor, self.static = region_size, positive_floor, static
        self.input_channels = state_channels+condition_channels+5+16
        self.network = nn.Sequential(nn.Conv2d(self.input_channels,64,1),nn.SiLU(),nn.Conv2d(64,1,1))

    def forward(self, hidden, condition_features, time, geometry_stats, free):
        with torch.autocast(device_type=hidden.device.type,enabled=False):
            state = region_mean(hidden.detach().float(),free,self.region_size)
            return self.predict_regions(state,condition_features,time,geometry_stats)

    def predict_regions(self,state,condition_features,time,geometry_stats):
        with torch.autocast(device_type=state.device.type,enabled=False):
            state=state.detach().float()
            if self.static:
                state = torch.zeros_like(state)
            condition = condition_features.detach().float()
            if condition.shape[-2:] != state.shape[-2:]:
                condition = F.adaptive_avg_pool2d(condition,state.shape[-2:])
            time_features = sinusoidal(time,16)[:,:,None,None].expand(-1,-1,*state.shape[-2:])
            inputs = torch.cat((state,condition,geometry_stats.detach().float(),time_features),dim=1)
            return F.softplus(self.network(inputs))+self.positive_floor
