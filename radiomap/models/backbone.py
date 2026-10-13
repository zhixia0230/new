import math
import torch
from torch import nn
import torch.nn.functional as F


def group_norm(channels):
    groups = max(x for x in range(1, min(32,channels)+1) if channels % x == 0)
    return nn.GroupNorm(groups, channels)


def sinusoidal(time, dimensions):
    half = dimensions//2
    frequencies = torch.exp(-math.log(10000)*torch.arange(half,device=time.device,dtype=torch.float32)/max(half-1,1))
    phase = time.float().reshape(-1,1)*frequencies[None]
    return torch.cat((phase.sin(),phase.cos()),dim=1)


class ResidualBlock(nn.Module):
    def __init__(self, inputs, outputs, time_dim):
        super().__init__()
        self.norm1, self.norm2 = group_norm(inputs), group_norm(outputs)
        self.conv1 = nn.Conv2d(inputs,outputs,3,padding=1)
        self.conv2 = nn.Conv2d(outputs,outputs,3,padding=1)
        self.time = nn.Linear(time_dim,2*outputs)
        self.skip = nn.Identity() if inputs == outputs else nn.Conv2d(inputs,outputs,1)

    def forward(self, value, time):
        hidden = self.conv1(F.silu(self.norm1(value)))
        scale, shift = self.time(F.silu(time)).chunk(2,dim=1)
        hidden = self.norm2(hidden)*(1+scale[:,:,None,None])+shift[:,:,None,None]
        return self.skip(value)+self.conv2(F.silu(hidden))


class ConditionalUNet(nn.Module):
    def __init__(self, channels, cond_channels, res_blocks=2, time_features=256, time_embedding=512):
        super().__init__()
        self.channels, self.time_features = list(channels), time_features
        self.time_mlp = nn.Sequential(nn.Linear(time_features,time_embedding),nn.SiLU(),nn.Linear(time_embedding,time_embedding))
        self.input = nn.Conv2d(3,channels[0],3,padding=1)
        self.condition_projection = nn.ModuleList(nn.Conv2d(c,h,1) for c,h in zip(cond_channels,channels))
        self.encoder, self.downsample = nn.ModuleList(), nn.ModuleList()
        for index, width in enumerate(channels):
            self.encoder.append(nn.ModuleList(ResidualBlock(width,width,time_embedding) for _ in range(res_blocks)))
            if index+1 < len(channels):
                self.downsample.append(nn.Conv2d(width,channels[index+1],3,stride=2,padding=1))
        self.upsample, self.decoder = nn.ModuleList(), nn.ModuleList()
        for index in range(len(channels)-2,-1,-1):
            width = channels[index]
            self.upsample.append(nn.Conv2d(channels[index+1],width,3,padding=1))
            blocks = [ResidualBlock(2*width,width,time_embedding)]
            blocks += [ResidualBlock(width,width,time_embedding) for _ in range(res_blocks-1)]
            self.decoder.append(nn.ModuleList(blocks))
        self.output_norm = group_norm(channels[0])
        self.output = nn.Conv2d(channels[0],1,3,padding=1)

    def forward(self, state, time, conditions, reference, confidence):
        embedding = self.time_mlp(sinusoidal(time,self.time_features))
        hidden = self.input(torch.cat((state,reference,confidence),dim=1))
        skips = []
        for index, blocks in enumerate(self.encoder):
            hidden = hidden+self.condition_projection[index](conditions[index])
            for block in blocks:
                hidden = block(hidden,embedding)
            skips.append(hidden)
            if index < len(self.downsample):
                hidden = self.downsample[index](hidden)
        for upsample, blocks, skip in zip(self.upsample,self.decoder,reversed(skips[:-1])):
            hidden = upsample(F.interpolate(hidden,size=skip.shape[-2:],mode="nearest"))
            hidden = torch.cat((hidden,skip),dim=1)
            for block in blocks:
                hidden = block(hidden,embedding)
        hidden = F.silu(self.output_norm(hidden))
        return self.output(hidden), hidden
