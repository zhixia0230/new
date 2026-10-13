import torch
from torch import nn
from radiomap.models.backbone import ConditionalUNet
from radiomap.models.condition_encoder import ConditionEncoder
from radiomap.models.risk_head import RiskHead
from radiomap.utils.seed import stable_seed


class RadioMapSystem(nn.Module):
    def __init__(self, config, seed=42):
        super().__init__()
        model, risk = config["model"],config["risk"]
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(stable_seed("model_init",seed))
            self.condition_encoder = ConditionEncoder(model["cond_channels"])
            self.backbone = ConditionalUNet(model["channels"],model["cond_channels"],model["res_blocks"],
                                           model["time_features"],model["time_embedding"])
        self.risk_head = None
        mode = config["confidence"]["mode"]
        if mode in ("learned","static"):
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(stable_seed("head_init",seed))
                self.risk_head = RiskHead(model["channels"][0],model["cond_channels"][-1],
                                          risk["region_size"],risk["positive_floor"],static=mode=="static")

    def encode_condition(self, condition):
        return self.condition_encoder(condition["building"],condition["tx_map"])

    def predict(self, state, time, cond_features, reference=None, confidence=None):
        reference = torch.zeros_like(state) if reference is None else reference
        confidence = torch.zeros_like(state) if confidence is None else confidence
        return self.backbone(state,time,cond_features,reference,confidence)

    def generator_parameters(self):
        yield from self.condition_encoder.parameters()
        yield from self.backbone.parameters()
