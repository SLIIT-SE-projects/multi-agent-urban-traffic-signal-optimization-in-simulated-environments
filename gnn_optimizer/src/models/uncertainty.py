import torch.nn as nn
from src.config import ModelConfig

class BayesianDropout(nn.Module):
    
    def __init__(self, p=ModelConfig.DROPOUT_RATE):
        super().__init__()
        self.p = p
        self.dropout = nn.Dropout(p=p)
        self.force_on = False # Toggle this to True during uncertainty estimation

    def forward(self, x):
        # If we are training, OR if we forced it on (for uncertainty), use dropout
        if self.training or self.force_on:
            return self.dropout(x)
        return x

    def enable_mc_dropout(self):
        self.force_on = True

    def disable_mc_dropout(self):
        self.force_on = False