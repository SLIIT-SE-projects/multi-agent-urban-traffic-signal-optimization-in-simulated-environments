import torch.nn as nn

class TrafficPolicyHead(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # A simple linear layer to start, after MARL add more layers here for 'Actor-Critic'.
        self.actor_linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # Returns raw scores (logits).
        return self.actor_linear(x)