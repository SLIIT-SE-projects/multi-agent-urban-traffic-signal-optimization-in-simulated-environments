import torch.nn as nn

class TrafficPolicyHead(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        # 1. ACTOR: Estimates which action to take
        self.actor = nn.Linear(input_dim, output_dim)
        
        # 2. CRITIC: Estimates the value of the state
        self.critic = nn.Linear(input_dim, 1)

    def forward(self, x):
        
        action_logits = self.actor(x)
        state_value = self.critic(x)
        
        return action_logits, state_value