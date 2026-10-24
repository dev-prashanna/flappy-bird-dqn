import torch 
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self,state_dim=5, action_dim=2):
        super(DQN,self).__init__()

        self.network=nn.Sequential(
            nn.Linear(state_dim,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,action_dim)
        )
    def forward(self,x):
        return self.network(x)

    