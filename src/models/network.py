import torch
import torch.nn as nn
from typing import List

class AuthenticationAgent(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(AuthenticationAgent, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.classifier = nn.Linear(hidden_size, output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.classifier(features) 