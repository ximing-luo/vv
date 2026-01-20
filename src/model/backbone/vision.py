import torch
import torch.nn as nn

class VisionProjector(nn.Module):
    def __init__(self, vision_hidden_dim=768, hidden_size=512):
        super().__init__()
        self.vision_hidden_dim = vision_hidden_dim
        self.hidden_size = hidden_size
        self.projector = nn.Sequential(
            nn.Linear(self.vision_hidden_dim, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )

    def forward(self, x):
        return self.projector(x)