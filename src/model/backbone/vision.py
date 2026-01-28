import torch
import torch.nn as nn

class VisionProjector(nn.Module):
    """
    [基础] 视觉投影层 (Vision Projector)
    将 ViT 输出的图像特征投影到 LLM 的 Embedding 空间
    结构: MLP (Linear -> GELU -> Linear)
    """
    def __init__(self, vision_hidden_dim=768, hidden_size=512):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(vision_hidden_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, x):
        return self.projector(x)
