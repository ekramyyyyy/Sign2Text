import torch, torch.nn as nn
from config import *


class PoseEncoder(nn.Module):
    def __init__(self, d=256, layers=4, heads=4, drop=0.2):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(POSE_IN, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(drop))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.randn(1, T_POSE + 1, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, d * 4, drop, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.out_dim = d

    def forward(self, x):                                   # (B, T, POSE_IN)
        x = torch.cat([self.cls.expand(x.size(0), -1, -1), self.proj(x)], 1)
        x = self.enc(x + self.pos[:, :x.size(1)])
        return self.norm(x[:, 0])


class RGBEncoder(nn.Module):
    """Pretrained VideoMAE (Kinetics-400). Lower layers frozen to fit on a single consumer GPU."""

    def __init__(self, name=VIDEOMAE, freeze_layers=8, pretrained=True):
        super().__init__()
        from transformers import VideoMAEModel, VideoMAEConfig
        self.m = VideoMAEModel.from_pretrained(name) if pretrained else VideoMAEModel(VideoMAEConfig.from_pretrained(name))
        if pretrained and freeze_layers:
            for p in self.m.embeddings.parameters():
                p.requires_grad = False
            for layer in self.m.encoder.layer[:freeze_layers]:
                for p in layer.parameters():
                    p.requires_grad = False
        self.out_dim = self.m.config.hidden_size
        self.norm = nn.LayerNorm(self.out_dim)

    def forward(self, x):                                   # (B, T, 3, 224, 224)
        return self.norm(self.m(pixel_values=x).last_hidden_state.mean(1))


class SignNet(nn.Module):
    def __init__(self, num_classes, mode="hybrid", pretrained=True):
        super().__init__()
        assert mode in ("pose", "rgb", "hybrid")
        self.mode, dims = mode, 0
        if mode in ("pose", "hybrid"):
            self.pose = PoseEncoder(); dims += self.pose.out_dim
        if mode in ("rgb", "hybrid"):
            self.rgb = RGBEncoder(pretrained=pretrained); dims += self.rgb.out_dim
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(dims, 512), nn.GELU(),
                                  nn.Dropout(0.3), nn.Linear(512, num_classes))

    def forward(self, pose=None, rgb=None):
        feats = []
        if self.mode in ("pose", "hybrid"):
            feats.append(self.pose(pose))
        if self.mode in ("rgb", "hybrid"):
            feats.append(self.rgb(rgb))
        if self.training and self.mode == "hybrid":         # modality dropout -> robust when a stream is unreliable
            r = torch.rand(len(feats[0]), 1, device=feats[0].device)
            feats[0] = feats[0] * (r >= 0.1)
            feats[1] = feats[1] * ((r < 0.1) | (r >= 0.2))
        return self.head(torch.cat(feats, -1))
