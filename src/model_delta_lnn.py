import torch
import torch.nn as nn
import torchvision.models as tv

class StudyEncoder(nn.Module):
    """
    Input: images per study (B,2,3,H,W)
    Output: study embedding (B,D)
    """
    def __init__(self, embed_dim=256, freeze_backbone=True):
        super().__init__()
        base = tv.resnet18(weights=tv.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # (B,512,1,1)
        self.proj = nn.Linear(512, embed_dim)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, imgs):  # imgs: (B,2,3,H,W)
        B, V, C, H, W = imgs.shape
        x = imgs.view(B * V, C, H, W)
        f = self.backbone(x).view(B * V, 512)          # (B*2,512)
        e = self.proj(f).view(B, V, -1)                # (B,2,D)
        return e.mean(dim=1)                           # (B,D)

class LiquidCell(nn.Module):
    """
    Continuous-time update:
      h(t+dt) = exp(-dt/tau)*h(t) + (1-exp(-dt/tau))*tanh(Wx + Uh + b)
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.W = nn.Linear(input_dim, hidden_dim)
        self.U = nn.Linear(hidden_dim, hidden_dim)
        self.b = nn.Parameter(torch.zeros(hidden_dim))
        self.tau_un = nn.Parameter(torch.zeros(hidden_dim))  # learnable time constants
        self.act = nn.Tanh()

    def tau(self):
        return torch.nn.functional.softplus(self.tau_un) + 1e-3

    def integrate(self, x, h, dt):  # dt: (B,)
        u = self.act(self.W(x) + self.U(h) + self.b)   # (B,H)
        tau = self.tau().unsqueeze(0)                  # (1,H)
        dt = dt.unsqueeze(1)                           # (B,1)
        decay = torch.exp(-dt / tau)                   # (B,H)
        return decay * h + (1.0 - decay) * u

class DeltaLNN(nn.Module):
    def __init__(self, num_labels, embed_dim=256, hidden_dim=256, freeze_backbone=True):
        super().__init__()
        self.encoder = StudyEncoder(embed_dim=embed_dim, freeze_backbone=freeze_backbone)
        self.cell = LiquidCell(embed_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim * 3, 3 * num_labels)

    def forward(self, prev_imgs, curr_imgs, dt):
        prev_e = self.encoder(prev_imgs)               # (B,D)
        curr_e = self.encoder(curr_imgs)               # (B,D)

        B = prev_e.size(0)
        h0 = torch.zeros(B, self.cell.hidden_dim, device=prev_e.device)

        h_prev = self.cell.integrate(prev_e, h0, torch.zeros_like(dt))
        h_curr = self.cell.integrate(curr_e, h_prev, dt)

        feats = torch.cat([h_prev, h_curr, (h_curr - h_prev)], dim=1)
        logits = self.head(feats).view(B, 3, -1)       # (B,3,K)
        return logits