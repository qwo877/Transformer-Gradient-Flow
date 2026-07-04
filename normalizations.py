import torch
import torch.nn as nn


class NormBase(nn.Module):
    def forward(self, x, cond=None):
        raise NotImplementedError


class IdentityNorm(NormBase):
    #給 WN 用
    def forward(self, x, cond=None):
        return x


class SeqBatchNorm(NormBase):
    def __init__(self, d_model):
        super().__init__()
        self.bn = nn.BatchNorm1d(d_model)

    def forward(self, x, cond=None):
        # (B, L, D) -> (B, D, L) -> BN -> (B, L, D)
        return self.bn(x.transpose(1, 2)).transpose(1, 2)


class TokenLayerNorm(NormBase):
    def __init__(self, d_model):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x, cond=None):
        return self.ln(x)


class SeqInstanceNorm(NormBase):
    def __init__(self, d_model):
        super().__init__()
        self.inorm = nn.InstanceNorm1d(d_model, affine=True)

    def forward(self, x, cond=None):
        return self.inorm(x.transpose(1, 2)).transpose(1, 2)


class SeqGroupNorm(NormBase):
    def __init__(self, d_model, groups=8):
        super().__init__()
        self.gn = nn.GroupNorm(groups, d_model)

    def forward(self, x, cond=None):
        return self.gn(x.transpose(1, 2)).transpose(1, 2)


class RMSNorm(NormBase):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x, cond=None):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class AdaIN(NormBase):
    def __init__(self, d_model):
        super().__init__()
        self.inorm = nn.InstanceNorm1d(d_model, affine=False)
        self.to_gamma = nn.Linear(d_model, d_model)
        self.to_beta = nn.Linear(d_model, d_model)
        for lin in (self.to_gamma, self.to_beta):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, x, cond=None):
        assert cond is not None, "AdaIN 需要條件輸入 cond"
        h = self.inorm(x.transpose(1, 2)).transpose(1, 2)
        c = cond.mean(dim=1)                        # (B, D) 條件向量
        gamma = self.to_gamma(c).unsqueeze(1)       # (B, 1, D)
        beta = self.to_beta(c).unsqueeze(1)
        return h * (1.0 + gamma) + beta


class AdaINLocal(AdaIN):
    """AdaIN-local:與 AdaIN 唯一的差別是「不做序列平均池化」——
    gamma/beta 由條件圖逐位置生成 (B, L, D)。

    作為條件「粒度」的消融對照,孤立單一變因:
        AdaIN(全域向量) vs AdaIN-local(逐位置向量) vs SPADE(逐位置 + 共享 MLP)
    生成層繼承 AdaIN 的零初始化(初始等價於純 IN)。
    """

    def forward(self, x, cond=None):
        assert cond is not None, "AdaIN-local 需要條件輸入 cond"
        h = self.inorm(x.transpose(1, 2)).transpose(1, 2)
        gamma = self.to_gamma(cond)                 # (B, L, D) 逐位置
        beta = self.to_beta(cond)
        return h * (1.0 + gamma) + beta


class SPADE(NormBase):
    def __init__(self, d_model, hidden=64):
        super().__init__()
        self.inorm = nn.InstanceNorm1d(d_model, affine=False)
        self.shared = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU())
        self.to_gamma = nn.Linear(hidden, d_model)
        self.to_beta = nn.Linear(hidden, d_model)
        for lin in (self.to_gamma, self.to_beta):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, x, cond=None):
        assert cond is not None, "SPADE 需要條件輸入 cond"
        h = self.inorm(x.transpose(1, 2)).transpose(1, 2)
        a = self.shared(cond)                        # (B, L, hidden)
        return h * (1.0 + self.to_gamma(a)) + self.to_beta(a)


def build_norm(name: str, d_model: int) -> NormBase:
    # 建立對應
    table = {
        "bn": lambda: SeqBatchNorm(d_model),
        "ln": lambda: TokenLayerNorm(d_model),
        "in": lambda: SeqInstanceNorm(d_model),
        "gn": lambda: SeqGroupNorm(d_model),
        "rms": lambda: RMSNorm(d_model),
        "adain": lambda: AdaIN(d_model),
        "adain_local": lambda: AdaINLocal(d_model),
        "spade": lambda: SPADE(d_model),
        "wn": lambda: IdentityNorm(),
        "none": lambda: IdentityNorm(),
    }
    if name not in table:
        raise ValueError(f"未知的歸一化方法: {name}")
    return table[name]()
