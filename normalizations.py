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


class StraightThroughLayerNorm(NormBase):
    #前向與 LayerNorm 逐位元相同,反向讓梯度以恆等通過。

    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x, cond=None):
        mu = x.mean(dim=-1, keepdim=True)
        sigma = x.var(dim=-1, unbiased=False, keepdim=True).add(self.eps).sqrt()
        x_hat = (x - mu) / sigma
        # 前向取 x_hat 反向走恆等x + (x_hat - x).detach() 的值等於 x_hat  對 x 的梯度是 I
        x_st = x + (x_hat - x).detach()
        return x_st * self.weight + self.bias


class _RenormGrad(torch.autograd.Function):
    #前向恆等(位元完全不變),反向把梯度的 RMS 正規化成定值

    #改梯度的大小 不改方 把 逐層梯度大小失衡這一個因素單獨拿出來

    @staticmethod
    def forward(ctx, x, target_rms):
        ctx.target_rms = target_rms
        return x
#
    @staticmethod
    def backward(ctx, g):
        rms = g.pow(2).mean().sqrt()
        # 梯度為 0 時不動 避免放大數值雜訊
        scale = ctx.target_rms / rms if rms > 0 else 1.0
        return g * scale, None


class GradRenormLayerNorm(NormBase):
    #前向與 LayerNorm 位元相同,反向強制每層梯度 RMS 相等。


    def __init__(self, d_model, target_rms=6.0e-5):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        # 預設值是校準過的:等於 Post-LN 在 copy 任務、16 層、初始化時
        # 主幹逐層梯度 RMS 的中位數(6.007e-05)
        # 這樣重正規化只 拉平剖面 不改變整體梯度尺度
        # 否則在 SGD 下等於偷偷改了 effective learning rate 對照就不乾淨了
        # 此值與任務/loss 尺度相關 變換時需重新校準
        self.target_rms = target_rms

    def forward(self, x, cond=None):
        return _RenormGrad.apply(self.ln(x), self.target_rms)


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
    #AdaIN-local:與 AdaIN 唯一的差別是「不做序列平均池化」——
    #gamma/beta 由條件圖逐位置生成 (B, L, D)

    #作為條件「粒度」的消融對照,孤立單一變因:
    #    全域向量 vs 逐位置向量 vs SPADE(逐位置 + 共享 MLP)
    #生成層繼承 AdaIN 的零初始化(初始應等價於純 IN(? )
    

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


def build_norm(name: str, d_model: int, target_rms: float = None) -> NormBase:
    # 建立對應 target_rms 只對 ln_gr 有意義:None = 用 GradRenormLayerNorm
    # 的預設(在 copy 上對 Post-LN 校準的 6.0e-5)
    # 這個管道是實驗 3 需要的 —— 把同一個干預施加在 Pre-LN / DeepNorm 上時
    # 必須逐配置重新校準 否則等於偷偷改變了整體梯度尺度
    table = {
        "bn": lambda: SeqBatchNorm(d_model),
        "ln": lambda: TokenLayerNorm(d_model),
        "ln_st": lambda: StraightThroughLayerNorm(d_model),
        "ln_gr": lambda: (GradRenormLayerNorm(d_model) if target_rms is None
                          else GradRenormLayerNorm(d_model, target_rms)),
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
