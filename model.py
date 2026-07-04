import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

from normalizations import build_norm, IdentityNorm, TokenLayerNorm


def _make_linear(d_in, d_out, gain=1.0, use_wn=False):
    #建立 Linear:先做 Xavier 初始化(可帶 DeepNorm 的 beta gain) 再視需要包上 WeightNorm 重參數化(包裝時 g 會吸收現有範數 輸出不變)
    lin = nn.Linear(d_in, d_out)
    nn.init.xavier_uniform_(lin.weight, gain=gain)
    nn.init.zeros_(lin.bias)
    return weight_norm(lin) if use_wn else lin


class MHSA(nn.Module):
    #多頭因果自注意力 Q/K 用 gain=1;V/O 的初始化 gain 由外部指定(DeepNorm 只縮放 V/O 與 FFN 不動 Q/K)

    def __init__(self, d_model, n_heads, use_wn=False, gain_vo=1.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q = _make_linear(d_model, d_model, 1.0, use_wn)
        self.k = _make_linear(d_model, d_model, 1.0, use_wn)
        self.v = _make_linear(d_model, d_model, gain_vo, use_wn)
        self.o = _make_linear(d_model, d_model, gain_vo, use_wn)

    def forward(self, x, weights_out=None):
        B, L, D = x.shape

        def split(t):
            return t.view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        q, k, v = split(self.q(x)), split(self.k(x)), split(self.v(x))
        if weights_out is None:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            #訓練時不走這裡
            scores = q @ k.transpose(-2, -1) * self.d_head ** -0.5
            mask = torch.full((L, L), float("-inf"), device=x.device).triu(1)
            w = (scores + mask).softmax(dim=-1)
            weights_out.append(w.detach())
            out = w @ v
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.o(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, use_wn=False, gain=1.0):
        super().__init__()
        self.w1 = _make_linear(d_model, d_ff, gain, use_wn)
        self.w2 = _make_linear(d_ff, d_model, gain, use_wn)

    def forward(self, x):
        return self.w2(F.gelu(self.w1(x)))


class Block(nn.Module):
    N_NORMS = {"post": 2, "pre": 2, "deepnorm": 2, "sandwich": 4, "none": 0}

    def __init__(self, d_model, n_heads, d_ff, norm_name, placement,
                 alpha=1.0, beta=1.0):
        super().__init__()
        assert placement in self.N_NORMS, f"未知 placement: {placement}"
        use_wn = (norm_name == "wn")
        self.placement = placement
        self.alpha = alpha  # DeepNorm 殘差放大係數,其餘策略為 1.0
        self.attn = MHSA(d_model, n_heads, use_wn=use_wn, gain_vo=beta)
        self.ff = FeedForward(d_model, d_ff, use_wn=use_wn, gain=beta)
        self.norms = nn.ModuleList(
            build_norm(norm_name, d_model) for _ in range(self.N_NORMS[placement])
        )

    def forward(self, x, cond, weights_out=None):
        p = self.placement
        if p in ("post", "deepnorm"):
            x = self.norms[0](self.alpha * x + self.attn(x, weights_out), cond)
            x = self.norms[1](self.alpha * x + self.ff(x), cond)
        elif p == "pre":
            x = x + self.attn(self.norms[0](x, cond), weights_out)
            x = x + self.ff(self.norms[1](x, cond))
        elif p == "sandwich":
            x = x + self.norms[1](self.attn(self.norms[0](x, cond), weights_out), cond)
            x = x + self.norms[3](self.ff(self.norms[2](x, cond)), cond)
        else:  # none
            x = x + self.attn(x, weights_out)
            x = x + self.ff(x)
        return x


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, seq_len, d_model=128, n_heads=4,
                 d_ff=512, n_layers=16, norm_name="ln", placement="post"):
        super().__init__()
        if placement == "deepnorm":
            alpha = (2.0 * n_layers) ** 0.25
            beta = (8.0 * n_layers) ** -0.25
        else:
            alpha, beta = 1.0, 1.0

        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(seq_len, d_model)
        nn.init.normal_(self.tok.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)

        self.blocks = nn.ModuleList(
            Block(d_model, n_heads, d_ff, norm_name, placement, alpha, beta)
            for _ in range(n_layers)
        )
        # Pre-LN / Sandwich 的殘差主幹沒被歸一化過,輸出前需要一個最終 LN;
        # Post-LN / DeepNorm 出口已經被歸一化;WN(none) 依原始設定不加
        self.final_norm = (TokenLayerNorm(d_model)
                           if placement in ("pre", "sandwich") else IdentityNorm())
        self.head = nn.Linear(d_model, vocab_size)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, tokens, record_hidden=False, weights_out=None):
        B, L = tokens.shape
        x = self.tok(tokens) + self.pos(torch.arange(L, device=tokens.device))
        
        cond = x.clone()

        hiddens = []
        if record_hidden:
            x.retain_grad()
            cond.retain_grad()
            hiddens.append(x)
        for blk in self.blocks:
            x = blk(x, cond, weights_out)
            if record_hidden:
                x.retain_grad()
                hiddens.append(x)

        logits = self.head(self.final_norm(x))
        return logits, hiddens, cond
