import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
# import
from normalizations import build_norm, IdentityNorm, TokenLayerNorm


def _make_linear(d_in, d_out, gain=1.0, use_wn=False):
    #建立 Linear 先做 Xavier 初始化(可帶 DeepNorm 的 beta gain) ? 包上 WeightNorm 重參數化(包裝時 g 會吸收現有範數 輸出不變)
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
            #訓練時不走這裡 草
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


class ReinjectLinear(nn.Module):
    """變體 A:單一 Linear(d, d) zero-init
    zero-init 讓 step 0 的輸出 恆等於 0  因此掛上它的模型與  ln_post 
    在初始化時是同一個函數 這是與 AdaIN / AdaIN-local 的 y/B zero-init
    完全對應的作法 確保單一變因就是 有沒有重注入
    """

    def __init__(self, d_model):
        super().__init__()
        # E1 RNG 配對:skip_init 跳過 nn.Linear 的 reset_parameters 不抽任何 RNG
        # ln_post 沒有這條路徑 這裡只要動到全域 RNG 串流 同 seed 下 block 1
        # 之後的 attn/ff 權重就會整批錯開(修好前實測 260 個共有張量中 91 個不同
        # step-0 logits 最大差 4.83)配對比較就不成立
        # weight/bias 緊接著被歸零 未初始化記憶體不會被讀到
        self.proj = nn.utils.skip_init(nn.Linear, d_model, d_model)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, cond):
        return self.proj(cond)


class ReinjectMLP(nn.Module):
    #變體 B: W2(GELU(W1(cond)) 每個 norm 點兩個 Linear(d, d)
    

    def __init__(self, d_model):
        super().__init__()
        # E1 RNG 配對:同 ReinjectLinear 兩個 Linear 都以 skip_init 建立
        self.w1 = nn.utils.skip_init(nn.Linear, d_model, d_model)
        self.w2 = nn.utils.skip_init(nn.Linear, d_model, d_model)
        # w1 是整條路徑唯一需要真隨機初始化的地方(w2 zero-init 保證 step 0 輸出 0)
        # fork_rng 讓它照樣抽到可重現的值 抽完把 RNG 狀態還原 主幹不受影響
        with torch.random.fork_rng(devices=[]):
            nn.init.xavier_uniform_(self.w1.weight)
        nn.init.zeros_(self.w1.bias)
        nn.init.zeros_(self.w2.weight)
        nn.init.zeros_(self.w2.bias)

    def forward(self, cond):
        return self.w2(F.gelu(self.w1(cond)))


REINJECT_KINDS = {"linear": ReinjectLinear, "mlp": ReinjectMLP}


class Block(nn.Module):
    N_NORMS = {"post": 2, "pre": 2, "deepnorm": 2, "sandwich": 4, "none": 0}

    def __init__(self, d_model, n_heads, d_ff, norm_name, placement,
                 alpha=1.0, beta=1.0, reinject="none", target_rms=None):
        super().__init__()
        assert placement in self.N_NORMS, f"未知 placement: {placement}"
        use_wn = (norm_name == "wn")
        self.placement = placement
        self.alpha = alpha  # DeepNorm 殘差放大係數 其餘為 1.0
        self.attn = MHSA(d_model, n_heads, use_wn=use_wn, gain_vo=beta)
        self.ff = FeedForward(d_model, d_ff, use_wn=use_wn, gain=beta)
        self.norms = nn.ModuleList(
            build_norm(norm_name, d_model, target_rms)
            for _ in range(self.N_NORMS[placement])
        )
        # reinject:在每個 norm 點把 embedding 經一個(zero-init 的)投影加回殘差流
        # 與 cond_detach 一樣是 加一個旗標 預設 none 時整條路徑不存在
        # 既有 placement 的行為不變
        self.reinject = reinject
        if reinject != "none":
            assert reinject in REINJECT_KINDS, f"未知 reinject: {reinject}"
            assert placement in ("post", "deepnorm"), \
                "reinject 只在 norm 位於殘差主幹上的擺放有定義 你個B仔"
            maker = REINJECT_KINDS[reinject]
            self.reinjects = nn.ModuleList(
                maker(d_model) for _ in range(self.N_NORMS[placement])
            )

    def forward(self, x, cond, weights_out=None):
        p = self.placement
        if p in ("post", "deepnorm"):
            x = self.norms[0](self.alpha * x + self.attn(x, weights_out), cond)
            if self.reinject != "none":
                x = x + self.reinjects[0](cond)
            x = self.norms[1](self.alpha * x + self.ff(x), cond)
            if self.reinject != "none":
                x = x + self.reinjects[1](cond)
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
    COND_SOURCES = ("full", "tok", "pos", "rand")

    def __init__(self, vocab_size, seq_len, d_model=128, n_heads=4,
                 d_ff=512, n_layers=16, norm_name="ln", placement="post",
                 cond_detach=False, emb_std=0.02, reinject="none",
                 target_rms=None, cond_source="full"):
        super().__init__()
        if placement == "deepnorm":
            alpha = (2.0 * n_layers) ** 0.25
            beta = (8.0 * n_layers) ** -0.25
        else:
            alpha, beta = 1.0, 1.0

        self.cond_detach = cond_detach
        self.emb_std = emb_std

        self.reinject = reinject

        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(seq_len, d_model)
        nn.init.normal_(self.tok.weight, std=emb_std)
        nn.init.normal_(self.pos.weight, std=emb_std)

        # E2:條件圖要餵什麼 預設 full 與改動前逐位元相同
        #   full  tok + pos      內容 + 位置(原本的行為)
        #   tok   只有 token 內容 沒有位置
        #   pos   只有位置 沒有內容(pos 是可學的)
        #   rand  固定隨機 per-position 向量 有位置區分度 無語意 且不可學
        assert cond_source in self.COND_SOURCES, f"未知 cond_source: {cond_source}"
        self.cond_source = cond_source
        # 梯度分解 g_trunk = g_total - g_cond 只在 cond 是 x 的分支時成立
        #   tok / pos 的 cond 掛在 x 的上游 兩個梯度在不同節點上 相減沒有意義
        #        而 decompose_embedding_grad 的殘差是 g_total-(g_total-g_cond)-g_cond
        #   恆等於 0 驗不出這件事 所以要在這裡擋掉 不能靠殘差檢查
        #   rand 可以分解:它沒有梯度路徑 g_cond = 0、g_trunk = g_total 成立且有意義
        self.cond_decomposable = cond_source in ("full", "rand")
        if cond_source == "rand":
            # 尺度對齊 tok/pos 的初始化(emb_std)否則比較會被尺度混淆
            # fork_rng:抽完還原 RNG 狀態 不能污染主幹的權重序列(見 E1)
            with torch.random.fork_rng(devices=[]):
                rand_cond = torch.randn(seq_len, d_model) * emb_std
            self.register_buffer("cond_rand", rand_cond)

        self.blocks = nn.ModuleList(
            Block(d_model, n_heads, d_ff, norm_name, placement, alpha, beta,
                  reinject=reinject, target_rms=target_rms)
            for _ in range(n_layers)
        )
        # Pre-LN / Sandwich 的殘差主幹沒被歸一化過 輸出前需要一個最終 LN
        # Post-LN / DeepNorm 出口已經被歸一化 WN(none)
        self.final_norm = (TokenLayerNorm(d_model)
                           if placement in ("pre", "sandwich") else IdentityNorm())
        self.head = nn.Linear(d_model, vocab_size)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, tokens, record_hidden=False, weights_out=None):
        B, L = tokens.shape
        tok_out = self.tok(tokens)
        pos_out = self.pos(torch.arange(L, device=tokens.device))
        x = tok_out + pos_out

        src = self.cond_source
        if src == "full":
            base = x
        elif src == "tok":
            base = tok_out
        elif src == "pos":
            base = pos_out.unsqueeze(0).expand(B, -1, -1)
        else:  # rand
            base = self.cond_rand[:L].unsqueeze(0).expand(B, -1, -1)

        cond = base.clone().detach() if self.cond_detach else base.clone()

        hiddens = []
        if record_hidden:
            x.retain_grad()
            # detach 後 cond.requires_grad=False 直接呼叫 retain_grad() 會拋RuntimeError
            if cond.requires_grad:
                cond.retain_grad()
            hiddens.append(x)
        for blk in self.blocks:
            x = blk(x, cond, weights_out)
            if record_hidden:
                x.retain_grad()
                hiddens.append(x)

        logits = self.head(self.final_norm(x))
        return logits, hiddens, cond
