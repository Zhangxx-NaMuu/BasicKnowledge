import torch
import torch.nn as nn
from typing import Tuple, List, Union, Optional
import torch.nn.functional as F


class FourierEmbedder(nn.Module):
    """
    ```
    傅里叶变换(正弦/余弦位置)嵌入模块。给定形状为 [n_batch, ..., c_dim] 的输入张量 `x`，
    它将 `x[..., i]` 的每个特征维度转换为如下形式：
    
        [
            sin(x[..., i]),
            sin(f_1*x[..., i]),
            sin(f_2*x[..., i]),
            ...
            sin(f_N * x[..., i]),
            cos(x[..., i]),
            cos(f_1*x[..., i]),
            cos(f_2*x[..., i]),
            ...
            cos(f_N * x[..., i]),
            x[..., i]     # 仅当 include_input 为 True 时保留
        ]
    其中 f_i 表示频率。
    
    

    频率空间默认为 [0/num_freqs, 1/num_freqs, ..., (num_freqs-1)/num_freqs]。
    若 `logspace` 为 True，则频率按对数空间排列：f_i = [2^(0/num_freqs), 2^(1/num_freqs), ..., 2^((num_freqs-1)/num_freqs)]；
    否则，频率在 [1.0, 2^(num_freqs-1)] 范围内线性均匀分布。
    ```
    Args:
        num_freqs (int): 频率数量,默认为6;
        
        logspace (bool): 是否使用对数空间频率。若为True，频率为 2^(i/num_freqs)；否则线性间隔，默认为True；
        
        input_dim (int): 输入维度，默认为3；
        include_input (bool): 是否在输出中包含原始输入，默认为True；
        include_pi (bool): 是否将频率乘以π，默认为True。

    Attributes:
        frequencies (torch.Tensor): 频率张量。若 `logspace` 为True，则频率按指数间隔；否则线性间隔。
        out_dim (int): 嵌入后的维度。若 `include_input` 为True，则为 input_dim * (num_freqs*2 +1)；否则为 input_dim * num_freqs*2。
    """

    def __init__(self,
                 num_freqs: int = 6,
                 logspace: bool = True,
                 input_dim: int = 3,
                 include_input: bool = True,
                 include_pi: bool = True) -> None:
        """初始化方法"""
        super().__init__()

        # 生成频率
        if logspace:
            frequencies = 2.0 ** torch.arange(
                num_freqs,
                dtype=torch.float32
            )
        else:
            frequencies = torch.linspace(
                1.0,
                2.0 ** (num_freqs - 1),
                num_freqs,
                dtype=torch.float32
            )

        # 可选：将所有频率乘以π
        if include_pi:
            frequencies *= torch.pi

        # 注册为不持久化的缓冲区（不参与模型保存）
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.include_input = include_input
        self.num_freqs = num_freqs
        self.out_dim = self.get_dims(input_dim)

    def get_dims(self, input_dim: int) -> int:
        """计算输出维度"""
        temp = 1 if self.include_input or self.num_freqs == 0 else 0
        out_dim = input_dim * (self.num_freqs * 2 + temp)
        return out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量，形状为 [..., dim]
            
        Returns:
            embedding: 嵌入后的张量，形状为 [..., dim * (num_freqs*2 + temp)]，
                其中 temp 为1（若包含输入）或0。
        """
        if self.num_freqs > 0:
            # 计算 x 与频率的外积并展平
            embed = (x[..., None].contiguous() * self.frequencies).view(*x.shape[:-1], -1)
            # 按需拼接输入、正弦项、余弦项
            if self.include_input:
                return torch.cat((x, embed.sin(), embed.cos()), dim=-1)
            else:
                return torch.cat((embed.sin(), embed.cos()), dim=-1)
        else:
            # 无频率时直接返回原输入
            return x
        
        
        

class DropPath(nn.Module):
    """随机深度（Stochastic Depth）模块，用于在残差块的主路径上随机丢弃样本路径。
    
    该模块通过以概率 `drop_prob` 将输入张量置零（跳过当前残差块），同时根据 `scale_by_keep` 
    决定是否缩放输出值以保持期望不变。常用于正则化深层网络（如ResNet、Vision Transformer）。
    
    Notes:
    
        它与作者为 EfficientNet 等网络创建的 DropConnect 实现类似，但原来的名称具有误导性，
        因为 “Drop Connect” 在另一篇论文中是一种不同形式的丢弃技术。
        作者选择将层和参数名称更改为 “drop path”，
        而不是将 DropConnect 作为层名并使用 “survival rate（生存概率）” 作为参数。
        [https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956]
        

    Args:
        drop_prob (float): 路径丢弃概率，取值范围 [0, 1)，默认为 0（不丢弃）。
        scale_by_keep (bool): 若为 True，保留路径时会进行缩放补偿（除以 `1 - drop_prob`），
            以保持输出的期望值不变，默认为 True。

    Attributes:
        drop_prob (float): 继承自 Args 的路径丢弃概率。
        scale_by_keep (bool): 继承自 Args 的缩放开关。

    Example:
        >>> x = torch.randn(2, 3, 16, 16)
        >>> drop_path = DropPath(drop_prob=0.2)
        >>> train_output = drop_path(x)  # 训练时随机丢弃路径
        >>> drop_path.eval()
        >>> eval_output = drop_path(x)   # 推理时直接返回原值
    """

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        """初始化方法，配置丢弃概率和缩放开关"""
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播，训练时随机丢弃路径，推理时直接返回输入。

        具体实现逻辑：
        1. 若 `drop_prob=0` 或处于推理模式，直接返回输入。
        2. 生成与输入张量 `x` 的 batch 维度对齐的随机二值掩码。
        3. 根据 `scale_by_keep` 决定是否对保留路径的样本进行缩放补偿。

        Args:
            x (torch.Tensor): 输入张量，形状为 [batch_size, ...]

        Returns:
            torch.Tensor: 输出张量，训练时可能被部分置零，形状与输入一致。
        """
        # 若无需丢弃或处于推理模式，直接返回原值
        if self.drop_prob == 0. or not self.training:
            return x

        # 计算保留概率并生成随机二值掩码
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # 适配不同维度（Conv2D/3D, Linear等）
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)

        # 缩放补偿：保持输出的期望值 E[output] = x（仅训练时生效）
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)

        return x * random_tensor  # 随机置零部分样本的路径

    def extra_repr(self) -> str:
        """用于打印模块的附加信息"""
        return f'drop_prob={round(self.drop_prob, 3):0.3f}'



class MLP(nn.Module):
    """多层感知机（MLP）模块，包含扩展投影、激活函数、收缩投影和可选的 DropPath 正则化。

    Args:
        width (int): 输入特征维度。
        output_width (int, optional): 输出特征维度，默认为 None（与输入相同）。
        drop_path_rate (float): DropPath 的路径丢弃概率，默认为 0.0（不启用）。

    Shape:
        - 输入 x: (..., width)
        - 输出: (..., output_width or width)
    """

    def __init__(self, *, width: int, output_width: int = None, drop_path_rate: float = 0.0):
        super().__init__()
        self.width = width
        # 扩展层：将维度扩展为 4 倍以增加非线性能力
        self.c_fc = nn.Linear(width, width * 4)
        # 收缩层：将维度投影回目标输出维度（默认与输入相同）
        self.c_proj = nn.Linear(width * 4, output_width if output_width is not None else width)
        self.gelu = nn.GELU()  # GELU 激活函数
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播流程：扩展 -> 激活 -> 收缩 -> DropPath（若启用）"""
        x = self.c_fc(x)      # 扩展维度
        x = self.gelu(x)      # 激活函数
        x = self.c_proj(x)    # 投影回目标维度
        x = self.drop_path(x) # 应用 DropPath（训练时随机丢弃路径）
        return x

class QKVMultiheadCrossAttention(nn.Module):
    """基于查询（Query）、键值对（Key-Value）的多头交叉注意力计算模块。
    
    通过将输入的 `q` 和 `kv` 分割为多头，应用缩放点积注意力（Scaled Dot-Product Attention），
    并可选对 Q/K 进行归一化处理。

    Args:
        heads (int): 注意力头的数量。
        n_data (int, optional): 键值对数据的数量（上下文长度），默认为 None。
        width (int, optional): 输入特征的维度，默认为 None。
        qk_norm (bool): 是否对 Q/K 进行层归一化，默认为 False。
        norm_layer (nn.Module): 归一化层类型，默认为 `nn.LayerNorm`。

    Attributes:
        heads (int): 继承自 Args 的注意力头数量。
        q_norm (nn.Module): 查询向量的归一化层（若启用 qk_norm）。
        k_norm (nn.Module): 键向量的归一化层（若启用 qk_norm）。

    Shape:
        - 输入 q: (bs, n_ctx, width)
        - 输入 kv: (bs, n_data, width * 2)  # 包含键和值拼接后的张量
        - 输出: (bs, n_ctx, width)
    """

    def __init__(self, *, heads: int, n_data: Optional[int] = None, width=None, qk_norm=False, norm_layer=nn.LayerNorm):
        super().__init__()
        self.heads = heads
        self.n_data = n_data
        # 初始化 Q/K 归一化层（若启用）
        self.q_norm = norm_layer(width // heads, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(width // heads, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()

    def forward(self, q, kv):
        # 分割多头并计算注意力
        bs, n_ctx, _ = q.shape
        bs, n_data, width = kv.shape
        attn_ch = width // self.heads // 2  # 计算每个注意力头的通道数
        q = q.view(bs, n_ctx, self.heads, -1)
        kv = kv.view(bs, n_data, self.heads, -1)
        k, v = torch.split(kv, attn_ch, dim=-1)  # 分割键和值

        # 归一化处理
        q = self.q_norm(q)
        k = self.k_norm(k)

        # 重排维度并计算注意力
        from einops import rearrange
        q, k, v = map(lambda t: rearrange(t, 'b n h d -> b h n d', h=self.heads), (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(bs, n_ctx, -1)
        return out


class MultiheadCrossAttention(nn.Module):
    """多头交叉注意力模块，包含线性投影和注意力计算。

    将输入 `x` 和 `data` 分别投影为 Q 和 K/V，并通过 `QKVMultiheadCrossAttention` 计算交叉注意力。

    Args:
        width (int): 输入/输出特征维度。
        heads (int): 注意力头的数量。
        qkv_bias (bool): 是否在 Q/K/V 投影中添加偏置项，默认为 True。
        n_data (int, optional): 键值对数据的数量，默认为 None。
        data_width (int, optional): 输入数据 `data` 的特征维度，默认为 None（同 width）。
        norm_layer (nn.Module): 归一化层类型，默认为 `nn.LayerNorm`。
        qk_norm (bool): 是否对 Q/K 进行归一化，默认为 False。

    Shape:
        - 输入 x: (bs, n_ctx, width)
        - 输入 data: (bs, n_data, data_width)
        - 输出: (bs, n_ctx, width)
    """

    def __init__(self, *, width: int, heads: int, qkv_bias: bool = True, n_data: Optional[int] = None, 
                 data_width: Optional[int] = None, norm_layer=nn.LayerNorm, qk_norm: bool = False):
        super().__init__()
        self.n_data = n_data
        self.width = width
        self.heads = heads
        self.data_width = width if data_width is None else data_width

        # 初始化 Q/K/V 投影层
        self.c_q = nn.Linear(width, width, bias=qkv_bias)  # 查询投影
        self.c_kv = nn.Linear(self.data_width, width * 2, bias=qkv_bias)  # 键值投影
        self.c_proj = nn.Linear(width, width)  # 输出投影

        # 注意力计算模块
        self.attention = QKVMultiheadCrossAttention(
            heads=heads, n_data=n_data, width=width, norm_layer=norm_layer, qk_norm=qk_norm
        )

    def forward(self, x, data):
        x = self.c_q(x)  # 投影查询向量
        data = self.c_kv(data)  # 投影键值对
        x = self.attention(x, data)  # 计算交叉注意力
        x = self.c_proj(x)  # 投影回原始维度
        return x


class ResidualCrossAttentionBlock(nn.Module):
    """残差交叉注意力块，包含多头交叉注意力和 MLP 子模块。

    结构：LN -> Cross-Attention -> Add -> LN -> MLP -> Add

    Args:
        n_data (int, optional): 键值对数据的数量，默认为 None。
        width (int): 输入特征维度。
        heads (int): 注意力头的数量。
        data_width (int, optional): 输入数据 `data` 的特征维度，默认为 None（同 width）。
        qkv_bias (bool): 是否在 Q/K/V 投影中添加偏置项，默认为 True。
        norm_layer (nn.Module): 归一化层类型，默认为 `nn.LayerNorm`。
        qk_norm (bool): 是否对 Q/K 进行归一化，默认为 False。

    Shape:
        - 输入 x: (bs, n_ctx, width)
        - 输入 data: (bs, n_data, data_width)
        - 输出: (bs, n_ctx, width)
    """

    def __init__(self, *, n_data: Optional[int] = None, width: int, heads: int, data_width: Optional[int] = None,
                 qkv_bias: bool = True, norm_layer=nn.LayerNorm, qk_norm: bool = False):
        super().__init__()
        if data_width is None:
            data_width = width

        # 初始化子模块
        self.attn = MultiheadCrossAttention(
            n_data=n_data, width=width, heads=heads, data_width=data_width,
            qkv_bias=qkv_bias, norm_layer=norm_layer, qk_norm=qk_norm
        )
        self.ln_1 = norm_layer(width, eps=1e-6)  # 输入归一化
        self.ln_2 = norm_layer(data_width, eps=1e-6)  # 数据归一化
        self.ln_3 = norm_layer(width, eps=1e-6)  # MLP 前归一化
        self.mlp = MLP(width=width)  # 多层感知机

    def forward(self, x: torch.Tensor, data: torch.Tensor):
        # 残差连接：交叉注意力
        x = x + self.attn(self.ln_1(x), self.ln_2(data))
        # 残差连接：MLP
        x = x + self.mlp(self.ln_3(x))
        return x


class QKVMultiheadAttention(nn.Module):
    """基于 QKV 拼接的多头自注意力计算模块。

    将输入的拼接 QKV 张量分割为独立的 Q/K/V，并应用缩放点积注意力。

    Args:
        heads (int): 注意力头数量。
        n_ctx (int): 上下文长度（序列长度）。
        width (int, optional): 输入特征维度，默认为 None。
        qk_norm (bool): 是否对 Q/K 进行归一化，默认为 False。
        norm_layer (nn.Module): 归一化层类型，默认为 `nn.LayerNorm`。

    Shape:
        - 输入 qkv: (bs, n_ctx, width * 3)  # Q/K/V 拼接后的张量
        - 输出: (bs, n_ctx, width)
    """

    def __init__(self, *, heads: int, n_ctx: int, width=None, qk_norm=False, norm_layer=nn.LayerNorm):
        super().__init__()
        self.heads = heads
        self.n_ctx = n_ctx
        self.q_norm = norm_layer(width // heads, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(width // heads, eps=1e-6) if qk_norm else nn.Identity()

    def forward(self, qkv):
        bs, n_ctx, width = qkv.shape
        attn_ch = width // self.heads // 3  # 计算每个注意力头的通道数
        qkv = qkv.view(bs, n_ctx, self.heads, -1)
        q, k, v = torch.split(qkv, attn_ch, dim=-1)  # 分割 Q/K/V

        # 归一化处理
        q = self.q_norm(q)
        k = self.k_norm(k)

        # 重排维度并计算注意力
        from einops import rearrange
        q, k, v = map(lambda t: rearrange(t, 'b n h d -> b h n d', h=self.heads), (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(bs, n_ctx, -1)
        return out


class MultiheadAttention(nn.Module):
    """多头自注意力模块，包含 QKV 投影和注意力计算。

    Args:
        n_ctx (int): 上下文长度（序列长度）。
        width (int): 输入/输出特征维度。
        heads (int): 注意力头数量。
        qkv_bias (bool): 是否在 QKV 投影中添加偏置项，默认为 True。
        norm_layer (nn.Module): 归一化层类型，默认为 `nn.LayerNorm`。
        qk_norm (bool): 是否对 Q/K 进行归一化，默认为 False。
        drop_path_rate (float): DropPath 的丢弃概率，默认为 0.0（不启用）。

    Shape:
        - 输入 x: (bs, n_ctx, width)
        - 输出: (bs, n_ctx, width)
    """

    def __init__(self, *, n_ctx: int, width: int, heads: int, qkv_bias: bool, 
                 norm_layer=nn.LayerNorm, qk_norm: bool = False, drop_path_rate: float = 0.0):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.heads = heads

        # 初始化投影层
        self.c_qkv = nn.Linear(width, width * 3, bias=qkv_bias)  # QKV 拼接投影
        self.c_proj = nn.Linear(width, width)  # 输出投影
        self.attention = QKVMultiheadAttention(  # 注意力计算模块
            heads=heads, n_ctx=n_ctx, width=width, norm_layer=norm_layer, qk_norm=qk_norm
        )
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        x = self.c_qkv(x)  # 投影 QKV
        x = self.attention(x)  # 计算自注意力
        x = self.drop_path(self.c_proj(x))  # 输出投影 + DropPath
        return x


class ResidualAttentionBlock(nn.Module):
    """残差自注意力块，包含多头自注意力和 MLP 子模块。

    结构：LN -> Self-Attention -> Add -> LN -> MLP -> Add

    Args:
        n_ctx (int): 上下文长度（序列长度）。
        width (int): 输入特征维度。
        heads (int): 注意力头数量。
        qkv_bias (bool): 是否在 QKV 投影中添加偏置项，默认为 True。
        norm_layer (nn.Module): 归一化层类型，默认为 `nn.LayerNorm`。
        qk_norm (bool): 是否对 Q/K 进行归一化，默认为 False。
        drop_path_rate (float): DropPath 的丢弃概率，默认为 0.0。

    Shape:
        - 输入 x: (bs, n_ctx, width)
        - 输出: (bs, n_ctx, width)
    """

    def __init__(self, *, n_ctx: int, width: int, heads: int, qkv_bias: bool = True,
                 norm_layer=nn.LayerNorm, qk_norm: bool = False, drop_path_rate: float = 0.0):
        super().__init__()
        self.attn = MultiheadAttention(  # 自注意力模块
            n_ctx=n_ctx, width=width, heads=heads, qkv_bias=qkv_bias,
            norm_layer=norm_layer, qk_norm=qk_norm, drop_path_rate=drop_path_rate
        )
        self.ln_1 = norm_layer(width, eps=1e-6)  # 自注意力前归一化
        self.mlp = MLP(width=width, drop_path_rate=drop_path_rate)  # MLP 模块
        self.ln_2 = norm_layer(width, eps=1e-6)  # MLP 前归一化

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln_1(x))  # 残差连接：自注意力
        x = x + self.mlp(self.ln_2(x))    # 残差连接：MLP
        return x


class Transformer(nn.Module):
    """Transformer 模型，由多层 `ResidualAttentionBlock` 堆叠而成。

    Args:
        n_ctx (int): 上下文长度（序列长度）。
        width (int): 输入特征维度。
        layers (int): 残差注意力块的层数。
        heads (int): 注意力头数量。
        qkv_bias (bool): 是否在 QKV 投影中添加偏置项，默认为 True。
        norm_layer (nn.Module): 归一化层类型，默认为 `nn.LayerNorm`。
        qk_norm (bool): 是否对 Q/K 进行归一化，默认为 False。
        drop_path_rate (float): DropPath 的丢弃概率，默认为 0.0。

    Shape:
        - 输入 x: (bs, n_ctx, width)
        - 输出: (bs, n_ctx, width)
    """

    def __init__(self, *, n_ctx: int, width: int, layers: int, heads: int, qkv_bias: bool = True,
                 norm_layer=nn.LayerNorm, qk_norm: bool = False, drop_path_rate: float = 0.0):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.layers = layers
        # 初始化多层残差块
        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                n_ctx=n_ctx, width=width, heads=heads, qkv_bias=qkv_bias,
                norm_layer=norm_layer, qk_norm=qk_norm, drop_path_rate=drop_path_rate
            )
            for _ in range(layers)
        ])

    def forward(self, x: torch.Tensor):
        for block in self.resblocks:
            x = block(x)  # 逐层计算
        return x



class CrossAttentionDecoder(nn.Module):
    """交叉注意力解码器模块，用于通过潜在变量（Latents）增强查询（Queries）的特征表示。

    该模块将输入查询通过傅里叶嵌入编码后，与潜在变量进行交叉注意力交互，最终生成目标输出（如分类概率）。

    Args:
        num_latents (int): 潜在变量的数量（即每个样本的上下文标记数）。
        out_channels (int): 输出通道数（如分类类别数）。
        fourier_embedder (FourierEmbedder): 傅里叶特征嵌入器，用于编码输入查询。
        width (int): 特征投影后的维度（注意力模块的隐藏层宽度）。
        heads (int): 注意力头的数量。
        qkv_bias (bool): 是否在 Q/K/V 投影中添加偏置项，默认为 True。
        qk_norm (bool): 是否对 Q/K 进行层归一化，默认为 False。
        label_type (str): 输出标签类型，支持 "binary"（二分类）或 "multiclass"（多分类），默认为 "binary"。

    Attributes:
        query_proj (nn.Linear): 将傅里叶嵌入后的查询投影到指定宽度的线性层。
        cross_attn_decoder (ResidualCrossAttentionBlock): 残差交叉注意力块。
        ln_post (nn.LayerNorm): 输出前的层归一化。
        output_proj (nn.Linear): 最终输出投影层。

    Shape:
        - 输入 queries: (bs, num_queries, query_dim)
        - 输入 latents: (bs, num_latents, latent_dim)
        - 输出 occ: (bs, num_queries, out_channels)
    """

    def __init__(
        self,
        *,
        num_latents: int,
        out_channels: int,
        fourier_embedder: FourierEmbedder,
        width: int,
        heads: int,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        label_type: str = "binary"
    ):
        super().__init__()
        self.fourier_embedder = fourier_embedder

        # 将傅里叶嵌入后的查询投影到指定维度（width）
        self.query_proj = nn.Linear(self.fourier_embedder.out_dim, width)

        # 残差交叉注意力模块（处理查询与潜在变量的交互）
        self.cross_attn_decoder = ResidualCrossAttentionBlock(
            n_data=num_latents,
            width=width,
            heads=heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm
        )

        # 后处理层
        self.ln_post = nn.LayerNorm(width)  # 输出归一化
        self.output_proj = nn.Linear(width, out_channels)  # 输出投影
        self.label_type = label_type  # 标签类型（暂未直接使用，可扩展损失函数）

    def forward(self, queries: torch.FloatTensor, latents: torch.FloatTensor) -> torch.FloatTensor:
        """前向传播流程：傅里叶嵌入 -> 投影 -> 交叉注意力 -> 归一化 -> 输出投影。

        Args:
            queries (torch.FloatTensor): 输入查询张量，形状 (bs, num_queries, query_dim)
            latents (torch.FloatTensor): 潜在变量张量，形状 (bs, num_latents, latent_dim)

        Returns:
            torch.FloatTensor: 输出张量，形状 (bs, num_queries, out_channels)
        """
        # 傅里叶嵌入 + 投影（保持与潜在变量相同的数据类型）
        queries = self.query_proj(self.fourier_embedder(queries).to(latents.dtype))
        
        # 残差交叉注意力交互
        x = self.cross_attn_decoder(queries, latents)
        
        # 后处理与输出
        x = self.ln_post(x)
        occ = self.output_proj(x)  # 输出如占据概率、分类logits等
        
        return occ