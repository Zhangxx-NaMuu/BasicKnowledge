"""Hunyuan3D 扩散变换器(DiT)的多模态生成实现

模块包含以下核心实现:
- 图文联合处理的双流Transformer架构
- 基于扩散时间步的条件调制机制
- 3D感知的位置编码方案

典型用法示例:

    # 初始化模型
    model = Hunyuan3DDiT(
        in_channels=64,
        context_in_dim=1536,
        hidden_size=1024,
        num_heads=16,
        depth=16,
        depth_single_blocks=32
    )

    # 前向传播示例
    x = torch.randn(2, 16, 64)   # 潜在表示
    t = torch.rand(2)            # 扩散时间步
    context = torch.randn(2, 77, 1536)  # 文本嵌入
    
    output = model(x, t, contexts={'main': context})
"""

import math
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
from einops import rearrange
from torch import Tensor, nn

# 使用优化后的注意力机制（如果可用）
scaled_dot_product_attention = nn.functional.scaled_dot_product_attention
if os.environ.get('USE_SAGEATTN', '0') == '1':
    try:
        from sageattention import sageattn
    except ImportError:
        raise ImportError('USE_SAGEATTN启用时需要安装"sageattention"包')
    scaled_dot_product_attention = sageattn


def attention(q: Tensor, k: Tensor, v: Tensor, **kwargs) -> Tensor:
    """带缩放的点积注意力机制，可选位置编码
    
    Args:
        q: 查询张量，形状为 (B, H, L, D)
        k: 键张量，形状为 (B, H, S, D)
        v: 值张量，形状为 (B, H, S, D)
        **kwargs: 注意力计算的额外参数
    
    Returns:
        注意力输出张量，形状为 (B, L, H*D)
    """
    x = scaled_dot_product_attention(q, k, v)
    x = rearrange(x, "B H L D -> B L (H D)")
    return x


def timestep_embedding(t: Tensor, dim: int, max_period: int = 10000, 
                      time_factor: float = 1000.0) -> Tensor:
    """生成带频率缩放的正弦时间步嵌入
    
    Args:
        t: 1D时间步张量，形状 (N,)
        dim: 输出嵌入的维度
        max_period: 频率计算的最大周期
        time_factor: 时间步值的缩放因子
    
    Returns:
        位置嵌入张量，形状 (N, dim)
    """
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        t.device
    )

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


class GELU(nn.Module):
    """使用tanh近似的Gaussian Error Linear Unit激活函数
    
    Args:
        approximate: 近似方法，'tanh' 或 'none'
    """
    def __init__(self, approximate: str = 'tanh'):
        super().__init__()
        self.approximate = approximate

    def forward(self, x: Tensor) -> Tensor:
        return nn.functional.gelu(x.contiguous(), approximate=self.approximate)


class MLPEmbedder(nn.Module):
    """用于条件信号嵌入的双层MLP
    
    Args:
        in_dim: 输入维度
        hidden_dim: 隐藏层维度
    """
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(torch.nn.Module):
    """均方根层归一化
    
    Args:
        dim: 归一化维度
    """
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class QKNorm(torch.nn.Module):
    """注意力机制中查询和键的归一化
    
    Args:
        dim: 查询/键的向量维度
    """
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class SelfAttention(nn.Module):
    """带QK归一化的多头自注意力
    
    Args:
        dim: 输入维度
        num_heads: 注意力头数
        qkv_bias: 是否在qkv投影中使用偏置
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, pe: Tensor) -> Tensor:
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        x = attention(q, k, v, pe=pe)
        x = self.proj(x)
        return x


@dataclass
class ModulationOut:
    """调制参数输出容器"""
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation(nn.Module):
    """基于条件向量的动态特征调制
    
    Args:
        dim: 调制参数的维度
        double: 是否生成两组调制参数
    """
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec: Tensor) -> Tuple[ModulationOut, Optional[ModulationOut]]:
        out = self.lin(nn.functional.silu(vec))[:, None, :]
        out = out.chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )


class DoubleStreamBlock(nn.Module):
    """图文交互的双流Transformer块
    
    Args:
        hidden_size: 隐藏层维度
        num_heads: 注意力头数
        mlp_ratio: MLP隐藏层维度与hidden_size的比例
        qkv_bias: 是否在qkv投影中使用偏置
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool = False,
    ):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        
        # 图像流组件
        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)
        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        # 文本流组件
        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)
        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

    def forward(self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor) -> Tuple[Tensor, Tensor]:
        # 图像流处理
        img_mod1, img_mod2 = self.img_mod(vec)
        img_modulated = (1 + img_mod1.scale) * self.img_norm1(img) + img_mod1.shift
        img_q, img_k, img_v = rearrange(self.img_attn.qkv(img_modulated), 
                                      "B L (K H D) -> K B H L D", K=3, H=self.num_heads)[:3]
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # 文本流处理
        txt_mod1, txt_mod2 = self.txt_mod(vec)
        txt_modulated = (1 + txt_mod1.scale) * self.txt_norm1(txt) + txt_mod1.shift
        txt_q, txt_k, txt_v = rearrange(self.txt_attn.qkv(txt_modulated), 
                                      "B L (K H D) -> K B H L D", K=3, H=self.num_heads)[:3]
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        # 跨模态注意力
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)
        attn = attention(q, k, v, pe=pe)
        
        # 分割并处理输出
        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1]:]

        # 带残差连接的特征更新
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img = img + img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)

        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)
        return img, txt


class SingleStreamBlock(nn.Module):
    """带并行注意力与MLP路径的单流Transformer块
    
    Args:
        hidden_size: 隐藏层维度
        num_heads: 注意力头数
        mlp_ratio: MLP隐藏层维度与hidden_size的比例
        qk_scale: 注意力logits的可选缩放因子
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: Optional[float] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)
        self.norm = QKNorm(head_dim)
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp_act = GELU(approximate="tanh")
        self.modulation = Modulation(hidden_size, double=False)

    def forward(self, x: Tensor, vec: Tensor, pe: Tensor) -> Tensor:
        mod, _ = self.modulation(vec)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        
        # 并行处理注意力与MLP路径
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        
        attn = attention(q, k, v, pe=pe)
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        return x + mod.gate * output


class LastLayer(nn.Module):
    """带自适应调制的最终投影层
    
    Args:
        hidden_size: 输入维度
        patch_size: 输出块的空间尺寸（未使用）
        out_channels: 输出通道维度
    """
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        return self.linear(x)


class Hunyuan3DDiT(nn.Module):
    """Hunyuan3D扩散变换器主模型
    
    Args:
        in_channels: 输入潜在通道数
        context_in_dim: 文本嵌入维度
        hidden_size: 模型隐藏维度
        mlp_ratio: MLP扩展比例
        num_heads: 注意力头数
        depth: 双流块数量
        depth_single_blocks: 单流块数量
        axes_dim: 各轴的位置编码维度
        theta: 位置编码频率基数
        qkv_bias: 是否在QKV投影中使用偏置
        time_factor: 时间步嵌入缩放因子
        guidance_embed: 是否使用引导嵌入
        ckpt_path: 预训练权重的检查点路径
    
    Example:
        >>> model = Hunyuan3DDiT(
        ...     in_channels=64,
        ...     context_in_dim=1536,
        ...     hidden_size=1024,
        ...     num_heads=16,
        ...     depth=16,
        ...     depth_single_blocks=32
        ... )
        >>> x = torch.randn(2, 16, 64)  # 潜在编码批次
        >>> t = torch.rand(2)           # 随机时间步
        >>> context = torch.randn(2, 77, 1536)  # 文本嵌入
        >>> output = model(x, t, contexts={'main': context})
        >>> print(output.shape)
        torch.Size([2, 16, 64])
    """
    def __init__(
        self,
        in_channels: int = 64,
        context_in_dim: int = 1536,
        hidden_size: int = 1024,
        mlp_ratio: float = 4.0,
        num_heads: int = 16,
        depth: int = 16,
        depth_single_blocks: int = 32,
        axes_dim: List[int] = [64],
        theta: int = 10_000,
        qkv_bias: bool = True,
        time_factor: float = 1000,
        guidance_embed: bool = False,
        ckpt_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        # 维度约束验证
        if hidden_size % num_heads != 0:
            raise ValueError(f"隐藏维度{hidden_size}必须能被注意力头数{num_heads}整除")
        if sum(axes_dim) != hidden_size // num_heads:
            raise ValueError(f"轴维度{axes_dim}之和必须等于{hidden_size//num_heads}")

        # 初始化核心组件
        self.latent_in = nn.Linear(in_channels, hidden_size)
        self.time_in = MLPEmbedder(256, hidden_size)
        self.cond_in = nn.Linear(context_in_dim, hidden_size)
        self.guidance_in = MLPEmbedder(256, hidden_size) if guidance_embed else nn.Identity()

        # 构建Transformer块
        self.double_blocks = nn.ModuleList([
            DoubleStreamBlock(hidden_size, num_heads, mlp_ratio, qkv_bias)
            for _ in range(depth)
        ])
        self.single_blocks = nn.ModuleList([
            SingleStreamBlock(hidden_size, num_heads, mlp_ratio)
            for _ in range(depth_single_blocks)
        ])

        # 最终输出投影
        self.final_layer = LastLayer(hidden_size, 1, in_channels)

        # 加载预训练权重
        if ckpt_path:
            self._load_pretrained_weights(ckpt_path)

    def _load_pretrained_weights(self, ckpt_path: str):
        """从检查点加载预训练权重"""
        print(f'正在从{ckpt_path}加载预训练权重')
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get('state_dict', ckpt)  # 处理deepspeed检查点
        
        # 适配检查点键名
        final_state_dict = {}
        for k, v in state_dict.items():
            new_k = k.replace('_forward_module.', '').replace('model.', '')
            final_state_dict[new_k] = v
        
        load_info = self.load_state_dict(final_state_dict, strict=False)
        print(f'Unexpected keys: {load_info.unexpected_keys}')
        print(f'Missing keys: {load_info.missing_keys}')

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        contexts: dict,
        **kwargs,
    ) -> Tensor:
        """扩散变换器的前向传播
        
        Args:
            x: 输入潜在张量，形状 (B, L, C)
            t: 时间步张量，形状 (B,)
            contexts: 包含'main'键下文本嵌入的字典
            **kwargs: 包含可选引导强度的额外参数
        
        Returns:
            与输入同形状的预测噪声张量
        """
        # 处理输入条件
        cond = self.cond_in(contexts['main'])
        latent = self.latent_in(x)
        
        # 时间与引导嵌入
        vec = self.time_in(timestep_embedding(t, 256, self.time_factor).to(x.dtype))
        if self.guidance_embed:
            guidance = kwargs['guidance']
            vec += self.guidance_in(timestep_embedding(guidance, 256, self.time_factor))

        # 双流处理
        for block in self.double_blocks:
            latent, cond = block(img=latent, txt=cond, vec=vec, pe=None)
        
        # 单流处理
        combined = torch.cat((cond, latent), 1)
        for block in self.single_blocks:
            combined = block(combined, vec=vec, pe=None)
        
        # 最终投影
        return self.final_layer(combined[:, cond.shape[1]:], vec)