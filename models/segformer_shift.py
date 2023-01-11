import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from torch.nn.functional import _in_projection
from torch.nn.functional import _in_projection_packed
from torch.nn.functional import _mha_shape_check
from torch.nn.functional import pad
from torch.overrides import handle_torch_function
from torch.overrides import has_torch_function
from models.segformer_utils.logger import get_root_logger
from mmcv.runner import load_checkpoint
from kornia.contrib import extract_tensor_patches, combine_tensor_patches
from typing import Optional, Tuple
from munch import Munch
import torch
from torch import Tensor
torch.manual_seed(12345)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def randn_sampling(maxint, sample_size, batch_size):
    return torch.randint(maxint, size = (batch_size, sample_size, 2))


def collect_samples(feats, pxy, batch_size):
    return torch.stack([feats[i, :, pxy[i][:, 0], pxy[i][:, 1]] for i in range(batch_size)], dim = 0)


def collect_samples_faster(feats, pxy, batch_size):
    n, c, h, w = feats.size()
    feats = feats.view(n, c, -1).permute(1, 0, 2).reshape(c, -1)  # [n, c, h, w] -> [n, c, hw] -> [c, nhw]
    pxy = ((torch.arange(n).long().to(pxy.device) * h * w).view(n, 1) + pxy[:, :, 0] * h + pxy[:, :, 1]).view(
        -1)  # [n, m, 2] -> [nm]
    return (feats[:, pxy]).view(c, n, -1).permute(1, 0, 2)


def collect_positions(batch_size, N):
    all_positions = [[i, j] for i in range(N) for j in range(N)]
    pts = torch.tensor(all_positions)  # [N*N, 2]
    pts_norm = pts.repeat(batch_size, 1, 1)  # [B, N*N, 2]
    rnd = torch.stack([torch.randperm(N * N) for _ in range(batch_size)], dim = 0)  # [B, N*N]
    pts_rnd = torch.stack([pts_norm[idx, r] for idx, r in enumerate(rnd)], dim = 0)  # [B, N*N, 2]
    return pts_norm, pts_rnd


class DenseRelativeLoc(nn.Module):
    def __init__(self, in_dim, out_dim = 2, sample_size = 32, drloc_mode = "l1", use_abs = False):
        super(DenseRelativeLoc, self).__init__()
        self.sample_size = sample_size
        self.in_dim = in_dim
        self.drloc_mode = drloc_mode
        self.use_abs = use_abs

        if self.drloc_mode == "l1":
            self.out_dim = out_dim
            self.layers = nn.Sequential(
                    nn.Linear(in_dim * 2, 512),
                    nn.ReLU(),
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Linear(512, self.out_dim)
            )
        elif self.drloc_mode in ["ce", "cbr"]:
            self.out_dim = out_dim if self.use_abs else out_dim * 2 - 1
            self.layers = nn.Sequential(
                    nn.Linear(in_dim * 2, 512),
                    nn.ReLU(),
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Linear(512, 512)
            )
            self.unshared = nn.ModuleList()
            for _ in range(2):
                self.unshared.append(nn.Linear(512, self.out_dim))
        else:
            raise NotImplementedError("We only support l1, ce and cbr now.")

    def forward_features(self, x, mode = "part"):
        # x, feature map with shape: [B, C, H, W]
        B, C, H, W = x.size()

        if mode == "part":
            pxs = randn_sampling(H, self.sample_size, B).detach()
            pys = randn_sampling(H, self.sample_size, B).detach()

            deltaxy = (pxs - pys).float().to(x.device)  # [B, sample_size, 2]

            ptsx = collect_samples_faster(x, pxs, B).transpose(1, 2).contiguous()  # [B, sample_size, C]
            ptsy = collect_samples_faster(x, pys, B).transpose(1, 2).contiguous()  # [B, sample_size, C]
        else:
            pts_norm, pts_rnd = collect_positions(B, H)
            ptsx = x.view(B, C, -1).transpose(1, 2).contiguous()  # [B, H*W, C]
            ptsy = collect_samples(x, pts_rnd, B).transpose(1, 2).contiguous()  # [B, H*W, C]

            deltaxy = (pts_norm - pts_rnd).float().to(x.device)  # [B, H*W, 2]

        pred_feats = self.layers(torch.cat([ptsx, ptsy], dim = 2))
        return pred_feats, deltaxy, H

    def forward(self, x, normalize = False):
        pred_feats, deltaxy, H = self.forward_features(x)
        deltaxy = deltaxy.view(-1, 2)  # [B*sample_size, 2]

        if self.use_abs:
            deltaxy = torch.abs(deltaxy)
            if normalize:
                deltaxy /= float(H - 1)
        else:
            deltaxy += (H - 1)
            if normalize:
                deltaxy /= float(2 * (H - 1))

        if self.drloc_mode == "l1":
            predxy = pred_feats.view(-1, self.out_dim)  # [B*sample_size, Output_size]
        else:
            predx, predy = self.unshared[0](pred_feats), self.unshared[1](pred_feats)
            predx = predx.view(-1, self.out_dim)  # [B*sample_size, Output_size]
            predy = predy.view(-1, self.out_dim)  # [B*sample_size, Output_size]
            predxy = torch.stack([predx, predy], dim = 2)  # [B*sample_size, Output_size, 2]
        return predxy, deltaxy

    def flops(self):
        fps = self.in_dim * 2 * 512 * self.sample_size
        fps += 512 * 512 * self.sample_size
        fps += 512 * self.out_dim * self.sample_size
        if self.drloc_mode in ["ce", "cbr"]:
            fps += 512 * 512 * self.sample_size
            fps += 512 * self.out_dim * self.sample_size
        return fps

def scaled_dot_product_attention(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        temperature: Tensor,
        attn_mask: Optional[Tensor] = None,
        dropout_p: float = 0.0,
) -> Tuple[Tensor, Tensor]:
    B, Nt, E = q.shape
    q = q / temperature
    # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
    attn = torch.bmm(q, k.transpose(-2, -1))
    if attn_mask is not None:
        attn += attn_mask
    attn = F.softmax(attn, dim = -1)
    if dropout_p > 0.0:
        attn = F.dropout(attn, p = dropout_p)
    # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
    output = torch.bmm(attn, v)
    return output, attn


def multi_head_attention_forward(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        temperature: Tensor,
        embed_dim_to_check: int,
        num_heads: int,
        in_proj_weight: Tensor,
        in_proj_bias: Optional[Tensor],
        bias_k: Optional[Tensor],
        bias_v: Optional[Tensor],
        add_zero_attn: bool,
        dropout_p: float,
        out_proj_weight: Tensor,
        out_proj_bias: Optional[Tensor],
        training: bool = True,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        use_separate_proj_weight: bool = False,
        q_proj_weight: Optional[Tensor] = None,
        k_proj_weight: Optional[Tensor] = None,
        v_proj_weight: Optional[Tensor] = None,
        static_k: Optional[Tensor] = None,
        static_v: Optional[Tensor] = None,
        average_attn_weights: bool = True,
) -> Tuple[Tensor, Optional[Tensor]]:
    is_batched = _mha_shape_check(query, key, value, key_padding_mask, attn_mask, num_heads)

    # For unbatched input, we unsqueeze at the expected batch-dim to pretend that the input
    # is batched, run the computation and before returning squeeze the
    # batch dimension so that the output doesn't carry this temporary batch dimension.
    if not is_batched:
        # unsqueeze if the input is unbatched
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(0)

    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape
    assert embed_dim == embed_dim_to_check, \
        f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    if isinstance(embed_dim, torch.Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = embed_dim.div(num_heads, rounding_mode = 'trunc')
    else:
        head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    if use_separate_proj_weight:
        # allow MHA to have different embedding dimensions when separate projection weights are used
        assert key.shape[:2] == value.shape[:2], \
            f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
    else:
        assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

    #
    # compute in-projection
    #
    if not use_separate_proj_weight:
        q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
    else:
        assert q_proj_weight is not None, "use_separate_proj_weight is True but q_proj_weight is None"
        assert k_proj_weight is not None, "use_separate_proj_weight is True but k_proj_weight is None"
        assert v_proj_weight is not None, "use_separate_proj_weight is True but v_proj_weight is None"
        if in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = in_proj_bias.chunk(3)
        q, k, v = _in_projection(query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v)

    # prep attention mask
    if attn_mask is not None:
        if attn_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)
        else:
            assert attn_mask.is_floating_point() or attn_mask.dtype == torch.bool, \
                f"Only float, byte, and bool types are supported for attn_mask, not {attn_mask.dtype}"
        # ensure attn_mask's dim is 3
        if attn_mask.dim() == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(
                    f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(
                    f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
        else:
            raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

    # prep key padding mask
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn(
            "Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
        key_padding_mask = key_padding_mask.to(torch.bool)

    # add bias along batch dimension (currently second)
    if bias_k is not None and bias_v is not None:
        assert static_k is None, "bias cannot be added to static key."
        assert static_v is None, "bias cannot be added to static value."
        k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
        v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))
    else:
        assert bias_k is None
        assert bias_v is None

    #
    # reshape q, k, v for multihead attention and make em batch first
    #
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if static_k is None:
        k = k.contiguous().view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_k.size(0) == bsz * num_heads, \
            f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size(0)}"
        assert static_k.size(2) == head_dim, \
            f"expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}"
        k = static_k
    if static_v is None:
        v = v.contiguous().view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_v.size(0) == bsz * num_heads, \
            f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.size(0)}"
        assert static_v.size(2) == head_dim, \
            f"expecting static_v.size(2) of {head_dim}, but got {static_v.size(2)}"
        v = static_v

    # add zero attention along batch dimension (now first)
    if add_zero_attn:
        zero_attn_shape = (bsz * num_heads, 1, head_dim)
        k = torch.cat([k, torch.zeros(zero_attn_shape, dtype = k.dtype, device = k.device)], dim = 1)
        v = torch.cat([v, torch.zeros(zero_attn_shape, dtype = v.dtype, device = v.device)], dim = 1)
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    # update source sequence length after adjustments
    src_len = k.size(1)

    # merge key padding and attention masks
    if key_padding_mask is not None:
        assert key_padding_mask.shape == (bsz, src_len), \
            f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
        key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len). \
            expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
        if attn_mask is None:
            attn_mask = key_padding_mask
        elif attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.logical_or(key_padding_mask)
        else:
            attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

    # convert mask to float
    if attn_mask is not None and attn_mask.dtype == torch.bool:
        new_attn_mask = torch.zeros_like(attn_mask, dtype = q.dtype)
        new_attn_mask.masked_fill_(attn_mask, float("-inf"))
        attn_mask = new_attn_mask

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    #
    # (deep breath) calculate attention and out projection
    #
    attn_output, attn_output_weights = scaled_dot_product_attention(q, k, v, temperature, attn_mask, dropout_p)
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # optionally average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        if average_attn_weights:
            attn_output_weights = attn_output_weights.sum(dim = 1) / num_heads

        if not is_batched:
            # squeeze the output if input was unbatched
            attn_output = attn_output.squeeze(1)
            attn_output_weights = attn_output_weights.squeeze(0)
        return attn_output, attn_output_weights
    else:
        if not is_batched:
            # squeeze the output if input was unbatched
            attn_output = attn_output.squeeze(1)
        return attn_output, None


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features = None, out_features = None, act_layer = nn.GELU, drop = 0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std = .02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, sim_mat_shape, num_heads = 8, qkv_bias = False, qk_scale = None, attn_drop = 0., proj_drop = 0.,
                 sr_ratio = 1, masked=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = ((qk_scale or head_dim) * 0.5) ** -0.5
        self.scale = nn.Parameter(torch.Tensor([self.scale]), requires_grad = True)

        self.q = nn.Linear(dim, dim, bias = qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias = qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.masked = masked

        if self.masked:
            self.mask = 1 - torch.eye(sim_mat_shape[-1], device = device)
            self.mask = self.mask.expand(sim_mat_shape)


        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size = sr_ratio, stride = sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def masked_softmax(self,x, mask, dim = -1):
        masked_vec = x * mask.float()
        max_vec = torch.max(masked_vec, dim = dim, keepdim = True)[0]
        exps = torch.exp(masked_vec - max_vec)
        masked_exps = exps * mask.float()
        masked_sums = masked_exps.sum(dim, keepdim = True)
        zeros = (masked_sums == 0)
        masked_sums += zeros.float()
        return masked_exps / masked_sums

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std = .02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)


        if self.sr_ratio  > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]


        attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.masked:
            attn = self.masked_softmax(x = attn, mask = self.mask, dim = -1)
        else:
            attn = attn.softmax(dim = -1)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
class MultiHeadAttentionLSA(torch.nn.MultiheadAttention):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.temperature = nn.Parameter(torch.Tensor([1 / math.sqrt(float(self.kdim))]), requires_grad = True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std = .02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None,
                average_attn_weights: bool = True) -> Tuple[Tensor, Optional[Tensor]]:

        is_batched = query.dim() == 3
        if self.batch_first and is_batched:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = multi_head_attention_forward(
                    query, key, value, self.temperature, self.embed_dim, self.num_heads,
                    self.in_proj_weight, self.in_proj_bias,
                    self.bias_k, self.bias_v, self.add_zero_attn,
                    self.dropout, self.out_proj.weight, self.out_proj.bias,
                    training = self.training,
                    key_padding_mask = key_padding_mask, need_weights = need_weights,
                    attn_mask = attn_mask, use_separate_proj_weight = True,
                    q_proj_weight = self.q_proj_weight, k_proj_weight = self.k_proj_weight,
                    v_proj_weight = self.v_proj_weight, average_attn_weights = average_attn_weights)
        else:
            attn_output, attn_output_weights = multi_head_attention_forward(
                    query, key, value, self.temperature, self.embed_dim, self.num_heads,
                    self.in_proj_weight, self.in_proj_bias,
                    self.bias_k, self.bias_v, self.add_zero_attn,
                    self.dropout, self.out_proj.weight, self.out_proj.bias,
                    training = self.training,
                    key_padding_mask = key_padding_mask, need_weights = need_weights,
                    attn_mask = attn_mask, average_attn_weights = average_attn_weights)
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights


class Block(nn.Module):

    def __init__(self, dim, num_heads,sim_mat_shape,  mlp_ratio = 4., qkv_bias = False, qk_scale = None, drop = 0., attn_drop = 0.,
                 drop_path = 0., act_layer = nn.GELU, norm_layer = nn.LayerNorm, sr_ratio = 1, masked_attention=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
                dim,
                num_heads = num_heads, qkv_bias = qkv_bias, qk_scale = qk_scale,
                attn_drop = attn_drop, proj_drop = drop, sr_ratio = sr_ratio, masked = masked_attention, sim_mat_shape= sim_mat_shape)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features = dim, hidden_features = mlp_hidden_dim, act_layer = act_layer, drop = drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std = .02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class OverlapPatchEmbedSPT(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size = 224, patch_size = 7, stride = 4, in_chans = 3, embed_dim = 768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W


        self.SC = ShiftedConcatenator(image_size = img_size[0], patch_size = patch_size[0])

        self.proj = nn.Conv2d(in_chans*5, embed_dim, kernel_size = patch_size, stride = stride,
                              padding = (patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std = .02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):

        x = self.SC(x)
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)



        return x, H, W


def crop_to_bounding_box(
        images,
        offset_height,
        offset_width,
        target_height,
        target_width
):
    return images[:, :, offset_height: offset_height + target_height, offset_width: offset_width + target_width]


def pad_to_bounding_box(
        image,
        offset_height,
        offset_width,
        target_height,
        target_width,
):
    im = torch.ones((image.shape[0], image.shape[1], target_height, target_width))
    if str(device) == 'cuda':
        im = im.cuda()
    im = im * image.min()
    im[:, :, offset_height:image.shape[2] + offset_height, offset_width:image.shape[3] + offset_width] = image

    return im

class ShiftedConcatenator(nn.Module):
    """ Image to Patch concat
    """

    def __init__(
            self,
            image_size,
            patch_size,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.half_patch = patch_size // 2
        self.image_size = image_size

    def crop_shift_pad(self, images, mode):
        # Build the diagonally shifted images
        if mode == "left-up":
            crop_height = self.half_patch
            crop_width = self.half_patch
            shift_height = 0
            shift_width = 0
        elif mode == "left-down":
            crop_height = 0
            crop_width = self.half_patch
            shift_height = self.half_patch
            shift_width = 0
        elif mode == "right-up":
            crop_height = self.half_patch
            crop_width = 0
            shift_height = 0
            shift_width = self.half_patch
        else:
            crop_height = 0
            crop_width = 0
            shift_height = self.half_patch
            shift_width = self.half_patch

        # Crop the shifted images and pad them
        crop = crop_to_bounding_box(
                images,
                offset_height = crop_height,
                offset_width = crop_width,
                target_height = self.image_size - self.half_patch,
                target_width = self.image_size - self.half_patch,
        )
        shift_pad = pad_to_bounding_box(
                crop,
                offset_height = shift_height,
                offset_width = shift_width,
                target_height = self.image_size,
                target_width = self.image_size,
        )
        return shift_pad

    def forward(self, images):
        # Concat the shifted images with the original image
        images = torch.cat(
                    [
                        images,
                        self.crop_shift_pad(images, mode = "left-up"),
                        self.crop_shift_pad(images, mode = "left-down"),
                        self.crop_shift_pad(images, mode = "right-up"),
                        self.crop_shift_pad(images, mode = "right-down"),
                    ],
                    axis = 1,
            )


        return images
class ShiftedPatchTokenization(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(
            self,
            img_size,
            patch_size,
            stride,
            in_chans,
            embed_dim,
            layer_norm_eps = 1e-6,
            vanilla = False,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.vanilla = vanilla  # Flag to swtich to vanilla patch extractor
        self.image_size = img_size
        self.patch_size = patch_size
        self.half_patch = patch_size // 2
        self.num_patches = (img_size // patch_size) ** 2
        self.flattened_dim = 5 * patch_size * patch_size * in_chans
        self.projection = torch.nn.Linear(self.flattened_dim, embed_dim)
        self.layer_norm = torch.nn.LayerNorm(self.flattened_dim, eps = layer_norm_eps)
        self.stride = stride

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std = .02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def crop_shift_pad(self, images, mode):
        # Build the diagonally shifted images
        if mode == "left-up":
            crop_height = self.half_patch
            crop_width = self.half_patch
            shift_height = 0
            shift_width = 0
        elif mode == "left-down":
            crop_height = 0
            crop_width = self.half_patch
            shift_height = self.half_patch
            shift_width = 0
        elif mode == "right-up":
            crop_height = self.half_patch
            crop_width = 0
            shift_height = 0
            shift_width = self.half_patch
        else:
            crop_height = 0
            crop_width = 0
            shift_height = self.half_patch
            shift_width = self.half_patch

        # Crop the shifted images and pad them
        crop = crop_to_bounding_box(
                images,
                offset_height = crop_height,
                offset_width = crop_width,
                target_height = self.image_size - self.half_patch,
                target_width = self.image_size - self.half_patch,
        )
        shift_pad = pad_to_bounding_box(
                crop,
                offset_height = shift_height,
                offset_width = shift_width,
                target_height = self.image_size,
                target_width = self.image_size,
        )
        return shift_pad

    def forward(self, images):
        if not self.vanilla:
            # Concat the shifted images with the original image
            images = torch.cat(
                    [
                        images,
                        self.crop_shift_pad(images, mode = "left-up"),
                        self.crop_shift_pad(images, mode = "left-down"),
                        self.crop_shift_pad(images, mode = "right-up"),
                        self.crop_shift_pad(images, mode = "right-down"),
                    ],
                    axis = 1,
            )


        # 512*
        #
        # Patchify the images and flatten it
        patches = extract_tensor_patches(
                images,
                self.patch_size,
                self.stride,
                padding = self.patch_size // 2
        )

        #5329*32

        # 170k

        #

        flat_patches = torch.flatten(patches, start_dim = 2)
        if not self.vanilla:
            # Layer normalize the flat patches and linearly project it
            tokens = self.layer_norm(flat_patches)
            tokens = self.projection(tokens)
        else:
            # Linearly project the flat patches
            tokens = self.projection(flat_patches)
        return tokens


class PatchEncoder(nn.Module):
    def __init__(
            self, num_patches, embed_dim, **kwargs
    ):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.position_embedding = nn.Embedding(
                num_embeddings = num_patches, embedding_dim = embed_dim, device = device
        )  #
        self.positions = torch.arange(start = 0, end = self.num_patches, step = 1, device = device)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std = .02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, encoded_patches):
        encoded_positions = self.position_embedding(self.positions)
        encoded_patches = encoded_patches + encoded_positions
        return encoded_patches


class LinearMLP(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim = 2048, embed_dim = 768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

#682000

class DWConv(nn.Module):
    def __init__(self, dim = 768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias = True, groups = dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class Segformer(nn.Module):
    def __init__(
            self,
            batch_size = 8,
            pretrained = None,
            img_size = 1024,
            patch_size = 4,
            in_chans = 3,
            num_classes = 1,
            embed_dims = [64, 128, 256, 512],
            num_heads = [1, 2, 5, 8],
            mlp_ratios = [4, 4, 4, 4],
            qkv_bias = True,
            qk_scale = None,
            drop_rate = 0.,
            attn_drop_rate = 0.,
            drop_path_rate = 0.,
            norm_layer = nn.LayerNorm,
            depths = [3, 6, 40, 3],
            sr_ratios = [8, 4, 2, 1],
            decoder_dim = 256,
            positional_encoding = False,
            overlap_patch_embed = False,
            masked_attention = False,
            use_drloc = False,
            drloc_mode = "l1",
            sample_size = 32,
            use_abs = False

    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.shift_patch_tokenization = True
        self.overlap_patch_embed = overlap_patch_embed
        self.positional_encoding = positional_encoding
        self.use_drloc = use_drloc


        if True:
            # patch_embe
            if self.overlap_patch_embed:
                # patch_embed
                self.patch_embed1 = OverlapPatchEmbedSPT(
                        img_size = img_size,
                        patch_size = 7,
                        stride = 4,
                        in_chans = in_chans,
                        embed_dim = embed_dims[0]
                )
            else:
                self.patch_embed1 = ShiftedPatchTokenization(
                        img_size = img_size,
                        in_chans = in_chans,
                        stride = 4,
                        patch_size = 7,
                        embed_dim = embed_dims[0]
                )
            patch_size = 7
            padding = int(patch_size // 2)
            stride = 4
            num_patches0 = math.floor(((img_size + 2*padding - 1*(patch_size-1)-1)/stride)+1)**2

            self.patch_encoder1 = PatchEncoder(
                    num_patches = num_patches0,
                    embed_dim = embed_dims[0]
            )


            if self.overlap_patch_embed:
                self.patch_embed2 = OverlapPatchEmbedSPT(
                        img_size = img_size // 4,
                        patch_size = 3,
                        stride = 2,
                        in_chans = embed_dims[0],
                        embed_dim = embed_dims[1]
                )
            else:
                self.patch_embed2 = ShiftedPatchTokenization(
                        img_size = img_size // 4,
                        patch_size = 3,  #
                        stride = 2,
                        in_chans = embed_dims[0],
                        embed_dim = embed_dims[1]
                )

            patch_size = 3
            padding = int(patch_size // 2)
            stride = 2
            num_patches1 = math.floor((((img_size // 4) + 2*padding - 1*(patch_size-1)-1)/stride)+1)**2

            self.patch_encoder2 = PatchEncoder(
                    num_patches =num_patches1,
                    embed_dim = embed_dims[1]
            )

            if self.overlap_patch_embed:
                self.patch_embed3 = OverlapPatchEmbedSPT(
                        img_size = img_size // 8,
                        patch_size = 3,
                        stride = 2,
                        in_chans = embed_dims[1],
                        embed_dim = embed_dims[2]
                )

            else:

                self.patch_embed3 = ShiftedPatchTokenization(
                        img_size = img_size // 8,
                        patch_size = 3,
                        stride = 2,
                        in_chans = embed_dims[1],
                        embed_dim = embed_dims[2]
                )

            patch_size = 3
            padding = int(patch_size // 2)
            stride = 2
            num_patches2 = math.floor((((img_size // 8) + 2*padding - 1*(patch_size-1)-1)/stride)+1)**2

            self.patch_encoder3 = PatchEncoder(
                    num_patches = num_patches2,
                    embed_dim = embed_dims[2]
            )
            self.patch_embed4 = ShiftedPatchTokenization(
                    img_size = img_size // 16,
                    patch_size = 3,
                    stride = 2,
                    in_chans = embed_dims[2],
                    embed_dim = embed_dims[3]
            )

            patch_size = 3
            padding = int(patch_size // 2)
            stride = 2
            num_patches3 = math.floor((((img_size // 16) + 2*padding - 1*(patch_size-1)-1)/stride)+1)**2

            if self.overlap_patch_embed:
                self.patch_embed4 = OverlapPatchEmbedSPT(
                        img_size = img_size // 16,
                        patch_size = 3,
                        stride = 2,
                        in_chans = embed_dims[2],
                        embed_dim = embed_dims[3]
                )
            else:
                self.patch_encoder4 = PatchEncoder(
                        num_patches = num_patches3,
                        embed_dim = embed_dims[3]
                )

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        sim_mat0_shape = (batch_size, num_heads[0], num_patches0, num_patches0)
        sim_mat1_shape = (batch_size,num_heads[1], num_patches1, num_patches1)
        sim_mat2_shape = (batch_size, num_heads[2], num_patches2, num_patches2)
        sim_mat3_shape = (batch_size,num_heads[3], num_patches3, num_patches3)

        self.block1 = nn.ModuleList([Block(
                dim = embed_dims[0], num_heads = num_heads[0], mlp_ratio = mlp_ratios[0], qkv_bias = qkv_bias,
                drop = drop_rate, drop_path = dpr[cur + i], norm_layer = norm_layer, masked_attention = masked_attention, sim_mat_shape =  sim_mat0_shape)
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
                dim = embed_dims[1], num_heads = num_heads[1], mlp_ratio = mlp_ratios[1], qkv_bias = qkv_bias,

                drop = drop_rate, drop_path = dpr[cur + i], norm_layer = norm_layer, masked_attention = masked_attention, sim_mat_shape =  sim_mat1_shape)
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
                dim = embed_dims[2], num_heads = num_heads[2], mlp_ratio = mlp_ratios[2], qkv_bias = qkv_bias,
                drop = drop_rate, drop_path = dpr[cur + i], norm_layer = norm_layer, masked_attention = masked_attention, sim_mat_shape =  sim_mat2_shape)
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
                dim = embed_dims[3], num_heads = num_heads[3], mlp_ratio = mlp_ratios[3], qkv_bias = qkv_bias,
                drop = drop_rate, drop_path = dpr[cur + i], norm_layer = norm_layer, masked_attention = masked_attention, sim_mat_shape =  sim_mat3_shape)
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        # segmentation head
        self.linear_c4 = LinearMLP(input_dim = embed_dims[3], embed_dim = decoder_dim)
        self.linear_c3 = LinearMLP(input_dim = embed_dims[2], embed_dim = decoder_dim)
        self.linear_c2 = LinearMLP(input_dim = embed_dims[1], embed_dim = decoder_dim)
        self.linear_c1 = LinearMLP(input_dim = embed_dims[0], embed_dim = decoder_dim)
        self.linear_fuse = nn.Conv2d(4 * decoder_dim, decoder_dim, 1)
        self.dropout = nn.Dropout2d(drop_rate)
        self.linear_pred = nn.Conv2d(decoder_dim, num_classes, kernel_size = 1)

        self.dim = num_patches0

        if use_drloc:
            self.drloc = DenseRelativeLoc(
                    in_dim = self.dim,
                    out_dim = 2 if drloc_mode == "l1" else 14,
                    sample_size = sample_size,
                    drloc_mode = drloc_mode,
                    use_abs = use_abs
            )

        self.apply(self._init_weights)
        self.init_weights(pretrained = pretrained)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std = .02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained = None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, map_location = 'cpu', strict = False, logger = logger)

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool = ''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        # stage 1

        # x: B, C, H, W

        B, C, H, W = x.shape
        if self.overlap_patch_embed:
            x, H, W = self.patch_embed1(x)
        else:
            x = self.patch_embed1(x)
            H, W = int(H / (2 ** 2)), int(W / (2 ** 2))
        if self.positional_encoding:
            x = self.patch_encoder1(x)


        # H/2^{i+1} , W/2^{i+1}

        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        B, C, H, W = x.shape
        if self.overlap_patch_embed:
            x, H, W = self.patch_embed2(x)
        else:
            x = self.patch_embed2(x)
            H, W = int(H / 2), int(W / 2)
        if self.positional_encoding:
            x = self.patch_encoder2(x)

        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        B, C, H, W = x.shape
        if self.overlap_patch_embed:
            x, H, W = self.patch_embed3(x)
        else:
            x = self.patch_embed3(x)
            H, W = int(H / 2), int(W / 2)
        if self.positional_encoding:
            x = self.patch_encoder3(x)

        for i, blk in enumerate(self.block3):
            x = blk(x,H,W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        B, C, H, W = x.shape
        if self.overlap_patch_embed:
            x, H, W = self.patch_embed4(x)
        else:
            x = self.patch_embed4(x)
            H, W = int(H / 2), int(W / 2)

        if self.positional_encoding:
            x = self.patch_encoder4(x)

        for i, blk in enumerate(self.block4):
            x = blk(x,H,W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

    def forward(self, x):
        x = self.forward_features(x)

        c1, c2, c3, c4 = x

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape
        h_out, w_out = c1.size()[2], c1.size()[3]

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size = c1.size()[2:], mode = 'bilinear', align_corners = False)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size = c1.size()[2:], mode = 'bilinear', align_corners = False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size = c1.size()[2:], mode = 'bilinear', align_corners = False)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        x = torch.cat([_c4, _c3, _c2, _c1], dim=1)
        outs = Munch()
        # SSUP
        if self.use_drloc:
            x_last = x[:,1:] # B, L, C
            x_last = x_last.transpose(1, 2) # [B, C, L]
            B, C, HW = x_last.size()
            H = W = int(math.sqrt(HW))
            x_last = x_last.view(B, C, H, W) # [B, C, H, W]

            drloc_feats, deltaxy = self.drloc(x_last)
            outs.drloc = [drloc_feats]
            outs.deltaxy = [deltaxy]
            outs.plz = [H] # plane size


        _c = self.linear_fuse(x)

        x = self.dropout(_c)
        x = self.linear_pred(x)

        x = F.interpolate(input = x, size = (h_out, w_out), mode = 'bilinear', align_corners = False)
        x = torch.sigmoid(x)
        x = x.type(torch.float32)

        outs.x = x

        return outs
