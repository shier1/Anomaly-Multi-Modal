import torch
import torch.nn as nn
import numpy as np
import math
from math import sqrt


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class AnomalyAttention(nn.Module):
    def __init__(self, win_size, mask_flag=True, scale=None, attention_dropout=0.0, output_attention=False):
        super(AnomalyAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        window_size = win_size
        self.distances = torch.zeros((window_size, window_size)).cuda()
        for i in range(window_size):
            for j in range(window_size):
                self.distances[i][j] = abs(i - j)

    def forward(self, queries, keys, values, sigma, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        attn = scale * scores

        sigma = sigma.transpose(1, 2)  # B L H ->  B H L
        window_size = attn.shape[-1]
        sigma = torch.sigmoid(sigma * 5) + 1e-5
        sigma = torch.pow(3, sigma) - 1
        sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, window_size)  # B H L L
        prior = self.distances.unsqueeze(0).unsqueeze(0).repeat(sigma.shape[0], sigma.shape[1], 1, 1).cuda()
        prior = 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(-prior ** 2 / 2 / (sigma ** 2))

        series = self.dropout(torch.softmax(attn, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", series, values)

        if self.output_attention:
            return (V.contiguous(), series, prior, sigma)
        else:
            return (V.contiguous(), None)


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model,
                                          d_keys * n_heads)
        self.key_projection = nn.Linear(d_model,
                                        d_keys * n_heads)
        self.value_projection = nn.Linear(d_model,
                                          d_values * n_heads)
        self.sigma_projection = nn.Linear(d_model,
                                          n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)

        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        x = queries
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        sigma = self.sigma_projection(x).view(B, L, H)

        out, series, prior, sigma = self.inner_attention(
            queries,
            keys,
            values,
            sigma,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), queries.view(B, L, -1), series, prior, sigma


class CrossAttentionLayer(nn.Module):
    def __init__(self, attention, d_model, num_heads=8, qkv_bias=False, attn_drop=0.):
        super().__init__()
        assert d_model % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = d_model // num_heads
        self.scale = head_dim ** -0.5

        self.inner_attention = attention
        self.series_qkv = nn.Linear(d_model, d_model * 3, bias=qkv_bias)
        self.series_drop = nn.Dropout(attn_drop)
        self.series_proj = nn.Linear(d_model, d_model)

        self.freq_qkv = nn.Linear(d_model, d_model * 3, bias=qkv_bias)
        self.freq_attn_drop = nn.Dropout(attn_drop)
        self.freq_proj = nn.Linear(d_model, d_model)

        self.sigma_projection = nn.Linear(d_model,
                                          num_heads)

    def forward(self, x_freq, x_series, _, attn_mask):
        B, N, C = x_series.shape
        series_qkv = self.series_qkv(x_series).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4)
        freq_qkv = self.freq_qkv(x_freq).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4)
        series_q, series_k, series_v = series_qkv.unbind(0)
        freq_q, freq_k, freq_v = freq_qkv.unbind(0)
        series_sigma = self.sigma_projection(x_series)

        # print(freq_q.shape, series_k.shape, series_v.shape, series_sigma.shape)
        x_series, series_asso, prior_asso, series_sigma = self.inner_attention(
            freq_q,
            series_k,
            series_v,
            series_sigma,
            attn_mask
        )
        freq_attn = (series_q @ freq_k.transpose(-2, -1)) * self.scale
        freq_attn = freq_attn.softmax(dim=-1)
        freq_attn = self.freq_attn_drop(freq_attn)
        x_freq = (freq_attn @ freq_v).transpose(1, 2).reshape(B, N, C)

        series_attn = (freq_q @ series_v.transpose(-2, -1)) * self.scale
        series_attn = series_attn.softmax(dim=-1)
        series_attn = self.series_drop(series_attn)
        x_series = (series_attn @ series_v).transpose(-2, -1).reshape(B, N, C)

        x_series = x_series.reshape(B, N, -1)
        x_freq = x_freq.reshape(B, N, -1)

        return self.series_proj(x_series), \
                self.freq_proj(x_freq), \
                    series_asso, prior_asso, series_sigma
