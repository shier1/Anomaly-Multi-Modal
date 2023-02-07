import torch
import torch.nn as nn
import torch.nn.functional as F

#from .attn import AnomalyAttention, AttentionLayer, CrossAttentionLayer
from .attn_verison2 import AnomalyAttention, AttentionLayer, CrossAttentionLayer
from .embed import DataEmbedding, TokenEmbedding, PositionalEmbedding
from .series_spatial_block import Block


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.series_conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.series_conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.series_norm1 = nn.LayerNorm(d_model)
        self.series_norm2 = nn.LayerNorm(d_model)
        
        self.freq_conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.freq_conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.freq_norm1 = nn.LayerNorm(d_model)
        self.freq_norm2 = nn.LayerNorm(d_model)
 
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, series_token, freq_token, attn_mask=None):
        new_series_token, new_freq_token, attn, mask, sigma = self.attention(
            freq_token, series_token, series_token,
            attn_mask=attn_mask
        )
        new_series_token = series_token + self.dropout(new_series_token)
        new_freq_token = freq_token + self.dropout(new_freq_token)

        new_series_token = self.series_norm1(new_series_token)
        new_freq_token = self.freq_norm1(new_freq_token)

        series_identity = new_series_token
        freq_identity = new_freq_token

        new_series_token = self.dropout(self.activation(self.series_conv1(new_series_token.transpose(-1, 1))))
        new_series_token = self.dropout(self.series_conv2(new_series_token).transpose(-1, 1))

        new_freq_token = self.dropout(self.activation(self.freq_conv1(new_freq_token.transpose(-1, 1))))
        new_freq_token = self.dropout(self.freq_conv2(new_freq_token).transpose(-1, 1))

        return self.series_norm2(new_series_token + series_identity), self.freq_norm2(new_freq_token+freq_identity), attn, mask, sigma


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, series_token, freq_token, attn_mask=None):
        # x [B, L, D]
        series_list = []
        prior_list = []
        sigma_list = []
        for attn_layer in self.attn_layers:
            series_token, freq_token, series, prior, sigma = attn_layer(series_token, freq_token, attn_mask=attn_mask)
            series_list.append(series)
            prior_list.append(prior)
            sigma_list.append(sigma)

        if self.norm is not None:
            series_token = self.norm(series_token)

        return series_token, series_list, prior_list, sigma_list


class AnomalyTransformer(nn.Module):
    def __init__(self, win_size, enc_in, c_out, d_model=512, n_heads=8, e_layers=3, d_ff=512,
                 dropout=0.0, activation='gelu', output_attention=True):
        super(AnomalyTransformer, self).__init__()
        self.output_attention = output_attention

        # Encoding
        # self.position_embeding = PositionalEmbedding(d_model)
        # self.embeding = TokenEmbedding(d_model, d_model)
        self.series_embeding = DataEmbedding(enc_in, d_model)
        self.freq_embeding = DataEmbedding(enc_in, d_model)
        self.shared_conv_block = self._make_share_block(enc_in, 3)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    CrossAttentionLayer(
                        AnomalyAttention(win_size, False, attention_dropout=dropout, output_attention=output_attention),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, c_out, bias=True)

    def _make_share_block(self, enc_in, n):
        layer_list = []
        for i in range(n):
            layer_list.append(Block(enc_in))
        return nn.Sequential(*layer_list)

    def forward(self, x_series, x_freq):
        x_series = self.shared_conv_block(x_series)
        x_freq = self.shared_conv_block(x_freq)
        x_sereis_token = self.series_embeding(x_series)
        x_freq_token = self.freq_embeding(x_freq)

        enc_out, series, prior, sigmas = self.encoder(x_sereis_token, x_freq_token)
        enc_out = self.projection(enc_out)

        if self.output_attention:
            return enc_out, series, prior, sigmas
        else:
            return enc_out  # [B, L, D]
