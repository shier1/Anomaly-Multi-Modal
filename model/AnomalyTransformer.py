import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn import AnomalyAttention, AttentionLayer, CrossAttentionLayer
from .embed import DataEmbedding, TokenEmbedding, PositionalEmbedding


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
            x, freq_token, series, prior, sigma = attn_layer(series_token, freq_token, attn_mask=attn_mask)
            series_list.append(series)
            prior_list.append(prior)
            sigma_list.append(sigma)

        if self.norm is not None:
            x = self.norm(x)

        return x, series_list, prior_list, sigma_list

class SeriesConvs(nn.Module):
    def __init__(self, enc_in, d_model):
        super().__init__()
        self.conv1x = nn.Conv1d(enc_in, d_model, kernel_size=1, padding=0, bias=False)
        self.conv3x = nn.Conv1d(enc_in, d_model, kernel_size=3, padding=1, padding_mode='circular', bias=False)
        self.conv7x = nn.Conv1d(enc_in, d_model, kernel_size=7, padding=3, padding_mode='circular', bias=False)
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.avgpool = nn.AvgPool1d(kernel_size=2)
        self.conv_proj = nn.Conv1d(enc_in*2+d_model*3, d_model, kernel_size=1, padding=0, bias=False)
        self._init_weight()

    def _init_weight(self,):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = x.permute([0, 2, 1]).contiguous()
        identity = x
        conv1x = self.conv1x(x)
        conv3x = self.conv3x(x)
        conv7x = self.conv7x(x)
        max_f = self.maxpool(x)
        avg_f = self.avgpool(x)
        pool_f = torch.cat([max_f, avg_f], dim=-1)
        series_features = torch.cat([identity, conv1x, conv3x, conv7x, pool_f], dim=-2)
        out = self.conv_proj(series_features)
        out = out.permute([0, 2, 1]).contiguous()
        return out



class SpatialConvs(nn.Module):
    def __init__(self, win_size, num_features):
        super().__init__()
        self.conv1x = nn.Conv1d(win_size, win_size, kernel_size=1, padding=0, bias=False)
        self.conv3x = nn.Conv1d(win_size, win_size, kernel_size=3, padding=1, padding_mode='circular', bias=False)
        self.conv7x = nn.Conv1d(win_size, win_size, kernel_size=7, padding=3, padding_mode='circular', bias=False)
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.avgpool = nn.AvgPool1d(kernel_size=2)
        self.conv_proj = nn.Conv1d(win_size*5, win_size, kernel_size=1, padding=0, bias=False)
        self.fc = nn.Linear(num_features//2*2, num_features)
        self._init_weight()

    def _init_weight(self,):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        identity = x
        conv1x = self.conv1x(x)
        conv3x = self.conv3x(x)
        conv7x = self.conv7x(x)
        max_f = self.maxpool(x)
        avg_f = self.avgpool(x)
        pool_f = torch.cat([max_f, avg_f], dim=-1)
        pool_f = self.fc(pool_f)
        spatial_features = torch.cat([identity, conv1x, conv3x, conv7x, pool_f], dim=-2)
        out = self.conv_proj(spatial_features)
        return out

class Block(nn.Module):
    def __init__(self, win_size, enc_in, d_model):
        super().__init__()
        self.spatial_conv = SpatialConvs(win_size, enc_in)
        self.series_conv = SeriesConvs(enc_in, d_model)
    
    def forward(self, x):
        # spatial_features = self.spatial_conv(x)
        series_features = self.series_conv(x)
        return series_features


class AnomalyTransformer(nn.Module):
    def __init__(self, win_size, enc_in, c_out, d_model=512, n_heads=8, e_layers=3, d_ff=512,
                 dropout=0.0, activation='gelu', output_attention=True):
        super(AnomalyTransformer, self).__init__()
        self.output_attention = output_attention

        # Encoding
        self.position_embeding = PositionalEmbedding(d_model)
        self.shared_conv_block = self._make_share_block(win_size, enc_in, d_model, 3)
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

    def _make_share_block(self, win_size, enc_in, d_model, n):
        first_block = Block(win_size=win_size, enc_in=enc_in, d_model=d_model)
        layer_list = [first_block]
        for i in range(n-1):
            layer_list.append(Block(win_size, d_model, d_model))
        return nn.Sequential(*layer_list)

    def forward(self, x_series, x_freq):
        x_series = self.shared_conv_block(x_series)
        x_freq = self.shared_conv_block(x_freq)
        x_sereis_token = x_series + self.position_embeding(x_series)
        x_freq_token = x_freq + self.position_embeding(x_freq)

        enc_out, series, prior, sigmas = self.encoder(x_sereis_token, x_freq_token)
        enc_out = self.projection(enc_out)

        if self.output_attention:
            return enc_out, series, prior, sigmas
        else:
            return enc_out  # [B, L, D]
