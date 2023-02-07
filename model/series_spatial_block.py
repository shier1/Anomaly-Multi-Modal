import torch
import torch.nn as nn


class SeriesConvs(nn.Module):
    def __init__(self, enc_in, d_model):
        super().__init__()
        self.conv1x = nn.Conv1d(enc_in, d_model, kernel_size=1, padding=0, bias=False)
        self.conv3x = nn.Conv1d(enc_in, d_model, kernel_size=3, padding=1, padding_mode='circular', bias=False)
        self.conv7x = nn.Conv1d(enc_in, d_model, kernel_size=7, padding=3, padding_mode='circular', bias=False)
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.avgpool = nn.AvgPool1d(kernel_size=2)
        self.conv_proj = nn.Conv1d(enc_in*2 + d_model*3, d_model, kernel_size=1, padding=0, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.activate = nn.GELU()
        self.bn = nn.BatchNorm1d(num_features=d_model)
        self._init_weight()

    def _init_weight(self,):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = x.permute([0, 2, 1]).contiguous()
        conv1x = self.conv1x(x)
        conv3x = self.conv3x(x)
        conv7x = self.conv7x(x)
        max_f = self.maxpool(x)
        avg_f = self.avgpool(x)
        pool_f = torch.concat([max_f, avg_f], dim=-1)
        series_features = torch.concat([x, conv1x, conv3x, conv7x, pool_f], dim=-2)
        out = self.conv_proj(series_features)
        out = self.activate(self.bn(out))
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
        self.activate = nn.GELU()
        self.bn = nn.BatchNorm1d(num_features=win_size)
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
        pool_f = torch.concat([max_f, avg_f], dim=-1)
        pool_f = self.fc(pool_f)
        spatial_features = torch.concat([identity, conv1x, conv3x, conv7x, pool_f], dim=-2)
        out = self.conv_proj(spatial_features)
        out = self.activate(self.bn(out))
        return out


class MscaModule(nn.Module):
    def __init__(self, enc_in):
        super().__init__()
        self.conv1 = nn.Conv1d(enc_in, enc_in, kernel_size=1, padding=0, bias=False)
        self.conv5 = nn.Conv1d(enc_in, enc_in, kernel_size=5, padding=2, padding_mode='circular', bias=False)
        self.conv7 = nn.Conv1d(enc_in, enc_in, kernel_size=7, padding=3, padding_mode='circular', bias=False)
        self.conv11 = nn.Conv1d(enc_in, enc_in, kernel_size=11, padding=5, padding_mode='circular', bias=False)
        self.conv21 = nn.Conv1d(enc_in, enc_in, kernel_size=21, padding=10, padding_mode='circular', bias=False)
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.avgpool = nn.AvgPool1d(kernel_size=2)
        self.conv_proj = nn.Conv1d(5*enc_in, enc_in, kernel_size=1, padding=0, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.activate = nn.GELU()
        self.bn = nn.BatchNorm1d(num_features=enc_in)
        self.proj = nn.Conv1d(enc_in, enc_in, kernel_size = 1, padding=0, bias = False)
        self._init_weight()

    def _init_weight(self,):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = x.permute([0, 2, 1]).contiguous()
        short_cut = x
        x = self.bn(x)
        x = self.conv1(x)
        x = self.activate(x)

        attn = self.conv5(x)
        attn7x = self.conv7(attn)
        attn11x = self.conv11(attn)
        attn21x = self.conv21(attn)
        attn_max = self.maxpool(attn)
        attn_avg = self.avgpool(attn)
        attn_pool = torch.concat([attn_max, attn_avg], dim=-1)

        #attn = attn + attn7x + attn11x + attn21x + attn_pool
        attn = torch.concat([attn, attn7x, attn11x, attn21x, attn_pool], dim=-2)
        attn = self.conv_proj(attn)
        attn = self.softmax(attn)

        x = attn * x
        x = self.proj(x)
        out = x + short_cut
        out = out.permute([0, 2, 1]).contiguous()
        return out


class Block(nn.Module):
    def __init__(self, enc_in):
        super().__init__()
        # self.spatial_conv = SpatialConvs(win_size, enc_in)
        # self.series_conv = SeriesConvs(enc_in, d_model)
        self.series_block = MscaModule(enc_in)

    def forward(self, x):
        # spatial_features = self.spatial_conv(x)
        series_features = self.series_block(x)
        return series_features
