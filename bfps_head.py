import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(Conv, self).__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU(True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        
    def forward(self, x):
        out = self.norm(x)
        out = self.act(out)
        out = self.conv(out)
        return out


class upConv(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super(upConv, self).__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU(True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.scale_factor = scale_factor
        
    def forward(self, x):
        out = self.norm(x)
        out = self.act(out)
        out = F.interpolate(out, scale_factor=self.scale_factor, mode='bilinear')
        out = self.conv(out)
        return out


class FFTSeparator(nn.Module):
    def __init__(self, cutoff=0.7):
        super(FFTSeparator, self).__init__()
        self.cutoff = cutoff

    def forward(self, feature_map):
        batch_size, channels, height, width = feature_map.size()

        f_transform = torch.fft.fft2(feature_map)
        f_transform = torch.fft.fftshift(f_transform)

        crow, ccol = height // 2, width // 2
        mask = torch.zeros((height, width), dtype=torch.float32, device=feature_map.device)
        h_cut = int(self.cutoff * crow)
        w_cut = int(self.cutoff * ccol)
        mask[crow - h_cut:crow + h_cut, ccol - w_cut:ccol + w_cut] = 1.0

        mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, channels, -1, -1)

        f_low = f_transform * mask
        f_high = f_transform * (1.0 - mask)

        low_freq_feature_map = torch.fft.ifft2(torch.fft.ifftshift(f_low)).real
        high_freq_feature_map = torch.fft.ifft2(torch.fft.ifftshift(f_high)).real

        return low_freq_feature_map, high_freq_feature_map
    

class BFE(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, kernel_size=3, iterations=1):
        super(BFE, self).__init__()
        self.scale_factor = scale_factor
        self.kernel_size = kernel_size
        self.iterations = iterations
        self.conv1 = Conv(in_channels, out_channels//2, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = Conv(in_channels, out_channels//2, kernel_size=3, stride=1, padding=1, bias=False)

    def dialate(self, pred):
        return F.max_pool2d(pred, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size // 2)

    def erode(self, pred):
        return -F.max_pool2d(-pred, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size // 2)

    def forward(self, low_level_feature, high_level_pred):
        if self.scale_factor > 1:
            high_level_pred = F.interpolate(high_level_pred, scale_factor = self.scale_factor, mode='bilinear')
        high_level_pred_sigmoid = torch.sigmoid(high_level_pred)
        dialated_pred = high_level_pred_sigmoid
        eroded_pred = high_level_pred_sigmoid
        for _ in range(self.iterations):
            dialated_pred = self.dialate(dialated_pred)
            eroded_pred = self.erode(eroded_pred)
        dialated_background = 1 - eroded_pred
        feature1 = self.conv1(low_level_feature * dialated_pred)
        feature2 = self.conv2(low_level_feature * dialated_background)
        return torch.cat([feature1, feature2], dim=1)


class HFA(nn.Module):
    def __init__(self, in_channels):
        super(HFA, self).__init__()

        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid())

    def forward(self, x, high_freq_feature_map):
        residual = x
        attention_map = self.attention(high_freq_feature_map)
        f = x * attention_map

        out = residual + f
        return out


class BFPSHead(nn.Module):
    def __init__(self, feature_channels, num_classes):
        super().__init__()
        self.sep1 = FFTSeparator(cutoff=0.7)
        self.sep2 = FFTSeparator(cutoff=0.7)
        self.sep3 = FFTSeparator(cutoff=0.7)
        self.sep4 = FFTSeparator(cutoff=0.7)

        self.attn1 = HFA(feature_channels[3])
        self.decoder1_1 = Conv(feature_channels[3], feature_channels[3]//2, kernel_size=3, stride=1, padding=1, bias=False)
        self.decoder1_2 = Conv(feature_channels[3]//2, feature_channels[3]//4, kernel_size=3, stride=1, padding=1, bias=False)
        self.decoder1_3 = Conv(feature_channels[3]//4, feature_channels[3]//8, kernel_size=3, stride=1, padding=1, bias=False)
        self.decoder1_4 = Conv(feature_channels[3]//8, feature_channels[3]//16, kernel_size=3, stride=1, padding=1, bias=False)
        self.decoder1_5 = Conv(feature_channels[3]//16, 1, kernel_size=3, stride=1, padding=1, bias=False)

        self.decoder2_up1 = upConv(feature_channels[3], feature_channels[3]//2, scale_factor=2)
        self.bfe2 = BFE(feature_channels[3]//2 + feature_channels[2], feature_channels[2], iterations=1)
        self.attn2 = HFA(feature_channels[2])
        self.decoder2_1 = Conv(feature_channels[2], feature_channels[2]//2, kernel_size=3, stride=1, padding=1, bias=False)
        self.decoder2_2 = Conv(feature_channels[2]//2, feature_channels[2]//4, kernel_size=3, stride=1, padding=1, bias=False)
        self.decoder2_3 = Conv(feature_channels[2]//4, feature_channels[2]//8, kernel_size=3, stride=1, padding=1, bias=False)
        self.decoder2_4 = Conv(feature_channels[2]//8, feature_channels[2]//16, kernel_size=3, stride=1, padding=1, bias=False)
        self.decoder2_5 = Conv(feature_channels[2]//16, 1, kernel_size=3, stride=1, padding=1, bias=False)

        self.decoder3_up2 = upConv(feature_channels[2], feature_channels[2]//2, scale_factor=2)
        self.bfe3 = BFE(feature_channels[2]//2 + feature_channels[1], feature_channels[1], iterations=1)
        self.attn3 = HFA(feature_channels[1])
        self.decoder3_1 = Conv(feature_channels[1], feature_channels[1]//2, kernel_size=3, stride=1, padding=1, bias=False)        
        self.decoder3_2 = Conv(feature_channels[1]//2, feature_channels[1]//4, kernel_size=3, stride=1, padding=1, bias=False)
        self.decoder3_3 = Conv(feature_channels[1]//4, feature_channels[1]//8, kernel_size=3, stride=1, padding=1, bias=False)
        self.decoder3_4 = Conv(feature_channels[1]//8, 1, kernel_size=3, stride=1, padding=1, bias=False)

        self.decoder4_up3 = upConv(feature_channels[1], feature_channels[1]//2, scale_factor=2)
        self.bfe4 = BFE(feature_channels[1]//2 + feature_channels[0], feature_channels[0], iterations=1)
        self.attn4 = HFA(feature_channels[0])
        
        self.decoder5_up4 = upConv(feature_channels[0], feature_channels[0]//2, scale_factor=2)
        self.decoder5_1 = Conv(feature_channels[0]//2, feature_channels[0]//4, kernel_size=3, stride=1, padding=1, bias=False)
        self.decoder5_2 = Conv(feature_channels[0]//4, num_classes, kernel_size=3, stride=1, padding=1, bias=False)

        
    def forward(self, features):
        low_1, high_1 = self.sep1(features[3])
        low_2, high_2 = self.sep2(features[2])
        low_3, high_3 = self.sep3(features[1])
        low_4, high_4 = self.sep4(features[0])

        x1 = low_1
        x1 = self.attn1(x1, high_1)
        x1_up = self.decoder2_up1(x1)
        x1 = self.decoder1_1(x1)
        x1 = self.decoder1_2(x1)
        x1 = self.decoder1_3(x1)
        x1 = self.decoder1_4(x1)
        pred1 = self.decoder1_5(x1)

        x2 = torch.cat([low_2, x1_up], dim=1)
        x2 = self.bfe2(x2, pred1)
        x2 = self.attn2(x2, high_2)
        x2_up = self.decoder3_up2(x2)
        x2 = self.decoder2_1(x2)
        x2 = self.decoder2_2(x2)
        x2 = self.decoder2_3(x2)
        x2 = self.decoder2_4(x2)
        pred2 = self.decoder2_5(x2)
        
        x3 = torch.cat([low_3, x2_up], dim=1)
        x3 = self.bfe3(x3, pred2)
        x3 = self.attn3(x3, high_3)
        x3_up = self.decoder4_up3(x3)
        x3 = self.decoder3_1(x3)
        x3 = self.decoder3_2(x3)
        x3 = self.decoder3_3(x3)
        pred3 = self.decoder3_4(x3)
        
        x4 = torch.cat([low_4, x3_up], dim=1)
        x4 = self.bfe4(x4, pred3)
        x4 = self.attn4(x4, high_4)
        x4_up = self.decoder5_up4(x4)

        x5 = self.decoder5_1(x4_up)
        pred_final = self.decoder5_2(x5)
        output = F.interpolate(pred_final, scale_factor = 2, mode = 'bilinear')
        
        return output