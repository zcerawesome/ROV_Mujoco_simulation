"""PipeNet: a small encoder-decoder that predicts, per pixel, whether it is pipe.
The bounding box is read off that mask (detector.py), which gives the network
dense supervision instead of four numbers per image - a lot harder to overfit.
Training-only; at runtime the exported ONNX runs through OpenCV."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def block(cin, cout, stride=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride, padding=dilation, dilation=dilation, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True))


class PipeNet(nn.Module):
    def __init__(self, c=(24, 48, 96, 128), dropout=0.1):
        super().__init__()
        self.e1 = nn.Sequential(block(3, c[0], 2), block(c[0], c[0]))            # 1/2
        self.e2 = nn.Sequential(block(c[0], c[1], 2), block(c[1], c[1]))         # 1/4
        self.e3 = nn.Sequential(block(c[1], c[2], 2), block(c[2], c[2]))         # 1/8
        # 1/16, dilated: the whole frame is in view, so a pipe is judged in context
        self.e4 = nn.Sequential(block(c[2], c[3], 2), block(c[3], c[3], dilation=2),
                                block(c[3], c[3], dilation=4), nn.Dropout2d(dropout))
        self.d3 = block(c[3] + c[2], c[2])
        self.d2 = block(c[2] + c[1], c[1])
        self.d1 = block(c[1] + c[0], c[0])
        self.head = nn.Conv2d(c[0], 1, 1)

    def forward(self, x):
        """x: N x 3 x H x W raw RGB in 0..255. Returns N x 1 x H x W logits."""
        x = x / 127.5 - 1
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        y = self.e4(e3)
        up = lambda a: F.interpolate(a, scale_factor=2, mode="bilinear", align_corners=False)
        y = self.d3(torch.cat([up(y), e3], 1))
        y = self.d2(torch.cat([up(y), e2], 1))
        y = self.d1(torch.cat([up(y), e1], 1))
        return up(self.head(y))
