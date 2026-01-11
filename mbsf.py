import torch
import torch.nn as nn

class MBSF(nn.Module):
    def __init__(self, c1, c2, dilation=2):
        super().__init__()
        self.c2 = c2
        self.dilation = dilation
        self.residual_proj = nn.Identity() if c1 == c2 else nn.Conv2d(c1, c2, 1, 1, 0, bias=False)
        self._build(c1)

    def _build(self, c1):
        def cb(in_c, out_c, k, s=1, d=1, p=None):
            if isinstance(k, tuple):
                if p is None:
                    p = (k[0] // 2, k[1] // 2)
            else:
                if p is None:
                    p = d * (k - 1) // 2
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, k, s, p, d, bias=False),
                nn.BatchNorm2d(out_c),
                nn.SiLU()
            )
        c_branch = self.c2 // 4
        self.b1 = nn.Sequential(
            cb(c1, c_branch, 1),
            cb(c_branch, c_branch, (1, 3)),
            cb(c_branch, c_branch, (3, 1)),
            cb(c_branch, c_branch, 3, d=self.dilation)
        )
        self.b2 = nn.Sequential(
            cb(c1, c_branch, 1),
            cb(c_branch, c_branch, (3, 1)),
            cb(c_branch, c_branch, (1, 3)),
            cb(c_branch, c_branch, 3, d=self.dilation)
        )
        self.b3 = nn.Sequential(
            cb(c1, c_branch, 1),
            cb(c_branch, c_branch, 3)
        )
        self.b4 = cb(c1, self.c2, 1)
        self.cat_proj = cb(3 * c_branch, self.c2, 1)

    def _split(self, x):
        return (
            x[:, :, ::2, ::2],
            x[:, :, ::2, 1::2],
            x[:, :, 1::2, ::2],
            x[:, :, 1::2, 1::2],
        )
    def _merge(self, q0, q1, q2, q3, H, W):
        b, c, h, w = q0.shape
        y = torch.zeros(b, c, H, W, device=q0.device, dtype=q0.dtype)
        y[:, :, ::2, ::2] = q0
        y[:, :, ::2, 1::2] = q1
        y[:, :, 1::2, ::2] = q2
        y[:, :, 1::2, 1::2] = q3
        return y
    def forward(self, x):
        b, c, H, W = x.shape
        identity = x
        q0, q1, q2, q3 = self._split(x)
        def proc(q):
            y1 = self.b1(q)
            y2 = self.b2(q)
            y3 = self.b3(q)
            y_cat = torch.cat((y1, y2, y3), dim=1)
            y_cat = self.cat_proj(y_cat)
            y4 = self.b4(q)
            return y_cat + y4
        o0 = proc(q0)
        o1 = proc(q1)
        o2 = proc(q2)
        o3 = proc(q3)
        out = self._merge(o0, o1, o2, o3, H, W)
        return out + self.residual_proj(identity)
