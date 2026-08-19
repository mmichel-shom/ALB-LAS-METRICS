import torch
import torch.nn as nn
import spconv.pytorch as spconv
from typing import Optional, Union, Sequence, Tuple

# ----------------------------
# Utils
# ----------------------------
def to_tuple(v: Union[int, Sequence[int]]) -> Tuple[int, int, int]:
    if isinstance(v, int):
        return (v, v, v)
    if isinstance(v, (list, tuple)):
        if len(v) != 3:
            raise ValueError("Expected int or sequence of length 3")
        return tuple(int(x) for x in v)
    raise ValueError("Expected int or sequence of length 3")

def compute_padding(kernel: Union[int, Sequence[int]], dilation: Union[int, Sequence[int]]) -> Tuple[int, int, int]:
    kx, ky, kz = to_tuple(kernel)
    dx, dy, dz = to_tuple(dilation)

    px = ((kx - 1) // 2) * dx
    py = ((ky - 1) // 2) * dy
    pz = ((kz - 1) // 2) * dz
    return (px, py, pz)


# ----------------------------
# Residual Block
# ----------------------------
class ResBlock(spconv.SparseModule):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_fn,
        indice_key: Optional[str] = None,
        kernel_size: Union[int, Sequence[int]] = 3,
        dilation: Union[int, Sequence[int]] = 1,
    ):
        super().__init__()

        kernel_size_t = to_tuple(kernel_size)
        dilation_t = to_tuple(dilation)
        padding_t = compute_padding(kernel_size_t, dilation_t)

        self.proj = None
        if in_channels != out_channels:
            self.proj = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, out_channels, kernel_size=1, bias=False, indice_key=indice_key),
                norm_fn(out_channels),
            )

        self.conv1 = spconv.SubMConv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size_t,
            padding=padding_t,
            bias=False,
            indice_key=indice_key,
            dilation=dilation_t,
        )
        self.bn1 = norm_fn(out_channels)
        self.conv2 = spconv.SubMConv3d(
            out_channels,
            out_channels,
            kernel_size=kernel_size_t,
            padding=padding_t,
            bias=False,
            indice_key=indice_key,
            dilation=dilation_t,
        )
        self.bn2 = norm_fn(out_channels)
        self.relu = nn.ReLU(True)

    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        residual = x.features if self.proj is None else self.proj(x).features
        out = self.conv1(x)
        out = out.replace_feature(self.relu(self.bn1(out.features)))
        out = self.conv2(out)
        out = out.replace_feature(self.relu(self.bn2(out.features) + residual))
        return out


# ----------------------------
# Residual Attention Gate
# ----------------------------
class AttentionGateSparse(nn.Module):
    def __init__(self, F_g: int, F_l: int, F_int: int, indice_key: Optional[str] = None):
        super().__init__()

        self.W_g = spconv.SubMConv3d(F_g, F_int, kernel_size=1, bias=False, indice_key=indice_key)
        self.W_x = spconv.SubMConv3d(F_l, F_int, kernel_size=1, bias=False, indice_key=indice_key)
        self.psi = spconv.SubMConv3d(F_int, 1, kernel_size=1, bias=True, indice_key=indice_key)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        nn.init.kaiming_normal_(self.W_g.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.W_x.weight, nonlinearity="relu")
        nn.init.constant_(self.psi.weight, 0.0)
        if self.psi.bias is not None:
            nn.init.constant_(self.psi.bias, 0.0)

    def forward(self, g: spconv.SparseConvTensor, x: spconv.SparseConvTensor):
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        combined = self.relu(g1.features + x1.features)
        psi = self.psi(g1.replace_feature(combined))
        alpha = self.sigmoid(psi.features)

        out_feats = x.features * (1 + alpha)
        return x.replace_feature(out_feats)


# ----------------------------
# SpConvUNet
# ----------------------------
class SpConvUNet(nn.Module):
    """ 
    Transposition de MinkUNetBase (8 étapes) en SpConv. 
    Compatible avec les variantes 14/18/34/50/101 en modifiant LAYERS et PLANES. 
    SpConvUnetSmall:
    INIT_DIM = 16
    PLANES = (16, 32, 64, 128, 128, 64, 32, 16)
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
    (*) SpConvUNet14A: 
    PLANES = (32, 64, 128, 256, 128, 128, 96, 96) 
    LAYERS = (1, 1, 1, 1, 1, 1, 1, 1) 
    SpConvUNet18B: 
    PLANES = (32, 64, 128, 256, 128, 128, 128, 128) 
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2) 
    SpConvUNet34A: 
    PLANES = (32, 64, 128, 256, 256, 128, 64, 64) 
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2) 
    SpConvUNet50 (Bottleneck): 
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)
    LAYERS = (3, 4, 6, 3, 2, 2, 2, 2) 
    """ 
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        BLOCK = ResBlock,
        INIT_DIM: int = 32,
        PLANES: Tuple[int, ...] = (32, 64, 128, 256, 128, 128, 96, 96),
        LAYERS: Tuple[int, ...] = (1, 1, 1, 1, 1, 1, 1, 1),

        KERNEL_SIZE: Sequence[Tuple[int, int, int]] = ((3, 3, 3),) * 8,
        DILATION:    Sequence[Tuple[int, int, int]] = ((1, 1, 1),) * 8, # PLEASE DO NOT TOUCH

        KERNEL_SIZE_BLOCK: Sequence[Tuple[int, int, int]] = ((3, 3, 3),) * 8,
        DILATION_BLOCK:    Sequence[Tuple[int, int, int]] = ((1, 1, 1),) * 8,

        norm_groups: int = 8,
        use_attention: bool = True,
        use_ds: bool = True,
    ):
        super().__init__()

        self.BLOCK = BLOCK
        self.PLANES = PLANES
        self.LAYERS = LAYERS
        self.INIT_DIM = INIT_DIM
        self.use_attention = use_attention
        self.use_ds = use_ds

        # normalize kernels/dilations
        self.KERNEL_SIZE = [to_tuple(k) for k in KERNEL_SIZE]
        self.DILATION = [to_tuple(d) for d in DILATION]
        self.KERNEL_SIZE_BLOCK = [to_tuple(kb) for kb in KERNEL_SIZE_BLOCK]
        self.DILATION_BLOCK = [to_tuple(db) for db in DILATION_BLOCK]

        norm_fn = lambda c: nn.GroupNorm(num_groups=min(norm_groups, c), num_channels=c)
        self.relu = nn.ReLU(True)

        # === ENCODER ===
        self.conv0p1s1 = spconv.SubMConv3d(in_channels, INIT_DIM, kernel_size=5, padding=2, bias=False, indice_key="conv0")
        self.bn0 = norm_fn(INIT_DIM)

        PADDING = compute_padding(self.KERNEL_SIZE[0], self.DILATION[0])
        self.conv1p1s2 = spconv.SparseConv3d(INIT_DIM, INIT_DIM, kernel_size=self.KERNEL_SIZE[0], stride=2, padding=PADDING, bias=False, indice_key="down1", dilation=self.DILATION[0])
        self.bn1 = norm_fn(INIT_DIM)
        self.block1 = self._make_layer(BLOCK, INIT_DIM, PLANES[0], LAYERS[0], norm_fn, "enc1",
                                       kernel_size=self.KERNEL_SIZE_BLOCK[0], dilation=self.DILATION_BLOCK[0])

        PADDING = compute_padding(self.KERNEL_SIZE[1], self.DILATION[1])
        self.conv2p2s2 = spconv.SparseConv3d(PLANES[0], PLANES[0], kernel_size=self.KERNEL_SIZE[1], stride=2, padding=PADDING, bias=False, indice_key="down2", dilation=self.DILATION[1])
        self.bn2 = norm_fn(PLANES[0])
        self.block2 = self._make_layer(BLOCK, PLANES[0], PLANES[1], LAYERS[1], norm_fn, "enc2",
                                       kernel_size=self.KERNEL_SIZE_BLOCK[1], dilation=self.DILATION_BLOCK[1])

        PADDING = compute_padding(self.KERNEL_SIZE[2], self.DILATION[2])
        self.conv3p4s2 = spconv.SparseConv3d(PLANES[1], PLANES[1], kernel_size=self.KERNEL_SIZE[2], stride=2, padding=PADDING, bias=False, indice_key="down3", dilation=self.DILATION[2])
        self.bn3 = norm_fn(PLANES[1])
        self.block3 = self._make_layer(BLOCK, PLANES[1], PLANES[2], LAYERS[2], norm_fn, "enc3",
                                       kernel_size=self.KERNEL_SIZE_BLOCK[2], dilation=self.DILATION_BLOCK[2])

        PADDING = compute_padding(self.KERNEL_SIZE[3], self.DILATION[3])
        self.conv4p8s2 = spconv.SparseConv3d(PLANES[2], PLANES[2], kernel_size=self.KERNEL_SIZE[3], stride=2, padding=PADDING, bias=False, indice_key="down4", dilation=self.DILATION[3])
        self.bn4 = norm_fn(PLANES[2])
        self.block4 = self._make_layer(BLOCK, PLANES[2], PLANES[3], LAYERS[3], norm_fn, "enc4",
                                       kernel_size=self.KERNEL_SIZE_BLOCK[3], dilation=self.DILATION_BLOCK[3])

        # === DECODER ===
        self.convtr4p16s2 = spconv.SparseInverseConv3d(PLANES[3], PLANES[4], kernel_size=self.KERNEL_SIZE[4], bias=False, indice_key="down4")
        self.bntr4 = norm_fn(PLANES[4])
        self.block5 = self._make_layer(BLOCK, PLANES[4] + PLANES[2], PLANES[4], LAYERS[4], norm_fn, "dec4",
                                       kernel_size=self.KERNEL_SIZE_BLOCK[4], dilation=self.DILATION_BLOCK[4])

        self.convtr5p8s2 = spconv.SparseInverseConv3d(PLANES[4], PLANES[5], kernel_size=self.KERNEL_SIZE[5], bias=False, indice_key="down3")
        self.bntr5 = norm_fn(PLANES[5])
        self.block6 = self._make_layer(BLOCK, PLANES[5] + PLANES[1], PLANES[5], LAYERS[5], norm_fn, "dec3",
                                       kernel_size=self.KERNEL_SIZE_BLOCK[5], dilation=self.DILATION_BLOCK[5])

        self.convtr6p4s2 = spconv.SparseInverseConv3d(PLANES[5], PLANES[6], kernel_size=self.KERNEL_SIZE[6], bias=False, indice_key="down2")
        self.bntr6 = norm_fn(PLANES[6])
        self.block7 = self._make_layer(BLOCK, PLANES[6] + PLANES[0], PLANES[6], LAYERS[6], norm_fn, "dec2",
                                       kernel_size=self.KERNEL_SIZE_BLOCK[6], dilation=self.DILATION_BLOCK[6])

        self.convtr7p2s2 = spconv.SparseInverseConv3d(PLANES[6], PLANES[7], kernel_size=self.KERNEL_SIZE[7], bias=False, indice_key="down1")
        self.bntr7 = norm_fn(PLANES[7])
        self.block8 = self._make_layer(BLOCK, PLANES[7] + INIT_DIM, PLANES[7], LAYERS[7], norm_fn, "dec1",
                                       kernel_size=self.KERNEL_SIZE_BLOCK[7], dilation=self.DILATION_BLOCK[7])

        # === FINAL ===
        self.final = spconv.SubMConv3d(PLANES[7], out_channels, kernel_size=1, bias=True)

        # === DEEP SUPERVISION ===
        if self.use_ds:
            self.ds8 = spconv.SubMConv3d(PLANES[4], out_channels, kernel_size=1, bias=True)
            self.ds4 = spconv.SubMConv3d(PLANES[5], out_channels, kernel_size=1, bias=True)
            self.ds2 = spconv.SubMConv3d(PLANES[6], out_channels, kernel_size=1, bias=True)
        else:
            self.ds8 = self.ds4 = self.ds2 = None

        # === ATTENTION ===
        if self.use_attention:
            self.att4 = AttentionGateSparse(F_g=PLANES[4], F_l=PLANES[2], F_int=max(1, PLANES[2]//2), indice_key="down4")
            self.att3 = AttentionGateSparse(F_g=PLANES[5], F_l=PLANES[1], F_int=max(1, PLANES[1]//2), indice_key="down3")
            self.att2 = AttentionGateSparse(F_g=PLANES[6], F_l=PLANES[0], F_int=max(1, PLANES[0]//2), indice_key="down2")
            self.att1 = AttentionGateSparse(F_g=PLANES[7], F_l=INIT_DIM, F_int=max(1, INIT_DIM//2), indice_key="down1")
        else:
            self.att4 = self.att3 = self.att2 = self.att1 = None

        self._init_weights()

    def _make_layer(self, block, in_planes, out_planes, blocks, norm_fn, indice_key=None,
                    kernel_size: Union[int, Sequence[int]] = 3, dilation: Union[int, Sequence[int]] = 1):
        layers = [block(in_planes, out_planes, norm_fn, indice_key=indice_key, kernel_size=kernel_size, dilation=dilation)]
        for _ in range(1, blocks):
            layers.append(block(out_planes, out_planes, norm_fn, indice_key=indice_key, kernel_size=kernel_size, dilation=dilation))
        return spconv.SparseSequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (spconv.SubMConv3d, spconv.SparseConv3d, spconv.SparseInverseConv3d)):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: spconv.SparseConvTensor, return_ds_levels=()):
        # === Encoder ===
        out = self.conv0p1s1(x)
        out = out.replace_feature(self.relu(self.bn0(out.features)))
        out_p1 = out

        out = self.conv1p1s2(out)
        out = out.replace_feature(self.relu(self.bn1(out.features)))
        out_b1p2 = self.block1(out)

        out = self.conv2p2s2(out_b1p2)
        out = out.replace_feature(self.relu(self.bn2(out.features)))
        out_b2p4 = self.block2(out)

        out = self.conv3p4s2(out_b2p4)
        out = out.replace_feature(self.relu(self.bn3(out.features)))
        out_b3p8 = self.block3(out)

        out = self.conv4p8s2(out_b3p8)
        out = out.replace_feature(self.relu(self.bn4(out.features)))
        out_b4p16 = self.block4(out)

        # === Decoder ===
        out = self.convtr4p16s2(out_b4p16)
        out = out.replace_feature(self.relu(self.bntr4(out.features)))
        skip = self.att4(out, out_b3p8) if self.use_attention else out_b3p8
        out = out.replace_feature(torch.cat([out.features, skip.features], dim=1))
        out_b3p8_dec = self.block5(out)

        # DS @ p8
        ds8 = self.ds8(out_b3p8_dec) if self.use_ds else None

        out = self.convtr5p8s2(out_b3p8_dec)
        out = out.replace_feature(self.relu(self.bntr5(out.features)))
        skip = self.att3(out, out_b2p4) if self.use_attention else out_b2p4
        out = out.replace_feature(torch.cat([out.features, skip.features], dim=1))
        out_b2p4_dec = self.block6(out)

        # DS @ p4
        ds4 = self.ds4(out_b2p4_dec) if self.use_ds else None

        out = self.convtr6p4s2(out_b2p4_dec)
        out = out.replace_feature(self.relu(self.bntr6(out.features)))
        skip = self.att2(out, out_b1p2) if self.use_attention else out_b1p2
        out = out.replace_feature(torch.cat([out.features, skip.features], dim=1))
        out_b1p2_dec = self.block7(out)

        # DS @ p2
        ds2 = self.ds2(out_b1p2_dec) if self.use_ds else None

        out = self.convtr7p2s2(out_b1p2_dec)
        out = out.replace_feature(self.relu(self.bntr7(out.features)))
        skip = self.att1(out, out_p1) if self.use_attention else out_p1
        out = out.replace_feature(torch.cat([out.features, skip.features], dim=1))
        out_p1_dec = self.block8(out)

        # final
        final_out = self.final(out_p1_dec)

        ds_dict = {"ds2": ds2, "ds4": ds4, "ds8": ds8}
        selected_ds = tuple(ds_dict[name] for name in return_ds_levels if name in ds_dict)

        return (final_out, *selected_ds)
