import torch
from torch import nn

from models.DynamicNet_multi.block import DRF, Mlp, LayerScale, Gate
from models.DynamicNet_multi.swin_attn import SwinTransformer
from models.DynamicNet_multi.tem_attn import Tem_attn
from models.layers.layer_utils import ACTIVATIONS


class Early_feature(nn.Module):
    def __init__(self, in_channels, out_channels, re_field, input_resolution, use_swin=False,
                 patch_size=4,  field_size=None, act='leakyrelu', local_conv=False, mlp_ratio=4):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.input_resolution = int(input_resolution)
        self.hid_channels = out_channels // (patch_size**2)
        self.local_conv = local_conv
        self.re_field = re_field
        self.use_swin = use_swin
        if re_field:
            field_size = field_size if field_size is not None else [3, 5]
            self.local = DRF(n_images=in_channels, out_channels=out_channels, patch_sizes=self.patch_size,
                             field_sizes=field_size, )
            self.act = ACTIVATIONS[act]()
            proj_channels = out_channels * len(field_size)
            if local_conv:
                self.local_conv_ = nn.Sequential(
                    nn.Conv2d(in_channels, self.hid_channels, 1, bias=True),
                    nn.Conv2d(self.hid_channels, self.hid_channels, 3, padding=3 // 2, bias=True),
                    nn.BatchNorm2d(self.hid_channels),
                )
                proj_channels = out_channels * (len(field_size) + 1)
            if use_swin:
                self.swin_conv = nn.Conv2d(in_channels*patch_size*patch_size, self.out_channels, 1, bias=True)
                self.swin_block = SwinTransformer(
                    self.out_channels, num_heads=8, qkv_bias=True, input_resolution=self.input_resolution,
                    act='gelu', window_size=4, mlp_ratio=mlp_ratio,
                )
                self.norm = nn.LayerNorm(out_channels)
            # proj_channels = out_channels * len(field_size)
            self.project = nn.Conv2d(in_channels=proj_channels, out_channels=out_channels, kernel_size=1, )
        else:
            hid_channels = in_channels*patch_size*patch_size
            self.vision_encoder = Mlp(hid_channels, hidden_features=int(hid_channels*mlp_ratio),
                                 out_features=out_channels)

    def forward(self, x):
        B, T, C, H, W = x.shape
        if self.re_field:
            map = self.act(self.local(x))
            if self.local_conv:
                local_map = (self.act(self.local_conv_(x.view(B * T, C, H, W)))
                             .view(B * T, self.hid_channels,
                                      H // self.input_resolution, self.input_resolution,
                                      W // self.input_resolution, self.input_resolution,
                                      )
                             .permute(0, 1, 2, 4, 3, 5).contiguous()
                             .view(B * T, self.out_channels, self.input_resolution, self.input_resolution)
                             )
                map = torch.cat((map, local_map), dim=1)
            x_patches = self.project(map)
            # Patchify
            x_patches = (x_patches.view(B, T, self.out_channels, self.input_resolution, self.input_resolution)
                         .permute(0, 3, 4, 1, 2).contiguous().view(-1, T, self.out_channels))

            if self.use_swin:
                global_map = (x.view(B * T, C, self.patch_size, self.input_resolution,
                                    self.patch_size, self.input_resolution)
                              .permute(0, 1, 2, 4, 3, 5).contiguous()
                              .view(B*T, C*self.patch_size*self.patch_size, self.input_resolution, self.input_resolution))
                global_map = (self.swin_conv(global_map)
                              .view(B, T, self.out_channels, self.input_resolution, self.input_resolution)
                              .permute(0, 3, 4, 1, 2).contiguous()
                              .view(-1, T, self.out_channels)
                              )
                global_map = (self.swin_block(global_map)
                              .view(B, T, self.input_resolution*self.input_resolution, self.out_channels)
                              .permute(0, 2, 1, 3).contiguous()
                              .view(-1, T, self.out_channels))

                x_patches = self.norm(x_patches + global_map)
        else:
            image_patches = (
                x.reshape(B, T, C,
                          H // self.patch_size, self.patch_size,
                          W // self.patch_size, self.patch_size,
                )
                .permute(0, 3, 5, 1, 2, 4, 6)
                .reshape(
                    B * H // self.patch_size * W // self.patch_size,
                    T,
                    C * self.patch_size * self.patch_size,
                )
            )
            x_patches = self.vision_encoder(image_patches)

        return x_patches  # (B * H * W, T, C)


class Mid_feature(nn.Module):
    def __init__(
        self,
        dim, num_heads, diffuse=True, mlp_ratio=4.0, qkv_bias=True, qk_norm=False, drop=0.1, attn_drop=0.1,
        init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,  window_size=7, tem_act=False,
        width=32, spa_act='gelu', fast_attn=False,
    ):
        super().__init__()
        self.tem_act = tem_act
        self.attn = Tem_attn(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm,
            attn_drop=attn_drop, proj_drop=drop, norm_layer=norm_layer, fast_attn=fast_attn,
        )
        self.input_resolution = width
        self.diffuse = diffuse

        if self.diffuse:
            self.spa_act = spa_act
            self.ls3 = (
                LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
            )
            self.spa_attn = SwinTransformer(
                dim, num_heads=num_heads, qkv_bias=qkv_bias,  input_resolution=width,
                act=self.spa_act, window_size=window_size,
            )

        if self.tem_act:
            self.gate = Gate(
                in_features=dim, hidden_features=int(dim * mlp_ratio), out_features=dim,
                act_layer=nn.SiLU, drop=drop,
            )
            self.norm3 = norm_layer(dim)
            self.ls4 = (
                LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
            )

        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        self.mlp = Mlp(
            in_features=dim, hidden_features=int(dim * mlp_ratio), out_features=dim,
            act_layer=act_layer, drop=drop,
        )

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        if self.tem_act:
            x = x + self.ls4(self.gate(self.norm3(x)))
        if self.diffuse:
            x = x + self.ls3(self.spa_attn(x))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x

