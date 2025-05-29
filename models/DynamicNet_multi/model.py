import argparse
import warnings
from typing import Optional, Union
import torch
import torch.nn as nn

from models.DynamicNet_multi.main_frame import Early_feature, Mid_feature
from models.layers.blocks import Mlp
from utils.tools import str2bool, get_sinusoid_encoding_table

warnings.filterwarnings("ignore", category=UserWarning)


class DynamicNet_multi(nn.Module):

    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        input_resolution = int(self.hparams.input_size / self.hparams.patch_size)
        self.early_feature = Early_feature(
            in_channels=self.hparams.n_image, out_channels=self.hparams.n_hidden,
            re_field=self.hparams.re_field, act=self.hparams.re_act, local_conv=self.hparams.local_conv,
            patch_size=self.hparams.patch_size, input_resolution=input_resolution, use_swin=self.hparams.use_swin,
        )

        if self.hparams.use_weather:
            self.embed_weather = Mlp(
                in_features=self.hparams.n_weather, hidden_features=self.hparams.n_hidden,
                out_features=self.hparams.n_hidden,
            )
        self.mask_token = nn.Parameter(torch.zeros(self.hparams.n_hidden))
        self.mid_feature = nn.ModuleList(
            [
                Mid_feature(
                    self.hparams.n_hidden, self.hparams.n_heads,
                    qkv_bias=True, norm_layer=nn.LayerNorm, fast_attn=self.hparams.fast_attn, diffuse=self.hparams.diffuse,
                    spa_act=self.hparams.spa_act, width=input_resolution,
                    window_size=self.hparams.window_size,
                )
                for _ in range(self.hparams.depth)
            ]
        )

        self.head = Mlp(
            in_features=self.hparams.n_hidden,
            hidden_features=self.hparams.n_hidden,
            out_features=self.hparams.n_out * self.hparams.patch_size * self.hparams.patch_size,
        )

    @staticmethod
    def add_model_specific_args(
        parent_parser: Optional[Union[argparse.ArgumentParser, list]] = None
    ):
        if parent_parser is None:
            parent_parser = []
        elif not isinstance(parent_parser, list):
            parent_parser = [parent_parser]

        parser = argparse.ArgumentParser(parents=parent_parser, add_help=False)

        parser.add_argument("--context_length", type=int, default=10)
        parser.add_argument("--target_length", type=int, default=20)
        parser.add_argument("--patch_size", type=int, default=4)
        parser.add_argument("--input_size", type=int, default=128)
        parser.add_argument("--n_image", type=int, default=8)
        parser.add_argument("--n_weather", type=int, default=24)
        parser.add_argument("--n_hidden", type=int, default=256)
        parser.add_argument("--n_out", type=int, default=1)
        parser.add_argument("--n_heads", type=int, default=8)
        parser.add_argument("--depth", type=int, default=3)
        parser.add_argument("--mlp_ratio", type=float, default=4.0)
        parser.add_argument("--mtm", type=str2bool, default=True)
        parser.add_argument("--leave_n_first", type=int, default=3)
        parser.add_argument("--p_mtm", type=float, default=0.5)
        parser.add_argument("--p_use_mtm", type=float, default=0.7)
        parser.add_argument("--mask_clouds", type=str2bool, default=True)
        parser.add_argument("--use_weather", type=str2bool, default=True)
        parser.add_argument("--re_field", type=str2bool, default=True)
        parser.add_argument("--re_act", type=str, default='leakyrelu')

        parser.add_argument("--fast_attn", type=str2bool, default=False)
        parser.add_argument("--local_conv", type=str2bool, default=False)
        parser.add_argument("--use_swin", type=str2bool, default=False)
        parser.add_argument("--diffuse", type=str2bool, default=True)
        parser.add_argument("--spa_act", type=str, default='gelu')
        parser.add_argument("--window_size", type=int, default=8)
        parser.add_argument("--add_last_ndvi", type=str2bool, default=False)

        return parser

    def forward(self, data, pred_start=10, preds_length=None):
        # Input handling
        # preds_length=0 in the training phase and preds_length=20 in the test phase.
        preds_length = self.hparams.target_length if preds_length is None else preds_length

        c_l = self.hparams.context_length if self.training else pred_start

        hr_dynamic_inputs = data["dynamic"][0]  # (B, T, C, H, W)
        hr_dynamic_mask = data["dynamic_mask"][0]  # (B, T, C, H, W)

        B, T, C, H, W = hr_dynamic_inputs.shape  # Explain the specific meaning of C

        if T == c_l:
            hr_dynamic_inputs = torch.cat(
                (hr_dynamic_inputs,
                    torch.zeros(B, preds_length, C, H, W, device=hr_dynamic_inputs.device)),
                dim=1)
            hr_dynamic_mask = torch.cat(
                (hr_dynamic_mask,
                    torch.zeros(B, preds_length, 1, H, W, device=hr_dynamic_mask.device)),
                dim=1)
            B, T, C, H, W = hr_dynamic_inputs.shape

        static_inputs = data["static"][0][:, :3, ...]  # (B, C, H, W)

        weather = data["dynamic"][1]
        if len(weather.shape) == 3:
            _, t_m, c_m = weather.shape  # Ask the c_m value
        else:
            _, t_m, c_m, h2, w2 = weather.shape
            weather = weather.reshape(B, t_m // 5, 5, c_m, h2, w2).mean(dim=2).mean(dim=(3, 4))
            _, t_m, c_m = weather.shape
        # B, C, H, W = static_inputs.shape --> B, T, C, H, W
        static_inputs = static_inputs.unsqueeze(1).repeat(1, T, 1, 1, 1)

        images = torch.cat([hr_dynamic_inputs, static_inputs], dim=2)
        B, T, C, H, W = images.shape  # 
        h = H // self.hparams.patch_size
        w = W // self.hparams.patch_size
        # Patchify
        image_patches_embed = self.early_feature(images)
        B_patch, N_patch, C_patch = image_patches_embed.shape

        mask_patches = (
            hr_dynamic_mask.reshape(B, T, 1,
                                    h, self.hparams.patch_size,
                                    w, self.hparams.patch_size,)
            .permute(0, 3, 5, 1, 2, 4, 6)
            .reshape(B * h * w,
                     T,
                     1 * self.hparams.patch_size * self.hparams.patch_size)
        )

        weather_patches = (
            weather.reshape(B, 1, t_m, c_m)
            .repeat(1, h * w, 1, 1)
            .reshape(B_patch, t_m, c_m)
        )

        # Add Token Mask

        if self.hparams.mtm and self.training and ((torch.rand(1) <= self.hparams.p_use_mtm).all()):
            token_mask = (
                (torch.rand(B_patch, N_patch, device=weather_patches.device) < self.hparams.p_mtm)
                .type_as(weather_patches)  # A common way to ensure tensor type consistency and device consistency
                .reshape(B_patch, N_patch, 1)
                .repeat(1, 1, self.hparams.n_hidden)
            )
            token_mask[:, : self.hparams.leave_n_first] = 0

        else:
            token_mask = (
                torch.ones(B_patch, N_patch, device=weather_patches.device)
                .type_as(weather_patches)
                .reshape(B_patch, N_patch, 1)
                .repeat(1, 1, self.hparams.n_hidden)
            )
            token_mask[:, :c_l] = 0

        image_patches_embed[token_mask.bool()] = (
            self.mask_token
            .reshape(1, 1, self.hparams.n_hidden)
            .repeat(B_patch, N_patch, 1)[token_mask.bool()]
        )

        if self.hparams.mask_clouds:
            cloud_mask = (
                (mask_patches.max(-1, keepdim=True)[0] > 0)
                .bool()
                .repeat(1, 1, self.hparams.n_hidden)
            )

            image_patches_embed[cloud_mask] = (
                self.mask_token
                .reshape(1, 1, self.hparams.n_hidden)
                .repeat(B_patch, N_patch, 1)[cloud_mask]
            )

        # Add Image and Weather Embeddings
        if self.hparams.use_weather:
            weather_patches_embed = self.embed_weather(weather_patches)
            patches_embed = image_patches_embed + weather_patches_embed
        else:
            patches_embed = image_patches_embed

        t_pos_embed = (
            get_sinusoid_encoding_table(N_patch, self.hparams.n_hidden, T=1000)
            .to(patches_embed.device)
            .unsqueeze(0)
            .repeat(B_patch, 1, 1)
        )
        x = patches_embed + t_pos_embed

        for blk in self.mid_feature:
            x = blk(x)

        x_out = self.head(x)

        images_out = (
            x_out.reshape(
                B, H // self.hparams.patch_size, W // self.hparams.patch_size, N_patch,
                self.hparams.n_out, self.hparams.patch_size, self.hparams.patch_size,
            )
            .permute(0, 3, 4, 1, 5, 2, 6)
            .reshape(B, N_patch, self.hparams.n_out, H, W)
        )

        if self.hparams.add_last_ndvi:
            mask = hr_dynamic_mask[:, :c_l, ...]

            indxs = (
                torch.arange(c_l, device=mask.device)
                .expand(B, self.hparams.n_out, H, W, -1)
                .permute(0, 4, 1, 2, 3)  # B, cl, n_out, H, W
            )

            ndvi = hr_dynamic_inputs[:, :c_l, : self.hparams.n_out, ...]

            last_pixel = torch.gather(
                ndvi, 1, (indxs * (mask < 1)).argmax(1, keepdim=True)
            )  # (B, 1, n_out, H, W)

            images_out += last_pixel.repeat(1, N_patch, 1, 1, 1)

        images_out = images_out[:, -preds_length:]

        return images_out, {}


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    arg = DynamicNet_multi.add_model_specific_args()
    args = arg.parse_args()
    model = DynamicNet_multi(args)
    model.to(device=device)
    data = {
        "dynamic": [
            torch.rand(1, 30, 5, 128, 128),
            torch.rand(1, 30, 24),
        ],
        "dynamic_mask": [torch.rand(1, 30, 1, 128, 128)],
        "static": [torch.rand(1, 3, 128, 128)],
    }
    for key, value in data.items():
        if isinstance(value, list):
            for i in range(len(value)):
                if isinstance(value[i], torch.Tensor):
                    data[key][i] = value[i].to(device)
        elif isinstance(value, torch.Tensor):
            data[key] = value.to(device)
    model.eval()
    preds, aux = model(data)
    print("of")
