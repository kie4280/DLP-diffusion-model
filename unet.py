from typing import Optional, Tuple, Union
from diffusers import UNet2DModel
import torch
from diffusers.models.unet_2d import UNet2DOutput
from torch import FloatTensor, Tensor

IMG_SIZE = (240, 320)


class UNet(UNet2DModel):
    def __init__(
        self,
        sample_size: Optional[Union[int, Tuple[int, int]]] = None,
        in_channels: int = 3,
        out_channels: int = 3,
        center_input_sample: bool = False,
        time_embedding_type: str = "positional",
        freq_shift: int = 0,
        flip_sin_to_cos: bool = True,
        down_block_types: Tuple[str] = (
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
        ),
        up_block_types: Tuple[str] = (
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
        ),
        block_out_channels: Tuple[int] = (224, 448, 672, 896),
        layers_per_block: int = 2,
        mid_block_scale_factor: float = 1,
        downsample_padding: int = 1,
        downsample_type: str = "conv",
        upsample_type: str = "conv",
        act_fn: str = "silu",
        attention_head_dim: Optional[int] = 8,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-5,
        resnet_time_scale_shift: str = "default",
        add_attention: bool = True,
        class_embed_type: Optional[str] = None,
        num_class_embeds: Optional[int] = None,
    ):
        super().__init__(
            sample_size,
            in_channels,
            out_channels,
            center_input_sample,
            time_embedding_type,
            freq_shift,
            flip_sin_to_cos,
            down_block_types,
            up_block_types,
            block_out_channels,
            layers_per_block,
            mid_block_scale_factor,
            downsample_padding,
            downsample_type,
            upsample_type,
            act_fn,
            attention_head_dim,
            norm_num_groups,
            norm_eps,
            resnet_time_scale_shift,
            add_attention,
            class_embed_type,
            num_class_embeds,
        )
        self.class_transform = torch.nn.Linear(24, 1)

    def forward(
        self,
        sample: FloatTensor,
        timestep: Tensor | float | int,
        class_labels: Tensor | None = None,
        return_dict: bool = True,
    ) -> UNet2DOutput | Tuple:
        class_labels = torch.flatten(self.class_transform(class_labels))
        x = super().forward(sample, timestep, class_labels, return_dict)
        return x
