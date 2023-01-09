from functools import partial

import torch
from torch import nn

from segformer import Segformer

from configs.segformer_config import config as cfg
from utils.modelsummary import get_model_summary

SEG_CFG = cfg.MODEL.B0
cfg.NUM_CLASSES = 1
print(SEG_CFG)

model = Segformer(pretrained = "/Users/pablo/Desktop/segmentation_pytorch/mit_b0.pth", img_size = 512,
                  patch_size = cfg.MODEL.PATCH_SIZE, embed_dims = SEG_CFG.CHANNEL_DIMS, num_heads = SEG_CFG.NUM_HEADS,
                  mlp_ratios = SEG_CFG.MLP_RATIOS, qkv_bias = SEG_CFG.QKV_BIAS, depths = SEG_CFG.DEPTHS,
                  sr_ratios = SEG_CFG.SR_RATIOS, drop_rate = SEG_CFG.DROP_RATE, drop_path_rate = SEG_CFG.DROP_PATH_RATE,
                  decoder_dim = SEG_CFG.DECODER_DIM, norm_layer = partial(nn.LayerNorm, eps = 1e-6),
                  shift_patch_tokenization = False)


x2 = torch.ones((2, 3, 512, 512))
details = get_model_summary(model, x2, verbose=True)
print(details)