import torch
import numpy as np
from PIL import Image


def convert_image_to_tensor(img):
    img = img.convert("RGB")
    img_np = np.array(img).astype(np.float32)
    img_np = np.transpose(img_np, axes=[2, 0, 1])[np.newaxis, :, :, :]
    img_pt = torch.from_numpy(img_np)
    return img_pt


def convert_tensor_to_image(img_pt):
    img_pt = img_pt[0, ...].permute(1, 2, 0)
    result_rgb_np = img_pt.cpu().numpy().astype(np.uint8)
    return Image.fromarray(result_rgb_np)


def convert_image_to_tensor2(img):
    # RGBA形式を確保して、透過チャンネルも適切に扱います。
    img_np = np.array(img).astype(np.float32)
    img_np = np.transpose(img_np, axes=[2, 0, 1])[np.newaxis, :, :, :]
    img_pt = torch.from_numpy(img_np)
    return img_pt

def convert_tensor_to_image2(img_pt, alpha_channel):
    # img_ptのデータ型を確認
    print(f"Type of img_pt: {type(img_pt)}")
    if isinstance(img_pt, torch.Tensor):
        img_pt = img_pt[0, ...].permute(1, 2, 0)  # 最初のバッチを選択し、HWCフォーマットに変換
        result_rgb_np = img_pt.cpu().numpy().astype(np.uint8)
        # alpha_channelがテンソルであることを確認し、必要な形状に変換
        if isinstance(alpha_channel, torch.Tensor):
            alpha_channel = alpha_channel.cpu().numpy().astype(np.uint8)
            if alpha_channel.ndim == 3:  # [1, H, W] または [H, W, 1]の可能性がある
                alpha_channel = np.expand_dims(alpha_channel, axis=-1) if alpha_channel.shape[2] != 1 else alpha_channel
            # RGBとアルファチャンネルを結合
            result_rgba_np = np.concatenate((result_rgb_np, alpha_channel), axis=2)
            return Image.fromarray(result_rgba_np)
        else:
            raise TypeError("alpha_channel must be a torch.Tensor")
    else:
        raise TypeError("img_pt must be a torch.Tensor")
