import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from models.module_edge_detector import EdgeDetectorModule
from models.module_pixel_effect import PixelEffectModule


class Photo2PixelModel(nn.Module):
    def __init__(self):
        super(Photo2PixelModel, self).__init__()
        self.module_pixel_effect = PixelEffectModule()
        self.module_edge_detect = EdgeDetectorModule()

    def forward(self, rgb, alpha, param_kernel_size=10, param_pixel_size=16, param_edge_thresh=112):
        """
        アルファチャンネルを扱うための変更を加えます。
        """
        rgb = self.module_pixel_effect(rgb, 4, param_kernel_size, param_pixel_size)

        edge_mask = self.module_edge_detect(rgb, param_edge_thresh, param_edge_dilate=3)
        rgb = torch.masked_fill(rgb, torch.gt(edge_mask, 0.5), 0)

        # アルファチャンネルは影響を受けないため、そのまま返します。
        return rgb, alpha

def test1():
    img = Image.open("../images/example_input_mountain.png").convert("RGBA")
    img_np = np.array(img).astype(np.float32)
    img_pt = np.transpose(img_np[:, :, :3], axes=[2, 0, 1])[np.newaxis, :, :, :]  # RGBのみをテンソルに変換
    alpha_channel = img_np[:, :, 3:]  # アルファチャンネルを保持

    img_pt = torch.from_numpy(img_pt)
    alpha_channel = torch.from_numpy(alpha_channel).unsqueeze(0)  # アルファチャンネルの次元を調整

    model = Photo2PixelModel()
    model.eval()

    with torch.no_grad():
        result_rgb_pt, result_alpha_pt = model(img_pt, alpha_channel, param_kernel_size=11, param_pixel_size=16)
        result_rgb_pt = result_rgb_pt[0, ...].permute(1, 2, 0)
        result_alpha_pt = result_alpha_pt[0, ...].permute(1, 2, 0)

    result_rgb_np = result_rgb_pt.cpu().numpy().astype(np.uint8)
    result_alpha_np = result_alpha_pt.cpu().numpy().astype(np.uint8)

    # RGBとアルファチャンネルを結合して画像を保存
    result_rgba_np = np.concatenate((result_rgb_np, result_alpha_np), axis=2)
    Image.fromarray(result_rgba_np).save("./test_result_photo2pixel.png")

if __name__ == '__main__':
    test1()
