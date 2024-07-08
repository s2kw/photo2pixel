import torch
from PIL import Image
import argparse
import numpy as np

from models.module_photo2pixel2 import Photo2PixelModel
from utils import img_common_util


def convert( k, p, e ):
    parser = argparse.ArgumentParser(description='algorithm converting photo to pixel art')
#     parser.add_argument('--input', type=str, default="./images/example_input_mountain.jpg", help='input image path')
    parser.add_argument('--input', type=str, default="./images/example_input.png", help='input image path')
    parser.add_argument('--output', type=str, default=f"./result_k{k}_p{p}_e{e}_____.png", help='output image path')
    parser.add_argument('-k', '--kernel_size', type=int, default=k, help='larger kernel size means smooth color transition')
    parser.add_argument('-p', '--pixel_size', type=int, default=p, help='individual pixel size')
    parser.add_argument('-e', '--edge_thresh', type=int, default=e, help='lower edge threshold means more black line in edge region')
    args = parser.parse_args()


    img_input = Image.open(args.input)
    img_pt_input = img_common_util.convert_image_to_tensor2(img_input)

    img_np = np.array(img_input).astype(np.float32)
    img_pt = np.transpose(img_np[:, :, :3], axes=[2, 0, 1])[np.newaxis, :, :, :]  # RGBのみをテンソルに変換
#     alpha_channel = img_np[:, :, 3:]  # アルファチャンネルを保持
    alpha_channel = torch.from_numpy(img_np[:, :, 3:]).unsqueeze(0).float()  # アルファチャンネルをTensorに変換


    model = Photo2PixelModel()
    model.eval()
    with torch.no_grad():
        result_rgb_pt, result_alpha_pt = model(
            img_pt_input,
            alpha_channel,
            param_kernel_size=args.kernel_size,
            param_pixel_size=args.pixel_size,
            param_edge_thresh=args.edge_thresh
        )

    result_rgb_np = result_rgb_pt.cpu().numpy().astype(np.uint8)
    result_alpha_np = result_alpha_pt.cpu().numpy().astype(np.uint8)

    # 形状を印刷して確認
    print("RGB shape:", result_rgb_np.shape)
    print("Alpha shape:", result_alpha_np.shape)

    # アルファチャンネルの次元を確認し、必要に応じて形状を調整
    if result_alpha_np.ndim == 2:
        result_alpha_np = result_alpha_np[:, :, np.newaxis]  # 新しい次元を追加
    
    # RGBとアルファチャンネルを結合して画像を保存
    result_rgba_np = np.concatenate((result_rgb_np, result_alpha_np), axis=2)
    Image.fromarray(result_rgba_np).save(args.output)


# for k in range(1, 31):  # kを1から30までループ
#     for p in range(1, 31):  # pを1から30までループ
#         # ここに何かの処理を書く
#         print(f"k={k}, p={p}")
#         convert( k, p, 1000 )

if __name__ == '__main__':
    convert(16,8,1000)



