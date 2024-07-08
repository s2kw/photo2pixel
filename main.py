import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2  # OpenCVをインポート
import argparse
import os

from models.module_photo2pixel2 import Photo2PixelModel
from utils import img_common_util

def convert(imagePath):
    parser = argparse.ArgumentParser(description='algorithm converting photo to pixel art')
    parser.add_argument('--input', type=str, default="./" + imagePath, help='input image path')
    parser.add_argument('--output', type=str, default="./result/" + imagePath + "_result.png", help='output image path')
    parser.add_argument('-k', '--kernel_size', type=int, default=10, help='larger kernel size means smooth color transition')
    parser.add_argument('-p', '--pixel_size', type=int, default=3, help='individual pixel size')
    parser.add_argument('-e', '--edge_thresh', type=int, default=2048, help='lower edge threshold means more black line in edge region')
    parser.add_argument('--resize', type=int, nargs=2, help='resize images to a fixed size (width height)')
    args = parser.parse_args()

    img_input = Image.open(args.input).convert("RGBA")
    
    if args.resize:
        img_input = img_input.resize(args.resize, Image.ANTIALIAS)
    
    img_np = np.array(img_input).astype(np.float32)
    img_rgb = img_np[:, :, :3]
    alpha_channel = img_np[:, :, 3]

    # サイズが異なる場合はOpenCVを使用してリサイズ
    if img_rgb.shape[:2] != alpha_channel.shape[:2]:
        alpha_channel = cv2.resize(alpha_channel, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

    img_pt = np.transpose(img_rgb, axes=[2, 0, 1])[np.newaxis, :, :, :]  # RGBのみをテンソルに変換
    img_pt = torch.from_numpy(img_pt)
    alpha_channel = torch.from_numpy(alpha_channel).unsqueeze(0).unsqueeze(0)  # アルファチャンネルの次元を調整

    print("Original RGB shape:", img_pt.shape)
    print("Original Alpha shape:", alpha_channel.shape)

    model = Photo2PixelModel()
    model.eval()
    with torch.no_grad():
        result_rgb_pt, result_alpha_pt = model(
            img_pt,
            alpha_channel,
            param_kernel_size=args.kernel_size,
            param_pixel_size=args.pixel_size,
            param_edge_thresh=args.edge_thresh
        )

    result_rgb_np = result_rgb_pt[0, ...].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    result_alpha_np = result_alpha_pt[0, 0, :, :].cpu().numpy().astype(np.uint8)

    print("Processed RGB shape:", result_rgb_np.shape)
    print("Processed Alpha shape:", result_alpha_np.shape)

    # 処理後のサイズが異なる場合は再度OpenCVを使用してリサイズ
    if result_rgb_np.shape[:2] != result_alpha_np.shape[:2]:
        result_alpha_np = cv2.resize(result_alpha_np, (result_rgb_np.shape[1], result_rgb_np.shape[0]), interpolation=cv2.INTER_NEAREST)

    # RGBとアルファチャンネルを結合して画像を保存
    result_rgba_np = np.concatenate((result_rgb_np, result_alpha_np[:, :, np.newaxis]), axis=2)

    output_dir = os.path.dirname(args.output)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    Image.fromarray(result_rgba_np).save(args.output)

if __name__ == '__main__':
    def find_png_images(directory):
        png_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".png"):
                    png_files.append(os.path.join(root, file))
        return png_files

    directory = 'images/400/'

    png_images = find_png_images(directory)
    for image in png_images:
        print(image)
        convert(image)
