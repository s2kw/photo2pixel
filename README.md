# Photo2PixelWithAlpha

---
日本語 | [English](./README_en.md) | [简体中文](./README_cn.md)

photo2pixelWithAlpha オリジナルのphoto2pixelにアルファ付きの画像を加工可能に改変したものです。
ピクセルアートっぽくしてくれるツールです。オリジナルのphoto2pixel制作者に敬意を表します。

[Online Tool](https://photo2pixel.co) |
[Colab](https://colab.research.google.com/drive/108np4teybhBXHKbPMZZ1fykDuUeF2aw8?usp=sharing) |
[Tutorial](#Tutorial)

## Prerequisites
- python3
- pytorch (for algorithm implementation)
- pillow (for image file io)
- opencv-python

## Tutorial
---
コマンドラインで動作します。下記のとおりです。

```bash
# venv上で作業しましょう
python -m venv .venv
source .venv/bin/activate

# pipを最新にする 
pip install --upgrade pip

# venv上の環境を整える
pip install -r requirements.txt

# main.pyを走らせる
python main.py

# images/origin_image 以下に配置されたpng画像全部を指定のパラメータに変換する
# この時、アルファ値があるとそれも含めて全部変換してくれる

# or use custom param
#python convert.py --kernel_size 12 --pixel_size 12 --edge_thresh 128
```

| Parameter   |                                Description                                |    Range    |               Default               |
|-------------|:-------------------------------------------------------------------------:|:-----------:|:-----------------------------------:|
| input       |                             input image path                              |      /      | ./images/example_input_mountain.jpg |
| output      |                             output image path                             |      /      |            ./result.png             |
| kernel_size |             larger kernel size means smooth color transition              |  unlimited  |                 10                  |
| pixel_size  |                           individual pixel size                           |  unlimited  |                 16                  |
| edge_thresh | the black line in edge region, lower edge threshold means more black line |    0~255    |                 100                 |