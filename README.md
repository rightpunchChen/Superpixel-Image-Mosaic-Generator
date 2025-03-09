# Superpixel-Image-Mosaic-Generator
This project uses OpenCV to generate image mosaics based on superpixel segmentation. The target image is divided into superpixel regions, and the most similar image tiles from an image pool are selected to fill these regions, creating a mosaic-like effect.

## Requirements
Ensure you have the following Python libraries installed:
```bash
pip install opencv-python opencv-contrib-python numpy tqdm
```

## Usage

1. **Parameter Description**

|Parameter|Description|Default|
|:---:|:---:|:---:|
|`--npz_dir`|Path to store processed image pool (in `.npz` format)|`./img.npz`|
|`--resize_factor`|	Image resize factor (reduces image size for faster processing)|	`2`|
|`--image_pool`|Path to the image pool directory|`./image_pool`|
|`--target_root`|Path to the target image|`./test.jpg`|
|`--output_dir`|Path to save the output image|`./output`|
|`--region_size`|Superpixel region size|`80`|
|`--ruler`|Smoothingfactor for superpixel algorithm|`150`|

2. **Run the Program**

Run the following command to generate the superpixel image mosaic:
```bash
python main.py --target_root ./test.jpg --image_pool ./image_pool --output_dir ./output
```

## Notes

1. Ensure the `image_pool` contains a diverse set of images for better mosaic effects.
2. A higher `resize_factor` speeds up processing but may reduce output quality.
3. Adjust `region_size` and `ruler` to control the granularity and smoothness of superpixels.

<img src="https://github.com/rightpunchChen/3d-mri-volume-visualizer-v2/blob/main/demo.png">
