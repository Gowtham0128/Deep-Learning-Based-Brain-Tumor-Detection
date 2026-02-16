"""process_brain_image.py

Simple CLI to annotate a brain image with a tumour mask.

Usage:
    python process_brain_image.py --image static/data/sample1.jpg
    python process_brain_image.py --image static/data/sample1.jpg --mask path/to/mask.png

If --mask is omitted, a simple heuristic mask will be generated (demo-only).
"""
import argparse
import os
from DATA import overlay_mask_on_image


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--image', required=True, help='Path to input image')
    p.add_argument('--mask', required=False, help='Optional path to binary mask')
    p.add_argument('--out', required=False, help='Optional output path')
    args = p.parse_args()

    image = args.image
    mask = args.mask
    out = args.out

    if not os.path.exists(image):
        print('Image not found:', image)
        return

    path = overlay_mask_on_image(image_path=image, mask_path=mask, out_path=out)
    print('Annotated image saved to', path)


if __name__ == '__main__':
    main()
