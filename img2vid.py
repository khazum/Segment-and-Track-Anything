import os
from typing import Iterable
import cv2

def images_to_video(img_dir: str, out_file: str, fps: int = 10) -> None:
    """Create a video from all images in a directory (sorted by name).

    Supports common image extensions. Raises ValueError if no images found.
    """
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Directory not found: {img_dir}")
    img_names = sorted(
        n for n in os.listdir(img_dir)
        if n.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    )
    if not img_names:
        raise ValueError(f"No images found in: {img_dir}")

    first = cv2.imread(os.path.join(img_dir, img_names[0]))
    height, width = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_file, fourcc, fps, (width, height))
    try:
        for name in img_names:
            img = cv2.imread(os.path.join(img_dir, name))
            out.write(cv2.resize(img, (width, height)))
    finally:
        out.release()

if __name__ == "__main__":
    # Example: convert a demo seq in assets
    images_to_video("./assets/840_iSXIa0hE8Ek", "./assets/840_iSXIa0hE8Ek.mp4", fps=10)