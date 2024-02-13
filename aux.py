import argparse
import cv2
import numpy as np
import os


def parse_args():
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description="YOLO Inference Pipeline Arguments")
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., coco)",
    )
    parser.add_argument(
        "-n",
        "--network",
        type=str,
        required=True,
        help="Network checkpoint name",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=1,
        help="Number of threads to use for inference",
    )
    parser.add_argument(
        "--batch", type=int, default=1, help="Batch size for batch inference"
    )
    parser.add_argument(
        "--imsize", type=int, default=640, help="Size of the images for inference"
    )
    parser.add_argument(
        "--maxim", type=int, default=1000, help="Maximum number of images to process"
    )
    args = parser.parse_args()
    return args


def to_img(outimg, i, img, xyxy):
    """saves annotated image to file"""
    img = np.array(img)
    for j in range(xyxy.shape[0]):
        cv2.rectangle(
            img,
            (int(xyxy[j][0]), int(xyxy[j][1])),
            (int(xyxy[j][2]), int(xyxy[j][3])),
            (0, 255, 0),
            2,
        )

    cv2.imwrite(os.path.join(outimg, f"{i}.jpeg"), img)
