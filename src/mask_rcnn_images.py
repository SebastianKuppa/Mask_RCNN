import torch
import torchvision
import cv2
import argparse

from PIL import Image
from utils import draw_segmentation_map, get_outputs
from torchvision.transforms import transforms as transforms

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required=True,
                    help='path to the input data')
parser.add_argument('-t', '--threshold', default=0.965, type=float,
                    help='score threshold for discarding detection')
args = vars(parser.parse_args())

