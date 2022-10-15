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

# MASK RCNN model
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, progress=True,
                                                           num_classes=91)
# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load model and set to eval mode
model.to(device).eval()

# convert image to tensor
transform = transforms.Compose([
    transforms.ToTensor()
])

image_path = args['input']
image = Image.open(image_path).convert('RGB')
# keep original image for opencv operations
orig_image = image.copy()

image = transform(image)
# add batch dimension
image = image.unsqueeze(0).to(device)

masks, boxes, labels = get_outputs(image, model, args['threshold'])

result = draw_segmentation_map(image, masks, boxes, labels)

# visualize image
cv2.imshow('Segmented image', result)
cv2.waitKey(0)

# set savePath
save_path = f"../outputs/{args['--input'].split('/')[-1].split('.')[0]}.jpg"
cv2.imwrite(save_path, result)
