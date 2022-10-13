import cv2
import numpy as np
import torch
import random

from coco_names import COCO_INSTANCE_CATEGORY_NAMES as coco_names

COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))


def get_outputs(image, model, threshold):
    with torch.no_grad():
        # forward images
        outputs = model(image)
        # get scores
        scores = list(outputs[0]['scores'].detach().cpu().numpy())
        # index scores above threshold
        thresholded_preds_inidices = [scores.index(i) for i in scores if i > threshold]
        thresholded_preds_count = len(thresholded_preds_inidices)
        # get masks
        masks = (outputs[0]['masks'] > .5).squeeze().detach().cpu().numpy()
        masks = masks[:thresholded_preds_count]

        # get bounding boxes in (x1, y1) (x2, y2) format
        boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))] for i in outputs[0]["boxes"].detach().cpu()]
        boxes = boxes[:thresholded_preds_count]

        # get labels
        labels = [coco_names[i] for i in outputs[0]["labels"]]

        return masks, boxes, labels

# print(f"{COLORS}")
