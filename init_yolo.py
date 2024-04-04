"""module for initializing YOLOv8 segmentation model. Used for fish length estimation for both above and below water task.
it can segment pagrus, and tray classes."""
from ultralytics import YOLO
from rotatingcaliper import _rotatingCaliper
import numpy as np
import os

# path of detection/segmentation model to use for prediction.
dir_path = os.path.dirname(__file__)
WEIGHTS_PATH = os.path.join(dir_path, "segmentation_model", "last.pt")

model = YOLO(WEIGHTS_PATH)
# use model to load it on gpu
model(np.zeros((100, 100, 3)), show=False, save=False)


def get_object_polygons_as_int(img: np.ndarray):
    results = model(img)[0]
    masks = results.masks
    if masks is None:
        return []
    for polygon_points in masks.cpu().xy:
        polygon_points: np.ndarray = polygon_points.astype(int)
        yield polygon_points


def get_object_total_length_and_endpoints(
    img: np.ndarray,
):
    for polygon_points in get_object_polygons_as_int(img):
        total_length, pt_dict = _rotatingCaliper(polygon_points)
        p1 = pt_dict["p1"]
        p2 = pt_dict["p2"]
        if p1 is None or p2 is None:
            continue
        yield total_length, p1, p2
