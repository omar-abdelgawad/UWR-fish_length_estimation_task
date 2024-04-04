"""This module contains old/deprecated functions for detecting fish in images using YOLOv8 detection model(bboxes).
These functions became obsolete after the development of the segmentation model."""
import numpy as np
from ultralytics import YOLO
import cv2


# these are functions that returns bboxes as rois
def _ret_bbox(
    img: np.ndarray,
    model: YOLO,
    conf: float = 0.5,
    show: bool = False,
) -> list[list[int]]:
    """Takes an input image and returns a list of bboxes in the format of xyxy

    Args:
        img (np.ndarray): Input image to detect fish
        model (YOLO): Yolo model instance
        conf (float, optional): Minimum confidence score for detection. Defaults to 0.5.
        show (bool, optional): Whether to show image after detection. Defaults to False.

    Returns:
        list[list[int]]: list of all bboxes in the provided image. Each bbox is a list in the form xyxy.
    """
    result = model(img, conf=conf, show=show)[0]  # takes 3.9 seconds for loading up.
    if model.device is None or model.device.type != "cuda":
        print(f"Error:Can't find gpu/cuda. Used {model.device} instead.")
    else:
        result = result.cpu()
    bboxes = result.boxes.xyxy.int().tolist()
    return bboxes


def ret_img_roi(
    img: np.ndarray,
    model: YOLO,
    conf: float = 0.5,
    show: bool = False,
) -> list[np.ndarray]:
    """Takes an input image and returns regions of interest as a list of rois all of them as ndarrays.

    Args:
        img (np.ndarray): Input image to detect fish
        model (YOLO): Yolo model instance
        conf (float, optional): Minimum confidence score for detection. Defaults to 0.5.
        show (bool, optional): Whether to show image after detection. Defaults to False.

    Returns
        list[np.ndarray]: list of all rois in the image
    """
    bboxes = _ret_bbox(img, model, conf=conf, show=show)
    rois = []
    for box in bboxes:
        x1, y1, x2, y2 = box
        roi = img[y1:y2, x1:x2]
        rois.append(roi)
    return rois


def _otsu_mask_generator(img: np.ndarray) -> np.ndarray:
    """
    Generates a binary mask using Otsu's thresholding method.

    Args:
        img (np.ndarray): Input image.

    Returns:
        np.ndarray: Binary mask generated using Otsu's thresholding method.
    """
    img = cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT)
    img = cv2.bilateralFilter(img, 9, 75, 75)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]


def _totalLength(
    img: np.ndarray,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
    """Takes an input image of a binary segmented fish with only the edges and
    calculates the length of the fish. Works in O(nlogn).

    Args:
        img (np.ndarray): Input image to find max dist of fish.

    Returns:
        tuple[tuple, tuple, float]: first two elements are tuples of the index of
        the two pixels which produces the max distance.
    """

    coordinates_of_nonzero_points_in_img = np.transpose(np.nonzero(img))
    # This error handling should be done more elegantly. Will do so after finishing the rest.
    if len(coordinates_of_nonzero_points_in_img) < 2:
        raise ValueError
    max_distance, pair = _rotatingCaliper(coordinates_of_nonzero_points_in_img)
    p1_max = pair["p1"]
    p2_max = pair["p2"]
    return p1_max, p2_max, max_distance


def total_length_finder(
    roi: np.ndarray,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
    """Processes input image of a single fish it and returns a tuple of the two pixel
    indices that achieve the maximum distance.

    Args:
        roi (np.ndarray): Input image to find max dist of fish.

    Returns:
        tuple[Optional[np.ndarray], Optional[np.ndarray], float]: first two elements are ndarrays
        of the index of the two pixels which produces the max distance.
    """
    # TODO: maybe use a segmentation model to avoid this mess
    # TODO: maybe use a contour detector after the cannny edge detector and get highest area contour.
    otsu_mask = _otsu_mask_generator(roi)
    roi_contour = cv2.Canny(otsu_mask, 190, 200)
    kernel = np.ones((3, 3), np.uint8)
    # TODO: find out if this morphological operation is necessary? it is currently removed
    roi_contour_denoised = cv2.morphologyEx(
        roi_contour, cv2.MORPH_OPEN, kernel, iterations=2
    )
    p1, p2, total_length = _totalLength(
        roi_contour
    )  # why use roi_countour instead of the denoised version???????
    return p1, p2, total_length
