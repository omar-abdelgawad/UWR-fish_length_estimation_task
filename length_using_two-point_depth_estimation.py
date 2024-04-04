"""This is a module that contains functions related to depth estimation."""
import numpy as np
import cv2
import torch
from math import dist
from typing import Optional
from ultralytics import YOLO
from length_estimation_combined_digital import ret_img_roi, total_length_finder
import time

WEIGHTS_PATH = r"C:\Users\OmarAbdelgawad\Desktop\deep fish\weights\weights\best.pt"
KNOWN_DEPTH = 50.0  # lets say it is in cm
FOCAL_LENGTH = 338.6  # perceived focal length has no units
REF_PIXEL = None
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
FONT_SCALE = 1
THICKNESS = 3

depth_model_type = (
    "DPT_Large"  # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
)
# depth_model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
# depth_model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
midas = torch.hub.load("intel-isl/MiDaS", depth_model_type)  # use of this line?

# Move model to GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

# Load transforms to resize and normalize the image for large or small model
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if depth_model_type == "DPT_Large" or depth_model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform


def ret_depth_img(img: np.ndarray, transform=transform) -> np.ndarray:
    """Takes as input an image and a transform for the model and returns the inverse relative depth image

    Args:
        img (np.ndarray): Input image to find its relative inverse depth image.
        transform (?): transform related to MIDAS model for transforming input image.

    Returns:
        np.ndarray: relative inverse depth image with the same dimensions as the original image but 1 channel only.
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()
    return output


# functions for calculating depth of wall
def _depth_estimation_from_rel_depth(
    ref: tuple, p1: tuple, p2: tuple, rel_depth_image: np.ndarray, ref_depth: float
) -> float:
    """Takes as input the two pixel indices. The first is the first pixel from which we will use
    the ref_depth parameter (last argument). It returns an estimated depth of the to_measure pixel.

    Args:
        ref (Tuple): index of the pixel that we know its ref_depth.
        p1 (Tuple): index of the first pixel which we want to measure its real_depth.
        p2 (Tuple): index of the second pixel which we wan to measure its real_depth.
        rel_depth_image (np.ndarray): inverse relative depth matrix of the original image.
        ref_depth (float): the ref_depth/known depth of the reference pixel.

    Returns:
        float: the estimated depth of the two given pixels.
    """
    avg_row = (p1[0] + p2[0]) // 2
    avg_col = (p1[1] + p2[1]) // 2
    rel_depth_measure: float = (
        float(
            rel_depth_image[p1[0], p1[1]]
            + rel_depth_image[p2[0], p2[1]]
            + rel_depth_image[avg_row, avg_col]
        )
        / 3
    )
    rel_ref_depth: float = (
        float(
            rel_depth_image[ref[0], ref[1]]
            + rel_depth_image[ref[0] - 1, ref[1] - 1]
            + rel_depth_image[ref[0] - 1, ref[1]]
            + rel_depth_image[ref[0] - 1, ref[1] + 1]
            + rel_depth_image[ref[0], ref[1] - 1]
            + rel_depth_image[ref[0], ref[1] + 1]
            + rel_depth_image[ref[0] + 1, ref[1] - 1]
            + rel_depth_image[ref[0] + 1, ref[1]]
            + rel_depth_image[ref[0] + 1, ref[1] + 1]
        )
        / 9
    )
    depth_ratio = ref_depth * rel_ref_depth
    estimated_depth = depth_ratio / rel_depth_measure
    # estimated_depth = ref_depth * (1 / rel_depth_measure - 1 / rel_ref_depth) #this is a wrong equation
    return estimated_depth


def _real_dist_formula_from_focal(
    pixel_dist: float, depth_of_pixel: float, focal_length: float
) -> float:
    """Takes as input the pixel distance, depth of the image and the perceived focal length to return
    the actual distance in the same measuring unit as the depth provided.

    Args:
        pixel_dist (float): distance between two wanted pixels to calculate their real world distance.
        depth_of_pixel (float): estimated depth of the two pixels that we wish to calculate their real distance in m or cm.
        focal_length (float): perceived focal length of the camera (constant for each camera).

    Returns:
        float: this number is the estimated actual distance corresponding to the given pixel distance.
    """
    return pixel_dist * depth_of_pixel / focal_length


def real_dist_from_pixels(
    p1: tuple,
    p2: tuple,
    rel_depth_image: np.ndarray,
    known_depth=None,
    reference_p=None,
    focal_length: float = 338.6,
) -> float:
    """Takes as input the two pixel indices that we want to calculate their real distance apart, the
    inverse relative depth image, known depth of a reference pixel and the index of a reference pixel and the focal length.

    Args:
        p1 (tuple): index of first pixel.
        p2 (tuple): index of second pixel.
        rel_depth_image (np.ndarray): inverse relative depth image from midas.
        known_depth (float): known depth in cm of the reference pixel.
        reference_p (tuple): index of the reference pixel with the known depth.
        focal_length (float): perceived focal_length of the camera which is the needed scaling factor.

    Returns:
        float: estimated length of the fish in cm.
    """
    if reference_p is None:
        reference_p = (rel_depth_image.shape[0] - 100, rel_depth_image.shape[1] - 100)
    if known_depth is None:
        known_depth = 25.0  # cm
    # F = (Pixel length * Depth)/actual length

    estimated_depth = _depth_estimation_from_rel_depth(
        ref=reference_p,
        p1=p1,
        p2=p2,
        rel_depth_image=rel_depth_image,
        ref_depth=known_depth,
    )
    pixel_dist = dist(p1, p2)  # from math module
    estimated_length_of_fish = _real_dist_formula_from_focal(
        pixel_dist=pixel_dist, depth_of_pixel=estimated_depth, focal_length=focal_length
    )

    return estimated_length_of_fish


def main():
    model_yolo = YOLO(WEIGHTS_PATH)
    capture = cv2.VideoCapture(0)
    while True:
        isTrue, frame = capture.read()
        if not isTrue:
            break
        #####################################################
        frame = cv2.resize(
            frame, (frame.shape[1] // 3, frame.shape[0] // 3)
        )  # this may be changed to a different size of maybe scaling instead
        rel_depth_img = ret_depth_img(frame, transform=transform)
        tic = time.perf_counter()
        rois = ret_img_roi(frame, model=model_yolo, conf=0.4, show=False)
        toc = time.perf_counter()
        print(toc - tic)
        for roi in rois:
            p1, p2, total_length_pix = total_length_finder(roi)
            estimated_real_length = real_dist_from_pixels(
                p1=p1,
                p2=p2,
                rel_depth_image=rel_depth_img,
                known_depth=KNOWN_DEPTH,
                reference_p=REF_PIXEL,
                focal_length=FOCAL_LENGTH,
            )
            center_pix = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
            cv2.putText(
                frame,
                f"{estimated_real_length:.1f}cm",
                (center_pix[1], center_pix[0]),
                cv2.FONT_HERSHEY_COMPLEX,
                FONT_SCALE,
                BLUE,
                THICKNESS,
            )
        ####################################################
        cv2.imshow("Video", frame)
        if cv2.waitKey(20) & 0xFF == ord("d"):
            break
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
