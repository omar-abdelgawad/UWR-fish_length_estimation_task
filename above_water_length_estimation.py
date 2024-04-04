"""This script is for labeling the total lengths of fish in a single image for the digital above water task. 
It contains a gui for selecting the image to be labeled. The image is then displayed with the total 
length of each fish labeled on the image."""

import cv2
import os
import tkinter as tk
from tkinter import filedialog
from rotatingcaliper import _rotatingCaliper
from init_yolo import get_object_total_length_and_endpoints
from typing import Optional

COLOR = (0, 255, 0)
THICKNESS = 4
test_image_path = None


def open_image():
    global test_image_path
    test_image_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff *.ppm *.pgm")]
    )


def activate_gui():
    root = tk.Tk()
    root.title("Image Viewer")

    # Create a button to open the file dialog
    open_button = tk.Button(root, text="Open Image", command=open_image)
    open_button.pack()

    # Create a label to display the selected image
    image_label = tk.Label(root)
    image_label.pack()

    root.mainloop()


def transform_total_length_to_cm(total_length: float) -> float:
    """Transform total length to cm."""
    return total_length


def main(argv: Optional[list[str]]) -> int:
    # TODO: find the scaling factor for the total length
    activate_gui()
    if test_image_path is None:
        return 1
    img = cv2.imread(test_image_path)

    for total_length, p1, p2 in get_object_total_length_and_endpoints(img):
        total_length = transform_total_length_to_cm(total_length)
        middle_point = (p1 + p2) // 2
        cv2.line(img, p1, p2, color=COLOR, thickness=THICKNESS)
        cv2.putText(
            img,
            f"{total_length:.2f} cm",
            middle_point,
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            COLOR,
            3,
        )

    cv2.imshow(f"digital fish length estimation", cv2.resize(img, None, fx=0.3, fy=0.3))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    exit(main(None))
