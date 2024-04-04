"""This module contains the definition of the _rotatingCaliper function. given n points in the plane, it
first computes the convex hull of the points, then computes the maximum distance between any two points. Total 
runtime is O(nlogn) from the convex hull computation."""
import numpy as np
import cv2
from math import dist
from typing import Optional


def _absArea(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
    """Calculates tha area of the triangle constructed from 3 input points.

    Args:
        p (np.ndarray): vector of two elements.
        q (np.ndarray): vector of two elements.
        r (np.ndarray): vector of two elements.

    Returns:
        float: area of the triangle.
    """
    return abs(
        (p[0] * q[1] + q[0] * r[1] + r[0] * p[1])
        - (p[1] * q[0] + q[1] * r[0] + r[1] * p[0])
    )


def _rotatingCaliper(
    points: np.ndarray,
) -> tuple[float, dict[str, Optional[np.ndarray]]]:
    """Takes as input a two dimensional ndarray where each row is point with a x and a y
    cooradinate and calculates the maximum distance between any two points in the set.
    Complexity is O(nlogn) due to sorting and it returns a tuple of The max_distance
    and a dict which contains the two points that are farthest apart.

    Args:
        points (np.ndarray): set of points to find their max distance.

    Returns:
        tuple[float, dict[str, Optional[np.ndarray]]]: a tuple that contains the maximum distance and
        a dict containing the two points that achieve it.
    """
    hull: np.ndarray = np.squeeze(cv2.convexHull(points))
    n = len(hull)
    pair: dict[str, Optional[np.ndarray]] = {"p1": None, "p2": None}

    # Base Cases
    if n == 1:
        raise ValueError
    if n == 2:
        pair["p1"] = hull[0]
        pair["p2"] = hull[1]
        return dist(hull[0], hull[1]), pair
    k = 1

    # Find the farthest vertex
    # from hull[0] and hull[n-1]
    while _absArea(hull[n - 1], hull[0], hull[(k + 1) % n]) > _absArea(
        hull[n - 1], hull[0], hull[k]
    ):
        k += 1

    res = 0.0
    # Check points from 0 to k
    for i in range(k + 1):
        j = (i + 1) % n
        while _absArea(hull[i], hull[(i + 1) % n], hull[(j + 1) % n]) > _absArea(
            hull[i], hull[(i + 1) % n], hull[j]
        ):
            # Update res
            new_dis = dist(hull[i], hull[(j + 1) % n])
            if new_dis > res:
                res = new_dis
                pair["p1"] = hull[i]
                pair["p2"] = hull[(j + 1) % n]
            j = (j + 1) % n
        new_dis = dist(hull[i], hull[j])
        if new_dis > res:
            res = new_dis
            pair["p1"] = hull[i]
            pair["p2"] = hull[j]

    # Return the result distance
    return res, pair
