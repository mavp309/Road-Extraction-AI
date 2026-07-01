"""
mask_processing.py

Handles:

1. Loading binary road mask
2. Skeletonization
3. Pixel neighbour utilities

Everything downstream works on the skeleton generated here.
"""

from pathlib import Path

import cv2
import numpy as np


DEFAULT_MASK = Path(__file__).resolve().parent.parent / "frontend" / "saved_mask.png"


# ----------------------------------------------------------
# Loading
# ----------------------------------------------------------

def load_mask(mask_path=DEFAULT_MASK):

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise FileNotFoundError(mask_path)

    return (mask > 127).astype(np.uint8)


# ----------------------------------------------------------
# Skeletonization
# ----------------------------------------------------------

def skeletonize(mask):
    """
    Thin a binary road mask to a strict 1-pixel-wide skeleton.

    NOTE: the previous implementation here (iterative erode/dilate/subtract)
    does not guarantee 1-pixel connectivity -- it commonly leaves 2-pixel-wide
    stretches of skeleton behind. Under 8-connectivity, nearly every pixel in
    a 2-pixel-wide stretch has degree >= 3, so detect_keypoints() below
    misclassifies huge swaths of ordinary road pixels as "junctions". That is
    why a simple road layout was producing thousands of graph nodes -- you
    were effectively getting one node per skeleton pixel, not one per real
    intersection.

    skimage's skeletonize() uses proper topological thinning (Zhang-Suen) and
    produces a true 1-pixel-wide skeleton, so degree >= 3 pixels only show up
    where roads genuinely branch. Requires: pip install scikit-image
    """
    from skimage.morphology import skeletonize as _sk_skeletonize

    bool_mask = mask > 0
    skeleton = _sk_skeletonize(bool_mask)

    return skeleton.astype(np.uint8)


# ----------------------------------------------------------
# Pixel Utilities
# ----------------------------------------------------------

OFFSETS_8 = [

    (-1, -1),
    (-1, 0),
    (-1, 1),

    (0, -1),
    (0, 1),

    (1, -1),
    (1, 0),
    (1, 1),

]


def neighbours(skeleton, y, x):

    h, w = skeleton.shape

    pts = []

    for dy, dx in OFFSETS_8:

        ny = y + dy
        nx = x + dx

        if ny < 0 or ny >= h:
            continue

        if nx < 0 or nx >= w:
            continue

        if skeleton[ny, nx]:

            pts.append((ny, nx))

    return pts


def degree(skeleton, y, x):

    return len(neighbours(skeleton, y, x))


# ----------------------------------------------------------
# Node Classification
# ----------------------------------------------------------

def is_dead_end(skeleton, y, x):

    return degree(skeleton, y, x) == 1


def is_regular(skeleton, y, x):

    return degree(skeleton, y, x) == 2


def is_intersection(skeleton, y, x):

    return degree(skeleton, y, x) >= 3


# ----------------------------------------------------------
# Collect Special Pixels
# ----------------------------------------------------------

def detect_keypoints(skeleton):

    intersections = []

    dead_ends = []

    h, w = skeleton.shape

    for y in range(h):

        for x in range(w):

            if skeleton[y, x] == 0:
                continue

            d = degree(skeleton, y, x)

            if d == 1:
                dead_ends.append((y, x))

            elif d >= 3:
                intersections.append((y, x))

    return {

        "intersections": intersections,

        "dead_ends": dead_ends,

        "junctions": intersections + dead_ends,

    }


# ----------------------------------------------------------
# Complete Pipeline
# ----------------------------------------------------------

def load_and_prepare(mask_path=DEFAULT_MASK):

    mask = load_mask(mask_path)

    skeleton = skeletonize(mask)

    keypoints = detect_keypoints(skeleton)

    return {

        "mask": mask,

        "skeleton": skeleton,

        "keypoints": keypoints,

    }