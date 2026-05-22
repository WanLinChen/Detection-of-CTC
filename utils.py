# -*- coding: utf-8 -*-
"""
CTC Detection Pipeline - Utility Functions
===========================================
This module consolidates all helper functions used by the main CTC
detection pipeline (main.py). It is organized into three sections:

    Section 1 — Slide Masking
        get_mask(): Detects the circular blood sample boundary using
        Hough Circle Transform and applies it to the EpCAM image,
        restricting all downstream analysis to the actual sample region.

    Section 2 — Adaptive Thresholding & SSIM
        compute_threshold(): Computes a robust per-image intensity
        threshold using the IQR method (Q50 + 4 * IQR).

        compare_structure(): Computes a custom Structural Similarity
        Index (SSIM) between two image patches — used to measure how
        circular a candidate CTC contour is.

    Section 3 — Contrast Utilities (experimental, not used in main pipeline)
        modify_contrast_and_brightness(): Standard alpha/beta contrast
        enhancement.
        modify_contrast_and_brightness2(): Contrast reduction using
        tangent-based scaling.

Usage in main.py:
    from utils import get_mask, compute_threshold, compare_structure

Author: Wan-Lin (Christine) Chen
Institution: Data Analysis and Interpretation Lab, NTHU
"""

import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from warnings import warn
from scipy.ndimage import uniform_filter, gaussian_filter
from skimage.util.dtype import dtype_range


# ══════════════════════════════════════════════════════════════════════
# SECTION 1: SLIDE MASKING
# ══════════════════════════════════════════════════════════════════════

def get_mask(hct, ep) -> np.ndarray:
    """
    Detect the circular slide boundary and apply it as a mask to the
    EpCAM image, restricting analysis to the actual blood sample region.

    The blood sample is placed on a circular microfluidic chip (SACA chip).
    Without masking, edge artifacts and non-blood regions would generate
    false positive CTC candidates. This function:
        1. Converts the Hoechst image to grayscale
        2. Applies Canny edge detection to find slide boundary edges
        3. Uses Hough Circle Transform to fit a circle to those edges
        4. Generates a binary circular mask of the slide region
        5. Multiplies the mask with the EpCAM grayscale image

    Args:
        hct (np.ndarray): Hoechst channel image (365nm, BGR format).
                          Used to detect the slide boundary because the
                          Hoechst stain clearly delineates the sample edge.
        ep  (np.ndarray): EpCAM channel image (488nm, BGR format).
                          This is the primary tumor marker image to be masked.

    Returns:
        np.ndarray: Grayscale EpCAM image with all pixels outside the
                    circular slide boundary set to zero.
    """
    mask_img = hct  # Hoechst image used for boundary detection
    img = ep        # EpCAM image to be masked

    # Convert both images to grayscale for processing
    mask_img_gray = cv2.cvtColor(mask_img, cv2.COLOR_RGB2GRAY)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # ── Step 1: Edge detection on Hoechst image ──
    # Canny edge detector finds the sharp boundary of the circular slide
    low_threshold = 15
    high_threshold = 30
    edges = cv2.Canny(mask_img_gray, low_threshold, high_threshold)

    # ── Step 2: Hough Circle Transform to fit a circle to the slide boundary ──
    # Parameters tuned for the SACA chip slide size (~3600px radius at 9081x9081 resolution)
    # minRadius/maxRadius constrain the search to plausible slide sizes
    circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=1000,           # Inverse ratio of accumulator resolution to image resolution
        minDist=100,       # Minimum distance between circle centers
        param1=100,        # Upper threshold for Canny edge detector inside HoughCircles
        param2=30,         # Accumulator threshold — lower = more circles detected
        minRadius=1000,    # Minimum slide radius in pixels
        maxRadius=2000     # Maximum slide radius in pixels
    )
    circle = circles[0, :, :]  # Extract detected circle parameters (x, y, radius)

    # ── Step 3: Generate binary circular mask at fixed slide position ──
    # Note: slide center is fixed at (4500, 4500) with radius 3600
    # based on the known chip geometry at the imaging resolution used
    shape = (9081, 9081, 3)
    origin = np.zeros(shape, np.uint8)

    # Draw filled white circle on black background to create the mask
    origin_circle = cv2.circle(origin, (4500, 4500), 3600, (255, 255, 255), -1)
    origin_circle = cv2.cvtColor(origin_circle, cv2.COLOR_RGB2GRAY)

    # Convert to binary (0 = outside slide, 1 = inside slide)
    origin_circle[origin_circle > 0] = 1

    # ── Step 4: Apply mask to EpCAM image ──
    # Pixels outside the slide boundary become 0 (black)
    # Pixels inside the slide retain their original EpCAM intensity values
    masked = origin_circle * img_gray

    return masked


# ══════════════════════════════════════════════════════════════════════
# SECTION 2: ADAPTIVE THRESHOLDING & STRUCTURAL SIMILARITY
# ══════════════════════════════════════════════════════════════════════

def compute_threshold(masked: np.ndarray) -> float:
    """
    Compute a robust adaptive intensity threshold for CTC candidate detection.

    Uses an IQR-based formula to adapt to per-image fluorescence intensity
    variation without requiring manual calibration:

        threshold = Q50 + 4 × (Q75 − Q50)

    Where Q50 is the median and Q75 is the 75th percentile of pixel
    intensities. The multiplier of 4 is empirically tuned to keep the
    threshold well above background fluorescence while capturing true
    CTC signals.

    This is more robust than a fixed threshold because fluorescence
    intensity can vary significantly between slides, staining batches,
    and imaging sessions.

    Args:
        masked (np.ndarray): Masked grayscale EpCAM image (output of get_mask).

    Returns:
        float: Intensity threshold above which pixels are considered
               candidate CTC signal.
    """
    image_stack = np.array(masked)
    image_stack = np.expand_dims(image_stack, 2)  # Add channel dimension for quantile computation

    Q50 = np.quantile(image_stack, 0.50)   # Median intensity
    Q75 = np.quantile(image_stack, 0.75)   # 75th percentile intensity
    IQR = Q75 - Q50                        # Interquartile range

    return Q50 + 4 * IQR


def _covariance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the covariance between two image arrays.

    This is a helper function used internally by compare_structure()
    to compute the cross-covariance term in the SSIM formula.

    Args:
        x (np.ndarray): First image array (flattened or 2D).
        y (np.ndarray): Second image array (same shape as x).

    Returns:
        float: Covariance value between x and y.
    """
    xbar = x.mean()
    ybar = y.mean()
    return np.sum((x - xbar) * (y - ybar)) / (len(x) * len(x) - 1)


def compare_structure(img1: np.ndarray, img2: np.ndarray,
                      data_range: float = None) -> float:
    """
    Compute a custom Structural Similarity Index (SSIM) between two patches.

    In the CTC pipeline, this is used to measure how closely a candidate
    cell contour resembles a perfect circle. A true CTC should be roughly
    circular due to its biological morphology, so high SSIM against an
    ideal circle is a positive indicator.

    This implementation computes only the structural component (S) of the
    full SSIM formula — the ratio of cross-covariance to the product of
    standard deviations, with stability constants:

        S = (cov(img1, img2) + C3) / (std(img1) × std(img2) + C3)

    Where C3 = (0.03 × R)² / 2 and R is the data range.

    Valid CTC SSIM range (empirically determined): 0.495 – 0.917

    Args:
        img1 (np.ndarray): Candidate cell contour patch (binary, thresholded).
        img2 (np.ndarray): Ideal circle of equivalent area (binary).
        data_range (float, optional): Dynamic range of the images.
                                      Auto-detected from dtype if not provided.

    Returns:
        float: SSIM structural similarity score. Higher = more circular.
    """
    # Auto-detect data range from image dtype if not specified
    if data_range is None:
        if img1.dtype != img2.dtype:
            warn("Inputs have mismatched dtype. Setting data_range based on "
                 "img1.dtype.", stacklevel=2)
        dmin, dmax = dtype_range[img1.dtype.type]
        data_range = dmax - dmin

    R = data_range
    K2 = 0.03
    C2 = (K2 * R) ** 2   # Stability constant to avoid division by zero
    C3 = C2 / 2           # Structural component stability constant

    # Compute standard deviations (measure of contrast/texture in each patch)
    stdev1 = np.std(img1)
    stdev2 = np.std(img2)

    # Compute cross-covariance (measure of structural similarity)
    cov = _covariance(img1, img2)

    # Structural similarity: ratio of shared structure to total structure
    S = (cov + C3) / (stdev1 * stdev2 + C3)

    return S


# ══════════════════════════════════════════════════════════════════════
# SECTION 3: CONTRAST UTILITIES (experimental)
# These functions were used during exploratory data analysis to improve
# image visibility. They are NOT part of the main detection pipeline.
# ══════════════════════════════════════════════════════════════════════

def modify_contrast_and_brightness(img: np.ndarray) -> np.ndarray:
    """
    Increase image contrast using standard alpha/beta linear scaling.

    Formula: output = alpha × input + beta
        - alpha > 1 increases contrast (spreads intensity range)
        - alpha < 1 decreases contrast
        - beta adjusts overall brightness

    Note: This method increases contrast for bright regions but does
    NOT make dark regions darker — a known limitation.

    Args:
        img (np.ndarray): Input grayscale or BGR image.

    Returns:
        np.ndarray: Contrast-enhanced image, clipped to [0, 255].
    """
    array_alpha = np.array([2.0])  # Contrast multiplier (>1 increases contrast)
    array_beta = np.array([0.0])   # Brightness offset (0 = no brightness change)

    img = cv2.add(img, array_beta)       # Apply brightness offset
    img = cv2.multiply(img, array_alpha) # Apply contrast scaling
    img = np.clip(img, 0, 255)           # Clamp to valid uint8 range

    return img


def modify_contrast_and_brightness2(img: np.ndarray,
                                     brightness: int = 0,
                                     contrast: int = 100) -> np.ndarray:
    """
    Adjust image contrast using tangent-based nonlinear scaling.

    Unlike the linear method above, this formula achieves true
    "make darks darker AND lights lighter" contrast enhancement
    by applying a tangent-based scaling curve:

        k = tan((45 + 44 × c) / 180 × π)
        output = (input − 127.5 × (1 − B)) × k + 127.5 × (1 + B)

    Where c = contrast / 255.0 and B = brightness / 255.0.
    Negative contrast values reduce contrast (push values toward gray).

    Args:
        img (np.ndarray): Input grayscale or BGR image.
        brightness (int): Brightness adjustment (-255 to 255). Default 0.
        contrast (int): Contrast adjustment (-255 to 255).
                        Positive = increase, negative = decrease. Default 100.

    Returns:
        np.ndarray: Contrast-adjusted image, clipped to [0, 255] uint8.
    """
    B = brightness / 255.0
    c = contrast / 255.0
    k = math.tan((45 + 44 * c) / 180 * math.pi)  # Nonlinear contrast scaling factor

    img = (img - 127.5 * (1 - B)) * k + 127.5 * (1 + B)
    img = np.clip(img, 0, 255).astype(np.uint8)

    return img
