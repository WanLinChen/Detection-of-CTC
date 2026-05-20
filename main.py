# -*- coding: utf-8 -*-
"""
CTC Detection Pipeline - Main Analysis Script
==============================================
Automated detection of Circulating Tumor Cells (CTCs) from
multi-channel fluorescence microscopy images of blood samples.

Pipeline:
    1. Load multi-channel fluorescence images (EpCAM, Hoechst, CD45)
    2. Apply slide mask to restrict analysis to blood sample region
    3. Detect candidate CTC spots via local maximum detection
    4. Extract 40x40 pixel patches around each candidate
    5. Compute 5 biological indices per candidate:
       - Diameter, Lightness, SSIM, Distance-to-nucleus, IoU
    6. Generate multi-channel visualization panels
    7. Save output images for clinical UI review

Author: Wan-Lin (Christine) Chen
Date: December 2022
Institution: Data Analysis and Interpretation Lab, NTHU
"""

import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import math
from skimage.feature import peak_local_max
from sklearn.mixture import GaussianMixture
from threshold import compute_threshold, compare_structure
from scipy.spatial import distance
from skimage import measure, color
from sklearn.metrics import confusion_matrix
from skimage.measure import label, regionprops, regionprops_table
from mask_ import get_mask

# ─────────────────────────────────────────────
# STEP 1: Load fluorescence images
# Each channel targets a different cell type:
#   col1 (_0.jpg) → 488nm EpCAM  → tumor cells
#   col2 (_1.jpg) → 365nm Hoechst → cell nuclei
#   col3 (_3.jpg) → 630nm CD45   → leukocytes
# ─────────────────────────────────────────────
DATADIRECTORY = r"/data/"
current_dir = os.path.abspath(os.path.dirname(__file__))
img_list = glob.glob(os.path.join(current_dir + DATADIRECTORY, "*.jpg"))

for i in img_list:
    if 'F_' and '_0.jpg' in i:
        img_col1 = cv2.imread(i)   # 488nm EpCAM channel (tumor marker)
    elif 'F_' and '_1.jpg' in i:
        img_col2 = cv2.imread(i)   # 365nm Hoechst channel (nucleus stain)
    elif 'F_' and '_3.jpg' in i:
        img_col3 = cv2.imread(i)   # 630nm CD45 channel (leukocyte marker)
    else:
        img_white = cv2.imread(i)  # White light image for morphology reference

# ─────────────────────────────────────────────
# STEP 2: Apply slide boundary mask
# Use Hoechst image to detect slide boundary,
# then apply mask to EpCAM image to restrict
# analysis to the actual blood sample region.
# ─────────────────────────────────────────────
img = get_mask(img_col1, img_col2)  # Returns masked grayscale EpCAM image
img1 = img

# Apply Gaussian blur to Hoechst channel to reduce noise before nucleus detection
blur_col1 = cv2.GaussianBlur(img_col1, (5, 5), 0)

# ─────────────────────────────────────────────
# STEP 3: Detect candidate CTC locations
# Find all local intensity maxima in the EpCAM
# image — these are candidate CTC locations.
# min_distance=20 ensures each peak corresponds
# to a distinct cell (typical CTC diameter < 20px)
# ─────────────────────────────────────────────
thr = compute_threshold(img1)  # Adaptive threshold: Q50 + 4*(Q75-Q50)
xy = peak_local_max(img, min_distance=20, threshold_abs=thr)  # Returns (row, col) coordinates

# Initialize result lists for all 5 biological indices
SIM = []    # SSIM scores (shape similarity to circle)
DTC = []    # Distance-to-center (EpCAM centroid to nearest Hoechst nucleus)
IOU = []    # IoU between EpCAM and Hoechst contours
m = []      # Candidate index list
light_mean = []  # Mean fluorescence intensity of detected contours
y_pred = []      # Predictions (reserved for future classifier integration)

# ─────────────────────────────────────────────
# STEP 4: Per-candidate feature extraction
# For each candidate spot, extract a 40x40 patch
# and compute all 5 biological indices.
# ─────────────────────────────────────────────
for n in range(len(xy)):

    # Extract 40x40 pixel patches centered on each candidate peak
    # from all fluorescence channels simultaneously
    img_sep = img1[xy[n, 0]-20:xy[n, 0]+20, xy[n, 1]-20:xy[n, 1]+20]
    img_sep1 = img1[xy[n, 0]-20:xy[n, 0]+20, xy[n, 1]-20:xy[n, 1]+20]

    img_sepb_col1 = blur_col1[xy[n, 0]-20:xy[n, 0]+20, xy[n, 1]-20:xy[n, 1]+20]   # Blurred Hoechst patch
    img_sep_col2 = img_col2[xy[n, 0]-20:xy[n, 0]+20, xy[n, 1]-20:xy[n, 1]+20]     # Hoechst patch
    img_sep_col3 = img_col3[xy[n, 0]-20:xy[n, 0]+20, xy[n, 1]-20:xy[n, 1]+20]     # CD45 patch
    img_sep_white = img_white[xy[n, 0]-20:xy[n, 0]+20, xy[n, 1]-20:xy[n, 1]+20]   # White light patch

    img_seperation = img[xy[n, 0]-20:xy[n, 0]+20, xy[n, 1]-20:xy[n, 1]+20]        # EpCAM patch (grayscale)
    img_seperation_BGR = cv2.cvtColor(img_seperation, cv2.COLOR_GRAY2BGR)           # Convert for colored overlay

    # ── INDEX ①: Compute contour centroid (intensity-weighted center of mass) ──
    cx = 0
    cy = 0
    s = np.sum(img_seperation)  # Total intensity (normalization factor)
    for i in range(40):
        cx += i * (np.sum(img_seperation[i, :])) / s   # Row-weighted centroid
        cy += i * (np.sum(img_seperation[:, i])) / s   # Col-weighted centroid

    # ── Fit Gaussian Mixture Model to EpCAM patch to determine background/foreground threshold ──
    GMM = GaussianMixture(n_components=2)
    GMM.fit_predict(img_seperation)
    thr_488 = round(np.mean(GMM.means_), 3)  # Mean of two GMM components as threshold

    # ── Apply Otsu thresholding for binary segmentation of EpCAM contour ──
    ret1, thresh = cv2.threshold(img_seperation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Isolate the red channel to display EpCAM contour in red
    tmp = img_seperation_BGR[:, :, 0]
    tmp[tmp < ret1] = 0    # Background → black
    tmp[tmp > ret1] = 250  # Foreground → bright red
    img_seperation_BGR[:, :, 0] = tmp
    img_seperation_BGR[:, :, 1] = 0  # Zero out green channel
    img_seperation_BGR[:, :, 2] = 0  # Zero out blue channel

    # Extract bright (foreground) pixels for lightness calculation
    light_488 = img_seperation.copy()
    light_488[light_488 < ret1] = 0  # Suppress background pixels

    # ── INDEX ②: Diameter — compute equivalent circle diameter from EpCAM contour area ──
    list_488 = img_seperation_BGR[:, :, 0].flatten()
    num_488 = sum(1 for val in list_488 if val == 250)  # Count foreground pixels
    area = num_488
    radius = math.sqrt(area / np.pi)   # Equivalent circle radius
    diameter = 2 * radius              # Equivalent circle diameter (valid CTC range: 6–16 nm)

    # Draw the equivalent circle on a black background for SSIM comparison
    img_black = np.zeros((40, 40, 3), np.uint8)
    img_black_circle = cv2.circle(img_black, (round(cx), round(cy)), round(radius), (255, 255, 255), -1)

    # ── INDEX ③: SSIM — compare EpCAM contour shape against its equivalent circle ──
    # A true CTC should be approximately circular → high SSIM (valid range: 0.495–0.917)
    img_seperation[img_seperation < thr] = 0
    img_seperation[img_seperation > thr] = 255
    img_black_circle = cv2.cvtColor(img_black_circle, cv2.COLOR_BGR2GRAY)
    ssim = compare_structure(img_seperation, img_black_circle)
    SIM.append(ssim)

    # ── INDEX ②: Lightness — mean fluorescence intensity of EpCAM contour region ──
    # (valid CTC range: 34.317–231.625)
    light = [val for val in light_488.flatten() if val > 0]
    light_488_mean = sum(light) / len(light)

    # Optional hard filter (commented out — currently using UI for human review)
    # if 6 <= round(diameter) <= 16 and 0.572 <= round(ssim,3) <= 0.783 \
    #    and 34.317 <= round(light_488_mean,3) <= 231.625:

    m.append(n)
    print(f"Candidate {n}")
    print(f"  Diameter:  {round(diameter)}")
    print(f"  Lightness: {light_488_mean:.3f}")
    print(f"  SSIM:      {round(ssim, 3)}")
    print("─" * 35)

    # ── Process Hoechst (365nm) channel for nucleus detection ──
    # Apply erosion to reduce noise, then fit GMM to separate nucleus from background
    kernel = np.ones((3, 3), np.uint8)
    erosion_365 = cv2.erode(img_sepb_col1, kernel, iterations=1)
    dilation_365_gray = cv2.cvtColor(erosion_365, cv2.COLOR_BGR2GRAY)

    GMM = GaussianMixture(n_components=2)
    GMM.fit(dilation_365_gray)
    thr_365 = round(np.mean(GMM.means_), 3)  # GMM-derived threshold for nucleus channel

    # Label connected components in Hoechst channel (each component = one nucleus)
    labels = measure.label(dilation_365_gray > thr_365, connectivity=2)
    regions = measure.regionprops(labels)
    props = regionprops_table(labels, properties=('centroid', 'area'))

    # ── INDEX ④: Distance-to-nucleus — find nearest Hoechst nucleus to EpCAM centroid ──
    # A true CTC should have a nucleus (Hoechst signal) co-localizing with EpCAM signal
    img_center = (20, 20)
    dictionaries = []
    for q in range(len(regions)):
        cx = props.get('centroid-0')[q]
        cy = props.get('centroid-1')[q]
        area = props.get('area')[q]
        contour_center = (cx, cy)
        dist = distance.euclidean(img_center, contour_center)
        dictionaries.append({'contour': q, 'center': contour_center,
                              'distance_to_center': dist, 'area': area})

    # Sort nuclei by distance to patch center → nearest nucleus is most likely the CTC nucleus
    sorted_distance = sorted(dictionaries, key=lambda i: i['distance_to_center'])
    DTC.append(sorted_distance[0]['distance_to_center'])

    # Extract the nearest nucleus contour mask (displayed in blue)
    area_img_cc = sorted_distance[0]['area']
    img_cc = labels == sorted_distance[0]['contour'] + 1
    img_cc_gray = (img_cc * 255).astype('uint8')
    img_cc_BGR = cv2.cvtColor(img_cc_gray, cv2.COLOR_GRAY2BGR)
    img_cc_BGR[:, :, 0] = 0           # Zero red channel
    img_cc_BGR[:, :, 1] = 0           # Zero green channel
    img_cc_BGR[:, :, 2] = img_cc_gray  # Blue channel = nucleus mask

    # ── INDEX ⑤: IoU — overlap between EpCAM contour (red) and Hoechst nucleus (blue) ──
    # High IoU means EpCAM and nucleus signals are co-localized → strong CTC indicator
    img_overlap = img_seperation_BGR + img_cc_BGR
    img_overlap_gray = cv2.cvtColor(img_overlap, cv2.COLOR_BGR2GRAY)
    dil_sum = img_overlap_gray.flatten()

    num_intersection = sum(1 for val in dil_sum if val > 0)    # Union area (either signal)
    num_inter = sum(1 for val in dil_sum if val > 100)         # Intersection area (both signals)
    iou_1 = num_inter / num_intersection                       # IoU = intersection / union
    IOU.append(iou_1)

    # ─────────────────────────────────────────────
    # STEP 5: Generate multi-channel visualization panel
    # Create a 2×5 subplot panel combining all channel images
    # and computed index values for clinical review.
    # ─────────────────────────────────────────────
    alpha = 0.35  # Overlay transparency weight

    # Create colored overlay images for each channel
    front = img_sep.copy()
    img_488_show = cv2.addWeighted(front, 1-alpha, img_seperation_BGR, alpha, 0)   # EpCAM + contour overlay
    img_365_show = cv2.addWeighted(img_sepb_col1, 1-alpha, img_cc_BGR, alpha, 0)   # Hoechst + nucleus overlay
    img_365630_over = cv2.addWeighted(img_sep_col3, 1-alpha, img_seperation_BGR, alpha, 0)  # CD45 + EpCAM overlap

    # Build visualization panel
    plt.figure(figsize=(10, 5))

    plt.subplot(2, 5, 1)
    plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    plt.title('Epcam_raw', {'fontsize': 10})
    plt.imshow(img_sep)

    plt.subplot(2, 5, 2)
    plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    plt.title('Epcam', {'fontsize': 10})
    plt.imshow(img_488_show)

    plt.subplot(2, 5, 3)
    plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    plt.title('hoechst', {'fontsize': 10})
    plt.imshow(img_365_show)

    plt.subplot(2, 5, 4)
    plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    plt.title('overlap', {'fontsize': 10})
    plt.imshow(img_overlap)

    plt.subplot(2, 5, 6)
    plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    plt.title('white', {'fontsize': 10})
    plt.imshow(img_sep_white)

    plt.subplot(2, 5, 7)
    plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    plt.title('CD45', {'fontsize': 10})
    plt.imshow(img_sep_col3)

    plt.subplot(2, 5, 8)
    plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    plt.title('Epcam & CD45 overlap', {'fontsize': 10})
    plt.imshow(img_365630_over)

    # Display computed index values as text annotations
    plt.text(50, 30, f'{n}_diameter: {round(diameter, 3)}', {'fontsize': 10})
    plt.text(50, 20, f'{n}_lightness: {round(light_488_mean, 3)}', {'fontsize': 10})
    plt.text(50, 10, f'{n}_SSIM: {round(ssim, 3)}', {'fontsize': 10})
    plt.text(100, 30, f'{n}_iou: {round(iou_1, 3)}', {'fontsize': 10})
    plt.text(100, 20, f'{n}_distance: {round(sorted_distance[0]["distance_to_center"], 3)}', {'fontsize': 10})

    # Save high-resolution output for clinical UI
    plt.savefig(f'output_{n}.jpg', dpi=1200, transparent=True)
