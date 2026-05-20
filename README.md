# Automated Circulating Tumor Cell (CTC) Detection

> **Clinical Deployment:** Chang Gung Memorial Hospital, Taiwan  
> **Key Result:** 86% reduction in false negatives vs. manual screening  
> **Data Source:** 20 colorectal cancer patients undergoing immunotherapy


## Background

Circulating Tumor Cells (CTCs) are cancer cells that detach from primary tumors and enter the bloodstream, their presence is a critical indicator for cancer metastasis monitoring. However, in 1mL of blood, there are **billions of red blood cells** and potentially only **a handful of CTCs**, making manual detection extremely difficult and time-consuming.

This project builds an automated end-to-end CTC detection pipeline from multi-channel fluorescence microscopy images of blood samples collected via **Self-Assembly Cell Array (SACA) microfluidic chips** from colorectal cancer patients at Chang Gung Memorial Hospital.


## Data Description

Blood samples are imaged across **3 fluorescence wavelength channels**, each targeting a different cell type:

| Channel | Wavelength | Antibody | Target |
|---------|-----------|----------|--------|
| UV | 365 nm | Hoechst | Cell nucleus |
| FITC | 488 nm | EpCAM | Tumor cells (CTCs) |
| APC | 630 nm | CD45 | Leukocytes (white blood cells) |

**Why EpCAM?** EpCAM is a transmembrane glycoprotein expressed on epithelial tissue and epithelial-origin tumor cells, but **not present in normal blood cells** — making it the primary marker for CTC identification.

**Dataset:** 20 colorectal cancer patients receiving immune checkpoint inhibitor therapy, sampled at 4 timepoints:
- Pre-treatment (-1~0 weeks)
- Early response (3~4 weeks)  
- Standard response (3 months)
- Late response (6 months)


## Pipeline Architecture

<img width="1659" height="948" alt="image" src="https://github.com/user-attachments/assets/eef459b5-6e62-4c02-8fc4-d5c029455364" />



## Key Technical Details

### Image Preprocessing
The Hoechst (365nm) channel — which stains all cell nuclei — is used to determine the physical boundary of the blood sample slide. A circular mask is then applied to the EpCAM image to ensure all downstream analysis is confined to the actual blood sample region, eliminating edge artifacts.

### Candidate Detection
Local maxima in the EpCAM channel are identified using `skimage.feature.peak_local_max` with a minimum inter-peak distance of 20 pixels. Each candidate is extracted as a 40×40 pixel patch — large enough to capture full cell morphology while keeping analysis tractable.

### Biological Feature Extraction
Rather than relying on a single intensity threshold, the pipeline computes **5 complementary indices** that together encode clinical knowledge about what a true CTC looks like:

- **SSIM** compares the cell contour shape against its expected circular morphology — CTCs should be roughly circular
- **IoU** between EpCAM and Hoechst contours validates that the detected bright spot corresponds to an actual cell nucleus
- **Diameter and Lightness** filter out debris, artifacts, and non-specific staining

### Threshold Computation
A robust threshold is computed per image using the **IQR method**:
```python
threshold = Q50 + 4 × (Q75 - Q50)
```
This adapts to per-image fluorescence intensity variation without requiring manual calibration.

### Segmentation
Cell contours are delineated using **Otsu thresholding** on the EpCAM patch. A **Gaussian Mixture Model (2 components)** is additionally fitted to determine the boundary between background and foreground fluorescence in both 488nm and 365nm channels.


## Results

| Metric | Result |
|--------|--------|
| False Negative Reduction | **86%** vs. manual baseline |
| Primary Validation Metric | SSIM + IoU benchmarking |
| Clinical Deployment | Chang Gung Memorial Hospital, Taiwan |
| Patient Cohort | 20 colorectal cancer patients |
| Timepoints Analyzed | 4 per patient |


## Tech Stack

```
Language:         Python 3
Image Processing: OpenCV (cv2), scikit-image
Scientific:       NumPy, SciPy, scikit-learn (GaussianMixture)
Visualization:    Matplotlib
UI:               Tkinter
```


## Repository Structure

```
Detection-of-CTC/
├── main.py          # Main pipeline: detection, feature extraction, visualization
├── threshold.py     # Threshold computation and SSIM implementation
├── mask_.py         # Slide boundary detection and mask generation for target area
├── data/            # Input fluorescence images (not included - patient data)
└── README.md
```


## Clinical Context

This pipeline was developed for the **liquid biopsy** workflow — a non-invasive alternative to surgical tumor biopsy. By analyzing a patient's blood sample rather than requiring surgery, CTCs can be tracked longitudinally to:

- Monitor cancer progression without invasive procedures
- Evaluate treatment response over time
- Identify potential cancer metastasis early

The system was designed to **augment, not replace** clinical judgment: the automated pipeline surfaces candidates with quantitative metrics, while clinicians make the final determination through the UI — combining computational speed with clinical expertise.


## Background Reference

Developed as undergraduate independent research at the **Data Analysis and Interpretation Laboratory, National Tsing Hua University (NTHU), Taiwan** (Sep 2021 – Sep 2022), supervised by Prof. Shun-Chi Wu.


## Author

**Wan-Lin (Christine) Chen**  
M.S. Bioengineering, UC San Diego  
[LinkedIn](https://linkedin.com/in/wan-lin-chen) | [GitHub](https://github.com/WanLinChen)
![CTC(UI)](https://raw.githubusercontent.com/WanLinChen/Detection-of-CTC/master/CTC.jpg)
