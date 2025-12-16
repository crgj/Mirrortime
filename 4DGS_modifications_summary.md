# 4DGS Codebase Modifications Summary

This document outlines the key modifications made to the standard 3D Gaussian Splatting (3DGS) codebase to support **4D Datasets (Video)** and the **Lifetime** feature (Temporal Existence).

## 1. 4D Dataset Support
The goal is to load and render dynamic scenes where the scene content (and potentially camera poses) changes over time.

### **Dataset Readers (`scene/dataset_readers.py`)**
*   **Frame Directory Structure**: Implemented `read4DGSSceneInfo` to handle datasets organized by time steps (e.g., `frame000`, `frame001`).
*   **Temporal Iteration**: The loader iterates through all frame directories to load cameras for each specific timestamp.
*   **Time Indexing**: A `time_idx` is assigned to every camera, identifying which frame it belongs to.
*   **Shared Parameters**: Added robust logic to check the root `sparse/0` directory for shared intrinsics/extrinsics. If found, these are loaded once and reused for all frames to save I/O and memory.
*   **PLY Initialization**: Prioritizes loading the initial Point Cloud from the root `sparse/0/points3D.ply` before falling back to individual frames.

### **Cameras (`scene/cameras.py`)**
*   **Time Attribute**: Added `self.time_idx` to the `Camera` class. This allows the renderer to know the exact time $t$ for any given view, which is essential for querying time-dependent Gaussian attributes (like opacity).
*   **Memory Management**: Introduced `to_device()` and `release()` methods to dynamically move camera data between CPU and GPU. This is critical for 4D training to prevent VRAM exhaustion when dealing with thousands of video frames.

### **Training Loop (`train.py`)**
*   **DataLoader**: Switched to using a PyTorch `DataLoader` instead of pre-loading all cameras. This supports lazy loading and efficient batching for large video datasets.
*   **Batch Recycling**: Implemented a logic to reuse the same batch of loaded cameras for `opt.batch_iterations` steps to amortize the expensive CPU-to-GPU transfer (PCIe bottleneck).

---

## 2. Lifetime (Temporal Existence)
The "Lifetime" feature models the *temporal lifespan* of each Gaussian, allowing points to appear and disappear smoothly over time.

### **Gaussian Model (`scene/gaussian_model.py`)**
*   **New Attributes**:
    *   `_lifetime_mu`: The temporal center (when the point is most active).
    *   `_lifetime_w`: The temporal half-width (duration/2).
    *   `_lifetime_k`: The temporal sharpness (controls how "hard" the appearance/disappearance is).
*   **Double Sigmoid Function**: Implemented `lifetime(t)` as a soft "Box Function" using two sigmoids:
    $$ \text{Lifetime}(t) = \sigma(k \cdot (t - (\mu - w))) \times \sigma(-k \cdot (t - (\mu + w))) $$
*   **Opacity Query**: `get_opacity_at_time(t)` computes the final effective opacity:
    $$ \text{Opacity}(t) = \text{Base Opacity} \times \text{Lifetime}(t) $$
*   **Initialization**: In `create_from_pcd`, lifetime parameters are initialized to cover the video duration roughly, with some randomization to break symmetry.

### **Training & Pruning (`train.py`)**
*   **Rendering**: The render call now extracts `time_idx` from the camera and uses it to modulate Gaussian opacity.
*   **Active Duration Statistics**: Calculates and logs `active_duration` (how long points stay visible) to monitor temporal consistency.
*   **PUP Pruning Safeguard**:
    *   Standard PUP pruning removes points with low "sensitivity" (Fisher Information).
    *   **Safeguard**: We added a check to **protect dynamic points**. Points that are short-lived (dynamic) might have low total sensitivity but are crucial for their specific frames.
    *   **Logic**: We only prune points that are **Low Sensitivity** AND **Long Duration (Static)**. Dynamic points are preserved.

---

## Summary of Files Modified
| File | Key Changes |
| :--- | :--- |
| `scene/dataset_readers.py` | 4D folder parsing, `time_idx` assignment, shared camera params. |
| `scene/cameras.py` | `time_idx` attribute, VRAM management methods. |
| `scene/gaussian_model.py` | `_lifetime_*` attributes, Double Sigmoid logic, initialization. |
| `train.py` | `DataLoader`, batch recycling, Lifetime stats, PUP dynamic safeguard. |
