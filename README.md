# Point Cloud Registration with ICP and KD-Tree from Scratch (NumPy Only)

High-performance point cloud registration pipeline fully implemented from scratch using NumPy.  
Demonstrated on both synthetic rigid-transformed data and real multi-view scans of the Stanford Bunny.

This project is based on the algorithmic work completed during an engineering internship,  
where I handled **600k+ point clouds**.  
Due to confidentiality, the original dataset is not included here.

---

## 🚀 Features

- ICP (Iterative Closest Point) implemented from scratch
- KD-Tree accelerated nearest-neighbor search — **O(N log N)**
- Works on **large-scale point clouds (600k+)**
- PCA-based initial alignment for partial overlap cases
- Supports multi-view scan registration
- Visualization for evaluation: before/after alignment, error plots

---

## 📂 Project Structure
src/ # Core implementations (ICP, KD-Tree, utilities) 
data/ # Public sample dataset (Stanford Bunny) 
results/ # Visualization images/GIFs 
README.md

## 📊 Results

### Synthetic Rigid Transformation
Recovering known rotation and translation applied to a single point cloud.

| Stage | Visualization |
|------|---------------|
| Initial | (image) |
| After ICP | (image) |
| Error curve | (image) |

✨ ICP converges correctly and restores the ground-truth motion.

---

### Real Multi-View Scans (bun000 ↔ bun045)
Partial scans with limited overlapping regions.

| Stage | Visualization |
|------|---------------|
| Initial (PCA) | (image) |
| Stage 1 ICP | (image) |
| Stage 2 ICP | (image) |

> Overlapping regions (back/head) align well.  
> Non-overlapping areas remain offset, which is expected in practical scanning scenarios.

---

## 🔧 Algorithms

### ICP – SVD-based Rigid Registration

1. Nearest-neighbor correspondence (KD-Tree)
2. Reject far correspondences (optional trimming)
3. Compute optimal rotation & translation using SVD
4. Iterate until convergence

Formula:
\[
R, t = \arg\min_{R,t} \sum_i \| R p_i + t - q_i \|^2
\]

---

### KD-Tree – Efficient Nearest Neighbor Search
Reduces search complexity from **O(N²)** to **O(N log N)**.

> Implemented with index permutation and median splitting for memory locality.

---
## 📌 Performance Notes
Original internship work used large-scale indoor scans:
｜Dataset	｜Points	｜Method	｜Result
｜Confidential project	｜ 600k+	｜ KD-Tree ICP	｜✔ Successful
｜Stanford Bunny	｜~30k	｜KD-Tree ICP	｜Demo included
