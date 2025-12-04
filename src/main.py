import numpy as np
from initialize import read_ply, voxel_downsample_ply, align_center_scale
from visualize import show_two_clouds
from pca_coarse import pca_coarse_align_numpy
from icp import icp_numpy_stable
from rmse import rmse


def main():

    # ====================================
    # STEP 1: Load raw data
    # ====================================
    print("===== STEP 1: Load data =====")
    points1 = read_ply("../data/bun000.ply")
    points2 = read_ply("../data/bun045.ply")

    # ====================================
    # STEP 2: Downsample (voxel size = 0.002)
    # ====================================
    print("===== STEP 2: Downsample (0.002) =====")
    down_points1 = voxel_downsample_ply(points1, 0.002)
    down_points2 = voxel_downsample_ply(points2, 0.002)

    print("down_points1.shape =", down_points1.shape)
    print("down_points1 size =", np.size(down_points1))

    show_two_clouds(down_points1, down_points2, title="Initial clouds")

    # ====================================
    # STEP 3: Center + Scale Alignment
    # ====================================
    print("===== STEP 3: Center + Scale Alignment =====")
    X_aligned, s, cX, cY = align_center_scale(
        down_points1, down_points2, method="rms", pct=0.95, eps=1e-12
    )

    # ====================================
    # STEP 4: PCA Coarse Alignment
    # ====================================
    print("===== STEP 4: PCA Coarse Alignment =====")
    coarse_aligned, R_init, t_init, rmse_internal_coarse = pca_coarse_align_numpy(
        X_aligned, down_points2
    )

    rmse_coarse = rmse(X_aligned, down_points2, R_init, t_init)
    print(f"PCA coarse alignment RMSE = {rmse_coarse:.6f}")
    show_two_clouds(coarse_aligned, down_points2, title="PCA Coarse Alignment")

    # ====================================
    # STEP 5: ICP Stage 1 (downsampled 0.1)
    # ====================================
    print("===== STEP 5: ICP Stage 1 =====")
    icp1_aligned, (R_icp, t_icp), rmse_internal_icp1 = icp_numpy_stable(
        coarse_aligned, down_points2
    )

    rmse_icp1 = rmse(coarse_aligned, down_points2, R_icp, t_icp)
    print(f"RMSE ICP stage1 = {rmse_icp1:.6f}")
    show_two_clouds(coarse_aligned, down_points2, title="ICP Stage 1 (Downsampled)")

    # ====================================
    # STEP 6: ICP Stage 2 (voxel size = 0.05)
    # ====================================
    print("===== STEP 6: ICP Stage 2 (downsample 0.001) =====")
    mid_points1 = voxel_downsample_ply(points1, 0.001)
    mid_points2 = voxel_downsample_ply(points2, 0.001)

    R_mid_init = R_icp @ R_init
    t_mid_init = R_icp @ t_init + t_icp

    icp2_aligned, (R_mid, t_mid), rmse_internal_icp2 = icp_numpy_stable(
        mid_points1, mid_points2, iters=80, trim=0.9, init_RT=(R_mid_init, t_mid_init)
    )

    rmse_icp2 = rmse(mid_points1, mid_points2, R_mid, t_mid)
    print(f"RMSE ICP stage2 = {rmse_icp2:.6f}")
    show_two_clouds(icp2_aligned, mid_points2, title="ICP Stage 2 (Downsampled)")

    # ====================================
    # STEP 7: Evaluate RMSE on full resolution point cloud
    # ====================================
    print("===== STEP 7: Final Evaluation =====")
    rmse_full = rmse(points1, points2, R_mid, t_mid)
    print("RMSE full =", rmse_full)

    full_aligned = (points1 @ R_mid.T) + t_mid
    show_two_clouds(
        full_aligned,
        points2,
        title="Final Alignment (Full Point Clouds)",
    )


if __name__ == "__main__":
    main()
