import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from kdtreenp import KDTreeNP


# 剛体変換
def svd_rigid(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """SVD を用いて点群 A を点群 B に剛体変換で合わせる。

    点群 A, B の対応を仮定し、両者の中心を一致させた上で
    SVD により最適な回転行列 R と平行移動ベクトル t を求める関数。
    距離（二乗誤差）を最小化する剛体変換（点到点 ICP に相当）を返す。

    Args:
        A (np.ndarray): ソース点群 (N, 3)。
        B (np.ndarray): ターゲット点群 (N, 3)。A と同じ順番で対応していると仮定。

    Returns:
        tuple[np.ndarray, np.ndarray]:
            R (np.ndarray): 3×3 の最適回転行列。
            t (np.ndarray): 3 次元の平行移動ベクトル。

    Raises:
        ValueError: 行列の次元が不正な場合に発生する可能性あり。

    """
    cA, cB = A.mean(axis=0), B.mean(axis=0)
    A0, B0 = A - cA, B - cB
    H = A0.T @ B0
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = cB - R @ cA
    return R, t


# ICP（point to point）
def icp_numpy_stable(
    src: np.ndarray[np.float64],
    tgt: np.ndarray[np.float64],
    iters: int = 60,
    trim: float = 0.8,
    subsample: int = 40000,
    early: float = 1.01,
    init_RT: tuple[np.ndarray[np.float64], np.ndarray[np.float64]] | None = None,
) -> tuple[
    np.ndarray[np.float64], tuple[np.ndarray[np.float64], np.ndarray[np.float64]], float
]:
    """NumPy のみを用いた安定版 ICP（点対点）アルゴリズム。

    最近傍探索（KDTree）、トリム（trim）、サブサンプリング、Early-stop などを
    組み合わせて、効率性と安定性を両立した ICP 実装。
    初期剛体変換（R0, t0）を与えることで、収束性を向上させることも可能。

    Args:
        src (np.ndarray[np.float64]): ソース点群 (N, 3)。
        tgt (np.ndarray[np.float64]): ターゲット点群 (M, 3)。
        iters (int): 最大反復回数。デフォルトは 60。
        trim (float): トリム率。距離が近い上位 trim 割合の対応のみ使用。
        subsample (int): 各反復で使用する最大点数。大規模点群の高速化に使用。
        early (float): Early-stop 係数。RMSE が最良値 × early を超えた場合に打ち切る。
        init_RT (tuple[np.ndarray, np.ndarray] | None):
            初期剛体変換 (R0, t0)。None の場合は未使用。

    Returns:
        tuple[np.ndarray, tuple[np.ndarray, np.ndarray], float]:
            3 要素タプル：
            - best_S (np.ndarray): 最良の剛体変換後のソース点群。
            - (best_R, best_t): 最終的な回転行列と並進ベクトル。
            - best_rmse (float): 収束時の RMSE。

    Raises:
        ValueError: trim が (0, 1] の範囲外の場合など、不正なパラメータが入力された時。

    Notes:
        - 最近傍探索には自作 KDTreeNP を使用（完全 NumPy ベース）。
        - トリムを用いることで外れ値の影響を軽減する。
        - Early-stop により、発散傾向の反復を素早く停止できる。
        - init_RT を与えると初期位置合わせが改善され、収束が速くなる場合がある。

    """
    S = src.copy()
    R_tot = np.eye(3)
    t_tot = np.zeros(3)

    if init_RT is not None:
        R0, t0 = init_RT
        S = (R0 @ S.T).T + t0
        R_tot, t_tot = R0.copy(), t0.copy()

    # RMSEの収束曲線を書く
    rmses = []

    best_rmse = np.inf
    best_S = S.copy()
    best_R, best_t = R_tot.copy(), t_tot.copy()

    # KDTree を使用して最近傍点を探索
    tree = KDTreeNP(tgt, leaf_size=32, axis_rule="var")  # B: (M,3)

    for it in range(iters):

        #
        print(f"\n=== Iteration {it+1}/{iters} ===")

        # ランダムにダウンサンプリングをする
        if subsample and len(S) > subsample:
            idx = np.random.choice(len(S), subsample, replace=False)
            S_sub = S[idx]
        else:
            S_sub = S

        # KDTree を使用して最近傍点を探索
        j, d = tree.query(S_sub)

        # トリミング（距離が上位 1 - trim の点を除外）
        K = int(len(d) * trim)
        keep = np.argpartition(d, K)[:K]  # O(N)
        A = S_sub[keep]
        B = tgt[j[keep]]

        # SVD による剛体変換の更新（方向に注意）
        R, t = svd_rigid(A, B)
        S = (R @ S.T).T + t

        # 変換の累積（順序に注意）
        R_tot = R @ R_tot
        t_tot = R @ t_tot + t

        rmse = np.sqrt(np.mean((d[keep] ** 2)))
        rmses.append(rmse)

        # 早期終了 + ロールバック
        if rmse < best_rmse:
            best_rmse = rmse
            best_S = S.copy()
            best_R, best_t = R_tot.copy(), t_tot.copy()
        elif rmse > best_rmse * early:
            S = best_S
            R_tot, t_tot = best_R, best_t
            break

        print(f"   RMSE: {rmse:.4f}")

    # RMSEの収束曲線を書く
    plt.plot(rmses)
    plt.xlabel("iteration")
    plt.ylabel("RMSE")
    plt.title("ICP RMSE Convergence Curve")
    plt.grid(True)
    plt.show()

    return best_S, (best_R, best_t), best_rmse
