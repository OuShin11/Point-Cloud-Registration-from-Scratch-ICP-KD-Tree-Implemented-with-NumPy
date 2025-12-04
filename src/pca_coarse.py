import numpy as np
from kdtreenp import KDTreeNP


def pca_axes(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """点群に対して PCA 主軸を推定する。

    点群の中心を計算し、共分散行列の固有分解によって
    主成分（固有ベクトル）を求める。固有値の大きい順に
    主軸を並べ替えて返す。

    Args:
        points (np.ndarray): 点群 (N, 3)。

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - axes (np.ndarray): 主軸ベクトル（3×3 行列）。列が主成分方向。
            - center (np.ndarray): 点群の中心 (3,)。
    """
    c = points.mean(axis=0)
    X = points - c
    C = X.T @ X
    _, V = np.linalg.eigh(C)
    return V[:, ::-1], c



def numpy_generate_permutations() -> np.ndarray:
    """ベクトル [0, 1, 2] の 6 通りの全排列を NumPy のみで生成する。

    Returns:
        np.ndarray: 形状 (6, 3) の整数配列。各行が異なる排列。
    """
    base = np.array([0, 1, 2])
    perms = []

    # swap操作で全順列生成
    perms.append(base.copy())
    perms.append(base[[0, 2, 1]])
    perms.append(base[[1, 0, 2]])
    perms.append(base[[1, 2, 0]])
    perms.append(base[[2, 0, 1]])
    perms.append(base[[2, 1, 0]])

    return np.array(perms)


def numpy_generate_signs() -> np.ndarray:
    """3 次元ベクトルに対する符号組み合わせ（±1）を NumPy で生成する。

    3 次元の直積（+1, -1）^3 により、8 種類の符号パターンを出力する。

    Returns:
        np.ndarray: 形状 (8, 3) の配列。各行が [+/-1, +/-1, +/-1]。
    """
    s = np.array([1, -1])
    # 3 次元のデカルト積で全組み合わせを生成（8通り）
    signs = np.stack(np.meshgrid(s, s, s), axis=-1).reshape(-1, 3)
    return signs  # shape = (8,3)


def pca_coarse_align_numpy(
    src: np.ndarray, tgt: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """PCA と軸排列・符号反転を用いた粗い姿勢合わせ（回転＋平行移動）。

    点群 src と tgt に対して PCA 主軸を求め、
    軸の排列（6 通り）×符号組み合わせ（8 通り）の計 48 通りの
    回転候補を総当たりで試し、RMSE が最小となる
    粗アラインメント（回転 R, 平行移動 t）を推定する。

    ICP の初期値として使用することを想定している。

    Args:
        src (np.ndarray): ソース点群 (N, 3)。
        tgt (np.ndarray): ターゲット点群 (M, 3)。

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, float]:
            - aligned (np.ndarray): 最良の粗アラインメント後の src 点群。
            - best_R (np.ndarray): 最適回転行列 (3, 3)。
            - best_t (np.ndarray): 最適平行移動ベクトル (3,)。
            - best_rmse (float): 最小 RMSE。

    Notes:
        - PCA 主軸は符号不定性を持つため、排列・符号反転を総当たりする。
        - 48 通りと少なく、NumPy のみでも計算負荷は軽い。
        - 粗アラインメントであり、高精度化には後続の ICP が必要。
    """
    # 中心化
    src_c = src - src.mean(axis=0)
    tgt_c = tgt - tgt.mean(axis=0)

    # PCA 主軸
    R1, _ = pca_axes(src_c)
    R2, _ = pca_axes(tgt_c)

    # 48 個の回転行列を生成
    perms = numpy_generate_permutations()  # (6,3)
    signs = numpy_generate_signs()  # (8,3)

    time = 0
    best_rmse = 1e18
    best_R = None
    best_t = None
    best_aligned = None

    tree = KDTreeNP(tgt)

    for p in perms:  # 6 種類の軸並び
        for s in signs:  # 8 種類の符号組み合わせ

            time += 1
            print(f"\n=== Loading {time}/48 ===")
            # permutation + sign による回転行列を構築
            Rperm = np.zeros((3, 3))
            Rperm[np.arange(3), p] = s

            # 最終回転
            R = R2 @ Rperm @ R1.T

            # 平行移動
            t = tgt.mean(axis=0) - (src.mean(axis=0) @ R.T)

            aligned = src @ R.T + t
            _, d = tree.query(aligned)
            rmse = np.sqrt(np.mean((d**2)))

            if rmse < best_rmse:
                best_rmse = rmse
                best_R = R
                best_t = t
                best_aligned = aligned

    return best_aligned, best_R, best_t, best_rmse
