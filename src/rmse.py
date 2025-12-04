# 计算rmse
import numpy as np
from kdtreenp import KDTreeNP


def rmse(src: np.ndarray, tgt: np.ndarray, R: np.ndarray, t: np.ndarray) -> float:
    """剛体変換（R, t）を適用した点群の RMSE（最近傍誤差）を計算する。

    点群 src に回転 R と平行移動 t を適用して整列させ、
    KDTree を用いて tgt との最近傍距離を求めた後、
    その二乗平均平方根（RMSE）を返す。

    ICP の収束判定や精度評価に用いる関数。

    Args:
        src (np.ndarray): ソース点群 (N, 3)。
        tgt (np.ndarray): ターゲット点群 (M, 3)。
        R (np.ndarray): 回転行列 (3, 3)。
        t (np.ndarray): 平行移動ベクトル (3,)。

    Returns:
        float: RMSE 値。値が小さいほど整列精度が高い。

    Notes:
        - 最近傍探索には NumPy 実装の KDTreeNP を使用。
        - 点群サイズが異なる場合でも最近傍距離にもとづいて RMSE を計算する。

    """
    # 剛体変換を適用
    src_aligned = (src @ R.T) + t

    # KDTreeを構築
    tree = KDTreeNP(tgt)

    # 各ソース点の最近傍を検索
    _, dist = tree.query(src_aligned)  # n_jobs=-1 マルチコア使用

    # RMSEを計算
    return np.sqrt(np.mean(dist**2))
