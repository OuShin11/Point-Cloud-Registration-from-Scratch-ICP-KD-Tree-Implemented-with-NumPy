import numpy as np
from plyfile import PlyData


# ファイルの読み込み
def read_ply(path: str) -> tuple[np.ndarray, np.ndarray]:
    """.ply点群データを読み込む。

    与えられたテキストファイルから、3 次元座標 (X, Y, Z) を読み込み、
    NumPy 配列として返す。

    Args:
        path (str): 読み込むテキストファイルのパス。

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - points (np.ndarray): 点群座標 (N, 3)。float32。

    Raises:
        OSError: 指定されたファイルが存在しない、または読み込めない場合。
        ValueError: データの列数が不足している場合。
    """
    plydata = PlyData.read(path)
    vertex = plydata['vertex']
    points = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=-1)
    return points


# ダウンサンプリング
def voxel_downsample_ply(
    points: np.ndarray, voxel_size: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    """ボクセルグリッドによる点群のダウンサンプリング（NumPy 実装）。

    指定したボクセルサイズで空間を格子状に区切り、各ボクセルにつき
    1 点（最初に出現した点）のみを残すことで、点群を減らす。
    点群 (points) と対応する色情報 (colors) を同じインデックスで間引く。

    Args:
        points (np.ndarray): 点群座標 (N, 3)。float32 または float64。
        voxel_size (float): ボクセルの一辺の長さ。デフォルトは 0.1。

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - points_ds (np.ndarray): ダウンサンプリング後の点群 (M, 3)。

    Raises:
        ValueError: 入力の形状が (N, 3) でない場合。

    Notes:
        - 本実装は各ボクセルの「最初に現れた点」を採用する簡易版であり、
          平均値による代表点生成（centroid）ではない。
        - `np.unique` の return_index=True を利用し、最初のインデックスを取得している。

    """
    voxel = np.floor(points / voxel_size).astype(np.int64)
    _, idx = np.unique(voxel, axis=0, return_index=True)
    return points[idx]


# 中心 + スケール合わせ
def align_center_scale(
    X: np.ndarray,
    Y: np.ndarray,
    method: str = "rms",
    pct: float = 0.95,
    eps: float = 1e-12,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """点群の中心合わせとスケール正規化（回転は行わない）。

    点群 X と Y をそれぞれ中心化し、選択したスケール推定方法
    （RMS 半径・バウンディングボックス・パーセンタイル半径）に基づいて
    X を Y にスケール合わせする。主に ICP 前の初期正規化として利用する。

    Args:
        X (np.ndarray): ソース点群 (N, 3)。
        Y (np.ndarray): ターゲット点群 (M, 3)。
        method (str):
            スケール推定方法。以下から選択：
            - 'rms': RMS 半径によるスケール推定。
            - 'bbox': AABB（バウンディングボックス）の対角長。
            - 'percentile': 距離ノルムの分位点（pct）を用いたロバスト推定。
        pct (float): 'percentile' を選んだ場合に使用する分位点（0〜1）。
        eps (float): ゼロ割防止用の小さな値。

    Returns:
        tuple[np.ndarray, float, np.ndarray, np.ndarray]:
            - X_aligned (np.ndarray): スケール合わせ＋中心合わせ後の X。
            - s (float): 推定されたスケール係数。
            - cX (np.ndarray): X の元の中心。
            - cY (np.ndarray): Y の元の中心。

    Raises:
        ValueError: method が不正な値の場合。

    Notes:
        - 'percentile' は外れ値や部分的な重なりがある場合により安定する。
        - 回転成分は推定しないため、主に前処理として利用する。

    """
    cX = X.mean(axis=0)
    cY = Y.mean(axis=0)
    Xc = X - cX
    Yc = Y - cY

    if method == "rms":
        # RMS 半径でスケールを推定
        rX = np.sqrt((Xc**2).sum(axis=1).mean())
        rY = np.sqrt((Yc**2).sum(axis=1).mean())
    elif method == "bbox":
        # bboxの対角長をスケールとして使用
        rX = np.linalg.norm(X.max(axis=0) - X.min(axis=0))
        rY = np.linalg.norm(Y.max(axis=0) - Y.min(axis=0))
    elif method == "percentile":
        # 分位点半径を使用
        dX = np.sqrt((Xc**2).sum(axis=1))
        dY = np.sqrt((Yc**2).sum(axis=1))
        rX = np.quantile(dX, pct)
        rY = np.quantile(dY, pct)
    else:
        raise ValueError("method must be 'rms' | 'bbox' | 'percentile'")

    s = rY / max(rX, eps)
    X_aligned = s * Xc + cY
    return X_aligned, s, cX, cY
