import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def show_two_clouds(
    P1: np.ndarray,
    P2: np.ndarray,
    C1: np.ndarray | None = None,
    C2: np.ndarray | None = None,
    s: float = 0.5,
    title: str | None = None,
) -> None:
    """2 つの点群を 3D 散布図として可視化する。

    点群 P1（整列後など）と点群 P2（ターゲット）を
    3D 表示し、ICP や PCA 粗合わせの結果を目視で確認するための関数。

    Args:
        P1 (np.ndarray): ソースまたは整列後の点群 (N, 3)。
        P2 (np.ndarray): ターゲット点群 (M, 3)。
        C1 (np.ndarray): P1の点群に対応する色（M, 3)。色情報がない時はblueにする
        C2 (np.ndarray): P2の点群に対応する色（M, 3)。色情報がない時はredにする
        s (float): 散布図の点サイズ。デフォルトは 0.5。
        title (str | None): 図のタイトル（任意）。

    Returns:
        None: 可視化のみ行い、値は返さない。

    Notes:
        - matplotlib の 3D Axes を用いて描画。
        - 点群の重なり具合やアラインメントの良否を確認できる。

    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect([1, 1, 1])

    if C1 is None:
        C1 = "blue"
    if C2 is None:
        C2 = "red"

    # ソース（レジストレーション後）
    ax.scatter(
        P1[:, 0], P1[:, 1], P1[:, 2], s=s, alpha=0.6, c=C1, label="Source aligned"
    )
    # ターゲット
    ax.scatter(P2[:, 0], P2[:, 1], P2[:, 2], s=s, alpha=0.6, c=C2, label="Target")
    # タイトル設定（任意）
    if title is not None:
        ax.set_title(title)

    plt.axis("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()
