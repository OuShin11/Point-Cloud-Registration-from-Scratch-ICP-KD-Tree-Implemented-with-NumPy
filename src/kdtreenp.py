import numpy as np


# ---------- KD-Tree 节点 ----------
class KDNode:
    """KD-Tree の単一ノードを表すクラス。

    各ノードは以下を保持する：
    - 子ノード（left, right）
    - 分割軸（axis）
    - 分割値（thr）
    - 対応するデータ（perm 配列内のインデックス範囲）
    - AABB（bbox_min, bbox_max）
    - 葉ノードであるかどうか（is_leaf）

    Attributes:
        idx_slice (tuple[int, int] | None):
            perm 配列内でこのノードが担当する範囲（start, end）。
            葉ノードのみで使用。
        axis (int | None):
            分割に使用する軸（0, 1, 2）。内部ノードでのみ設定。
        thr (float | None):
            axis における分割しきい値。
        left (KDNode | None):
            左の子ノード。
        right (KDNode | None):
            右の子ノード。
        bbox_min (np.ndarray | None):
            AABB の最小値（各軸）。
        bbox_max (np.ndarray | None):
            AABB の最大値（各軸）。
        is_leaf (bool):
            葉ノードかどうか。
    """

    __slots__ = (
        "idx_slice",
        "axis",
        "thr",
        "left",
        "right",
        "bbox_min",
        "bbox_max",
        "is_leaf",
    )

    def __init__(self):
        self.idx_slice: tuple[int, int] | None = None
        self.axis: int | None = None
        self.thr: float | None = None
        self.left: "KDNode" | None = None
        self.right: "KDNode" | None = None
        self.bbox_min: np.ndarray | None = None
        self.bbox_max: np.ndarray | None = None
        self.is_leaf: bool = False


# ---------- 构建 KD-Tree ----------
class KDTreeNP:
    """NumPy ベースの KD-Tree 実装。

    点群検索（最近傍探索）を効率的に行うための KD-Tree を NumPy のみで
    構築するクラス。主に ICP の最近傍探索に利用される。

    Args:
        points (np.ndarray): 点群 (N, k)。float の 2 次元配列を想定。
        leaf_size (int): 葉ノードあたりの最大点数。少ないほど探索が高速。
        axis_rule (str): 分割軸の決定方法。
            - "cyclic": 深さに応じて軸を循環させる。
            - "spread": AABB の広がりが最も大きい軸を選択。
            - "var": 方差が最も大きい軸を選択（デフォルト）。

    Attributes:
        P (np.ndarray): 元の点群（目標点群）データ。
        N (int): 点数。
        K (int): 次元（通常は 3）。
        perm (np.ndarray): データの並べ替えインデックス。
        root (KDNode): KD-Tree の根ノード。
    """

    def __init__(self, points: np.ndarray, leaf_size: int = 32, axis_rule: str = "var"):
        """
        points: (N, k) float array
        leaf_size: 叶子容量（小批量暴力时的上限）
        axis_rule: 选轴策略
            - "cyclic": depth % k 轮流
            - "spread": 选 bbox 跨度最大的维
            - "var":    选方差最大的维（默认，较稳）
        """
        P = np.asarray(points, dtype=np.float64)
        assert P.ndim == 2
        self.P = P
        self.N, self.K = P.shape
        self.leaf_size = int(leaf_size)
        self.axis_rule = axis_rule

        # perm: P の並び替えindexを保持（ノード内部ではスライスのみを保持）
        self.perm = np.arange(self.N, dtype=np.int64)

        # node listを事前に割り当て
        self.root = self._build(0, self.N, depth=0)

    # --- 軸を探す ---
    def _choose_axis(self, start: int, end: int, depth: int) -> int:
        """分割軸を選択する。

        指定されたデータ領域（perm[start:end]）に対して、
        axis_rule に基づいて軸を決定する。

        Args:
            start (int): perm の開始インデックス。
            end (int): perm の終了インデックス。
            depth (int): 木の深さ。

        Returns:
            int: 分割軸（0〜K-1）。
        """
        if self.axis_rule == "cyclic":
            return depth % self.K
        # 現在の部分集合の AABB / 分散から「広がり」を推定
        ids = self.perm[start:end]
        block = self.P[ids]
        if self.axis_rule == "spread":
            span = block.max(axis=0) - block.min(axis=0)
            return int(np.argmax(span))
        # var（デフォルト）：分散が最大の軸を選択
        v = block.var(axis=0)
        return int(np.argmax(v))

    # --- 再帰構築 ---
    def _build(self, start: int, end: int, depth: int) -> KDNode:
        """KD-Tree を再帰的に構築する内部メソッド。

        Args:
            start (int): perm における開始インデックス。
            end (int): perm における終了インデックス。
            depth (int): 現在のノードの深さ。

        Returns:
            KDNode: 構築されたノード。
        """
        node = KDNode()  
        ids = self.perm[start:end]
        block = self.P[ids]

        # AABB（拡張・可視化・剪定用）
        node.bbox_min = block.min(axis=0)
        node.bbox_max = block.max(axis=0)

        count = end - start
        if count <= self.leaf_size:
            node.is_leaf = True
            node.idx_slice = (start, end)
            return node

        axis = self._choose_axis(start, end, depth)
        node.axis = axis

        # np.partition を用いて中央値を取得（完全ソート不要）
        mid = start + count // 2
        # perm[start:end] の範囲を axis 方向で中央値分割
        part_keys = self.P[self.perm[start:end], axis]
        # この区間内で第 k 小の要素（局所インデックス）を取得
        kth = np.argpartition(part_keys, count // 2)[count // 2]
        # 局所インデックスを全体インデックスへ変換
        kth_global = start + kth

        # 中央値を mid へ移動：スワップ後、axis に基づく 3 分割で左右を保証
        self.perm[mid], self.perm[kth_global] = self.perm[kth_global], self.perm[mid]
        # 中央値の座標をしきい値として設定
        node.thr = self.P[self.perm[mid], axis]

        # 左右を安定的に分割（<=thr を左、>thr を右）して決定性を確保
        left_mask = self.P[self.perm[start:end], axis] < node.thr
        eq_mask = self.P[self.perm[start:end], axis] == node.thr
        # tie-break：等しい場合は元のインデックスが小さい方を左に寄せる
        eq_ids = self.perm[start:end][eq_mask]
        lt_ids = self.perm[start:end][left_mask]
        gt_ids = self.perm[start:end][~left_mask & ~eq_mask]

        # しきい値と等しい点は元インデックスでソートし、1 つを中央値として扱う
        if len(eq_ids) > 1:
            order = np.argsort(eq_ids)
            eq_ids = eq_ids[order]

        # 新しい順序を構築：
        # 左 (lt) + 中央値 + 残りの等値要素 (右側の最左端として) + 右 (gt)
        median_id = self.perm[mid]
        eq_rest = eq_ids[eq_ids != median_id]

        new_order = np.concatenate([lt_ids, [median_id], eq_rest, gt_ids])
        self.perm[start:end] = new_order

        # 左右の区間
        # 左区間の終了位置：start + len(lt_ids)
        left_end = start + len(lt_ids)
        # 右区間の開始位置：start + len(lt_ids) + 1（中央値をスキップ）
        right_start = left_end + 1

        # 左右の再帰構築
        node.left = (
            self._build(start, left_end, depth + 1) if left_end > start else None
        )
        node.right = (
            self._build(right_start, end, depth + 1) if right_start < end else None
        )

        return node

    # --- 葉ノード内での暴力最近傍探索 ---
    @staticmethod
    def _nn_leaf(
        point: np.ndarray, block: np.ndarray, ids: np.ndarray
    ) -> tuple[float, int]:
        """葉ノード内での暴力最近傍探索。

        Args:
            point (np.ndarray): クエリ点 (3,)。
            block (np.ndarray): 候補点群 (M, 3)。
            ids (np.ndarray): 元のインデックス (M,)。

        Returns:
            tuple[float, int]:
                - best_dist2 (float): 最小の距離二乗。
                - best_id (int): 対応する元のインデックス。
        """
        # (best_dist2, best_id) を返す
        diff = block - point
        d2 = np.einsum("ij,ij->i", diff, diff)
        minpos = np.argmin(d2)
        # tie-break：距離が同じ場合は元のインデックスが小さい方を選択
        # 等しい距離の位置をすべて取得
        same = np.where(d2 == d2[minpos])[0]
        if len(same) > 1:
            best = ids[same[np.argmin(ids[same])]]
            best_d2 = d2[same[0]]
        else:
            best = ids[minpos]
            best_d2 = d2[minpos]
        return best_d2, int(best)

    # --- 単一点検索 ---
    def query_one(self, point: np.ndarray) -> tuple[int, float]:
        """単一点の最近傍探索を行う。

        手書きスタックを用いて深さ優先で KD-Tree を辿り、
        最近傍点のインデックスと距離を返す。

        Args:
            point (np.ndarray): クエリ点 (3,)。

        Returns:
            tuple[int, float]:
                - best_id (int): 最近傍点の元のインデックス。
                - best_dist (float): 最近傍点までの距離。
        """
        p = np.asarray(point, dtype=np.float64)
        best_d2 = np.inf
        best_id = -1

        # 手書きスタックで再帰を回避
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node is None:
                continue

            if node.is_leaf:
                s, e = node.idx_slice
                ids = self.perm[s:e]
                block = self.P[ids]
                d2, iid = self._nn_leaf(p, block, ids)
                # best の更新（tie-break を含む）
                if (d2 < best_d2) or (d2 == best_d2 and iid < best_id):
                    best_d2, best_id = d2, iid
                continue

            # 探索優先度の高い側にまず進む
            axis = node.axis
            go_left = p[axis] <= node.thr
            near, far = (node.left, node.right) if go_left else (node.right, node.left)
            if far is not None:
                # 分割平面までの最小距離（二乗）を計算（剪定判定用）
                plane_d2 = (p[axis] - node.thr) ** 2
                # 遠側は「条件を満たす場合のみ」後で探索する（遅延アクセス）
                if plane_d2 < best_d2:
                    stack.append(far)
            stack.append(near)

        return best_id, np.sqrt(best_d2)

    # --- 複数点検索（ICP 用：A → B の最近傍） ---
    def query(self, Q: np.ndarray, batch: int = 4096) -> tuple[np.ndarray, np.ndarray]:
        """複数点の最近傍探索をまとめて行う。

        Args:
            Q (np.ndarray): クエリ点群 (N, 3)。
            batch (int): バッチサイズ。大きいほど高速だがメモリ使用量が増える。

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - out_idx (np.ndarray): 各点に対する最近傍点のインデックス (N,)。
                - out_dist (np.ndarray): 各点の最近傍距離 (N,)。
        """
        Q = np.asarray(Q, dtype=np.float64)
        n = len(Q)
        out_idx = np.empty(n, dtype=np.int64)
        out_dist = np.empty(n, dtype=np.float64)
        for i in range(0, n, batch):
            qi = Q[i : i + batch]
            for t, p in enumerate(qi):
                idx, d = self.query_one(p)
                out_idx[i + t] = idx
                out_dist[i + t] = d
        return out_idx, out_dist
