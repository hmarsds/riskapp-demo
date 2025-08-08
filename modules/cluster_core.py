# modules/cluster_core.py
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch

def spearman_ward_clusters(
    Y: pd.DataFrame,
    max_k: int = 10,
    min_size: int = 2
):
    """
    Compute Spearman correlation, Ward linkage and a stable cluster labelling.

    Returns
    -------
    labels : pd.Series
        Integer cluster labels (1..k), indexed by Y.columns (original order).
    link : np.ndarray
        SciPy linkage matrix (Ward).
    codep : pd.DataFrame
        Spearman correlation matrix (Y.columns x Y.columns).
    leaves : list[int]
        Dendrogram leaf order (positions) for visual reordering.
    """
    # Spearman codependence & distance
    codep = Y.corr(method="spearman")
    dist_mat = np.sqrt(0.5 * (1 - codep))              # Mantegna distance
    condensed = sch.distance.squareform(dist_mat.values)

    # Ward linkage (deterministic if inputs fixed)
    link = sch.linkage(condensed, method="ward", optimal_ordering=True)

    # Auto-k via two-difference gap + min cluster size
    heights = np.sort(link[:, 2])[-(max_k + 1):] if link.shape[0] else np.array([0.0])
    diffs = np.diff(heights) if heights.size > 1 else np.array([0.0])
    ddiffs = (diffs[1:] - diffs[:-1]) if diffs.size > 1 else np.array([0.0])
    best = int(np.argmax(ddiffs)) + 1 if ddiffs.size else 1

    n_leaves = link.shape[0] + 1 if link.shape[0] else len(codep.columns)
    k0 = max(2, n_leaves - (best + 1))

    k_final = 2
    for k in range(k0, 1, -1):
        labs_try = sch.fcluster(link, t=k, criterion="maxclust")
        if np.bincount(labs_try)[1:].min() >= min_size:
            k_final = k
            break

    labels_arr = sch.fcluster(link, t=k_final, criterion="maxclust")
    labels = pd.Series(labels_arr, index=codep.columns, name="cluster")

    leaves = sch.dendrogram(link, no_plot=True)["leaves"] if link.shape[0] else list(range(len(codep.columns)))
    return labels, link, codep, leaves