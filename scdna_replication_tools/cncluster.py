import umap
import hdbscan
import logging
import pandas as pd
import numpy as np
import sklearn.cluster
import scipy.spatial


def umap_hdbscan_cluster(
        cn,
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
    ):
    """ Cluster using umap and hdbscan.
    Args:
        cn: data frame columns as cell ids, rows as segments
    Returns:
        data frame with columns:
            cluster_id
            cell_id
            umap1
            umap2
    """
    embedding = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=42,
        metric='euclidean',
    ).fit_transform(cn.fillna(0).values.T)

    clusters = hdbscan.HDBSCAN(
        min_samples=10,
        min_cluster_size=30,
    ).fit_predict(embedding)

    df = pd.DataFrame({
        'cell_id': cn.columns, 'cluster_id': clusters,
        'umap1': embedding[:, 0], 'umap2': embedding[:, 1]
    })
    df = df[['cell_id', 'cluster_id', 'umap1', 'umap2']]
    df = df.dropna()

    return df


def compute_bic(kmeans, X):
    """ Computes the BIC metric for a given k means clustering
    Args:
        kmeans: a fitted kmeans clustering object
        X: data for which to calculate bic
    
    Returns:
        float: bic
    
    Reference: https://stats.stackexchange.com/questions/90769/using-bic-to-estimate-the-number-of-k-in-kmeans
    """
    centers = [kmeans.cluster_centers_]
    labels  = kmeans.labels_
    n_clusters = kmeans.n_clusters
    cluster_sizes = np.bincount(labels)
    N, d = X.shape

    # Compute variance for all clusters
    cl_var = (1.0 / (N - n_clusters) / d) * sum([sum(scipy.spatial.distance.cdist(X[np.where(labels == i)], [centers[0][i]], 
             'euclidean')**2) for i in range(n_clusters)])

    const_term = 0.5 * n_clusters * np.log(N) * (d+1)

    bic = np.sum([cluster_sizes[i] * np.log(cluster_sizes[i]) -
               cluster_sizes[i] * np.log(N) -
             ((cluster_sizes[i] * d) / 2) * np.log(2*np.pi*cl_var) -
             ((cluster_sizes[i] - 1) * d/ 2) for i in range(n_clusters)]) - const_term

    return bic


def kmeans_cluster(
        cn,
        min_k=2,
        max_k=100,
    ):
    """ Cluster using kmeans and bic.
    """

    X = cn.T.values
    ks = range(min_k, max_k + 1)

    logging.info(f'trying with max k={max_k}')

    kmeans = []
    bics = []
    for k in ks:
        logging.info(f'trying with k={k}')
        model = sklearn.cluster.KMeans(n_clusters=k, init="k-means++").fit(X)
        bic = compute_bic(model, X)
        kmeans.append(model)
        bics.append(bic)

    opt_k = np.array(bics).argmax()
    logging.info(f'selected k={opt_k}')

    model = kmeans[opt_k]

    # embedding = umap.UMAP(
    #     n_neighbors=15,
    #     min_dist=0.1,
    #     n_components=2,
    #     random_state=42,
    #     metric='euclidean',
    # ).fit_transform(cn.fillna(0).values.T)

    clusters = pd.DataFrame({
        'cell_id': cn.columns, 'cluster_id': model.labels_,
        # 'umap1': embedding[:, 0], 'umap2': embedding[:, 1]
    })

    return clusters
