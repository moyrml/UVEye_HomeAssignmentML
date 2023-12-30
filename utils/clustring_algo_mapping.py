from sklearn import cluster


def get_clustering_algo_from_name(name):
    algorithms = dict(
        KMeans=cluster.KMeans,
        DBSCAN=cluster.DBSCAN
    )

    return algorithms[name]