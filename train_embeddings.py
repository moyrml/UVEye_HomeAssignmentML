import pandas as pd
import pickle as pkl
from pathlib import Path
from sklearn.decomposition import PCA
from argparse import ArgumentParser

from utils.clustring_algo_mapping import get_clustering_algo_from_name
from utils.plot_utils import plot_2d_embeddings_scatter
from utils.reproducibility import set_random_seeds


def reduce_dim(embeddings, target_dim=None, pca=None):
    if pca is None:
        norm_params = dict(mean=embeddings.mean(0), std=embeddings.std(0))
        embeddings -= norm_params['mean']
        embeddings /= norm_params['std']

        pca = PCA(n_components=target_dim)
        embeddings = pca.fit_transform(embeddings)
        pca.norm_params = norm_params
    else:
        norm_params = pca.norm_params
        embeddings -= norm_params['mean']
        embeddings /= norm_params['std']
        embeddings = pca.transform(embeddings)
        target_dim = pca.n_components
    df = pd.DataFrame(embeddings, columns=[f'PC_{i+1}' for i in range(target_dim)])

    return df, pca


def cluster_embeddings(algo_type, n_clusters, embeddings):
    clusteringAlgo = get_clustering_algo_from_name(algo_type)
    cluster = clusteringAlgo(n_clusters=n_clusters, n_init='auto')
    cluster.fit(embeddings)

    return cluster


if __name__ == '__main__':
    set_random_seeds()

    parser = ArgumentParser()
    parser.add_argument('--master_dir', default='outputs/December_30_2023_10_59AM')
    parser.add_argument(
        '--pca_dim',
        type=int,
        default=2,
        help='output dimension for PCA dimension reduction. -1 for no PCA'
    )
    parser.add_argument(
        '--embeddings_file',
        default=None,
        help='path to embeddings. Do not specify if you want to use the default of '
             'model_dir/latent_vectors/$set_type/names_labels_embeddings.pkl'
    )
    parser.add_argument('--clustering_algo', default='KMeans')
    parser.add_argument('--n_clusters', type=int, default=4)
    parser.add_argument('--set_type', default='train')
    args = parser.parse_args()

    master_dir = Path(args.master_dir)

    embeddings_file = args.embeddings_file
    if args.embeddings_file is None:
        embeddings_file = master_dir / 'latent_vectors' / args.set_type / 'names_labels_embeddings.pkl'
    assert embeddings_file.exists(), f'Cannot find latent representations of images. Location given: {embeddings_file}'

    with open(embeddings_file, 'rb') as f:
        contents = pkl.load(f)

    pca = None
    df = pd.DataFrame(contents['embeddings'])
    if args.pca_dim != -1:
        df, pca = reduce_dim(contents['embeddings'], args.pca_dim)

    cluster = cluster_embeddings(args.clustering_algo, args.n_clusters, df)
    if args.pca_dim == 2:
        df['label'] = contents['labels']
        df['label'] = df['label'].astype(str)
        fig = plot_2d_embeddings_scatter(df, cluster)
        fig.write_html(master_dir / 'latent_vectors' / 'train' / '2d_pca_embeddings.html')

    with open(master_dir / 'embedding_model.pkl', 'wb') as f:
        pkl.dump(
            dict(
                reduce_dim=pca,
                cluster=cluster
            ),
            f
        )
    print('done')
