import pandas as pd
import pickle as pkl
from pathlib import Path
from sklearn.decomposition import PCA
from argparse import ArgumentParser
import json

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


def cluster_embeddings(algo_type, n_clusters, embeddings, class_size_prior=None):
    clusteringAlgo = get_clustering_algo_from_name(algo_type)
    cluster = clusteringAlgo(n_clusters=n_clusters, n_init='auto')
    cluster.fit(embeddings)

    cluster.class_size_prior = None
    if class_size_prior is not None:
        frequency = pd.Series(cluster.labels_).value_counts(normalize=True, ascending=False)
        cluster.class_size_prior = {cls: class_size_prior[i] for i, cls in enumerate(frequency.index)}

    return cluster


if __name__ == '__main__':
    set_random_seeds()

    parser = ArgumentParser()
    parser.add_argument('--model_path', default=None)
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
    parser.add_argument(
        '--class_size_prior',
        nargs='+',
        default=['screw', 'pill', 'metal_nut', 'capsule'],
        help='Give clusters names according to the size of the cluster, biggest to smallest. '
             'This prior derives straight from the instructions.'
    )
    args = parser.parse_args()

    if args.model_path is None:
        last_run_file = Path('train_ae_last_output_folder.json')
        assert last_run_file.exists(), f'No model_path specified and cannot find train_ae_last_output_folder.json'

        with open(last_run_file, 'r') as f:
            model_path = Path(json.load(f)['output_dir'])
    else:
        model_path = Path(args.model_path)
    print(f'Using model directory {model_path}')

    embeddings_file = args.embeddings_file
    if args.embeddings_file is None:
        embeddings_file = model_path / 'latent_vectors' / args.set_type / 'names_labels_embeddings.pkl'
    assert embeddings_file.exists(), f'Cannot find latent representations of images. Location given: {embeddings_file}'

    with open(embeddings_file, 'rb') as f:
        contents = pkl.load(f)

    pca = None
    df = pd.DataFrame(contents['embeddings'])
    if args.pca_dim != -1:
        df, pca = reduce_dim(contents['embeddings'], args.pca_dim)

    cluster = cluster_embeddings(args.clustering_algo, args.n_clusters, df, args.class_size_prior)
    if args.pca_dim == 2:
        df['label'] = contents['labels']
        df['label'] = df['label'].astype(str)
        fig = plot_2d_embeddings_scatter(df, cluster)
        fig.write_html(model_path / 'latent_vectors' / 'train' / '2d_pca_embeddings.html')

    with open(model_path / 'embedding_model.pkl', 'wb') as f:
        pkl.dump(
            dict(
                reduce_dim=pca,
                cluster=cluster
            ),
            f
        )
    print('done')
