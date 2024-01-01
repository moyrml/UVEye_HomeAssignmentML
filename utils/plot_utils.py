import plotly_express as px
import plotly.graph_objs as go


def plot_loss(epoch_losses, y_title='MSE Loss'):
    fig = px.line(x=list(range(len(epoch_losses))), y=epoch_losses, title='Training loss')
    fig.layout.xaxis.title = 'Epoch'
    fig.layout.yaxis.title = y_title
    return fig


def plot_2d_embeddings_scatter(df, cluster=None, label_col='label'):
    """

    :param cluster: None or an `sklearn.cluter` algorithm. In the latter case, the plot will contain the centroids
    :param df: pandas.DataFrame with columns [PC_1, PC_2, label]
    :return: plotly.graph_objs.Figure
    """
    if label_col is not None:
        df[label_col] = df[label_col].astype(str)
    fig = px.scatter(df, x='PC_1', y='PC_2', color=label_col)
    if cluster is not None:
        fig.add_trace(
            go.Scatter(
                x=cluster.cluster_centers_[:, 0],
                y=cluster.cluster_centers_[:, 1],
                mode='markers',
                marker=dict(
                    symbol='x',
                    size=16,
                    color='black'
                ),
                name='Clustring centroids'
            )
        )
    return fig


if __name__ == '__main__':
    epoch_losses = [0.1, 0.01, 0.0001]
    fig = plot_loss(epoch_losses)
