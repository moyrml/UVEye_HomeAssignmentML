import plotly_express as px
import plotly.graph_objs as go


def plot_loss(epoch_losses, y_title='MSE Loss'):
    fig = px.line(x=list(range(len(epoch_losses))), y=epoch_losses, title='Training loss')
    fig.layout.xaxis.title = 'Epoch'
    fig.layout.yaxis.title = y_title
    return fig


if __name__ == '__main__':
    epoch_losses = [0.1, 0.01, 0.0001]
    fig = plot_loss(epoch_losses)
