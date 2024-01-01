from torch import nn


def get_activation_func_from_name(func_name):
    activation_functions = dict(
        ReLU=nn.ReLU,
        PReLU=nn.PReLU,
        GELU=nn.GELU,
        GLU=nn.GLU,
        LeakyReLU=nn.LeakyReLU
    )

    return activation_functions[func_name]
