'''
Activation and init wrapper
Created June, 2017
Author: xiaodl@microsoft.com
'''


def linear(x):
    return x


def activation(func_a):
    """Activation function wrapper
    """
    return eval(func_a)


def init_wrapper(init='xavier_uniform'):
    return eval(init)
