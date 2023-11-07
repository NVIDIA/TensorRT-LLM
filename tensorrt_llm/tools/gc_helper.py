from ..module import Module
from ..network import Network


def cleanup(network: Network, model: Module):
    # TODO: A quick fix for the memory leak caused by Parameter.
    # Remove this method once the issue fixed in a proper way.
    for _, param in model.named_parameters():
        # param._value captures the numpy array so that gc can't collect
        # those buffers.
        param._value = None
    network._registered_ndarrays = None
