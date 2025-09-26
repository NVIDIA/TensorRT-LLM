import torch
import torch.cuda.nvtx as nvtx

from tensorrt_llm.logger import logger


class LayerwiseNvtxMarker(object):
    """ This module contains all the code needed to enable forward hooks in a pytorch network.

    To register the hooks for a given network, the user needs to instantiate a LayerwiseNvtxMarker object.
    Then call the register_hooks method.

    Example:

        my_layerwise_nvtx_marker = LayerwiseNvtxMarker()
        my_layerwise_nvtx_marker.register_hooks(my_network_model)
    """

    def __init__(self):
        """ Initialize module variables

        Args:
            None:

        Returns:
            None:

        Raises:
            None:
        """
        super().__init__()
        self.module_to_name_map = {}
        self.out_tensor_to_name_stack = {}
        self.iteration = 0

    @staticmethod
    def _print_tensor(tensor_obj, prefix, tensor_list=[]):
        """ Descends iterators that contains Tensors and prints the Tensor

        Recursive function that descends iterator type arguments until
        it finds a Tensor object.

        Args:
            tensor_obj: Could be a Tensor or an iterator type that contains Tensors
            prefix: String name to assign to the Tensor

        Returns:
            None:

        Raises:
            None:
        """
        tensor_dims = []
        if isinstance(tensor_obj, list) or isinstance(tensor_obj, tuple):
            for ten in tensor_obj:
                tensor_list = LayerwiseNvtxMarker._print_tensor(
                    ten, prefix, tensor_list)
        elif isinstance(tensor_obj, torch.Tensor):
            hex(id(tensor_obj))
            tensor_dims = list(tensor_obj.size())
            tensor_list.append(tensor_dims)
        return tensor_list

    def _module_fwd_hook(self, module_obj, in_tensor, out_tensor):
        """ Callback function that ends the NVTX marker

        Records the module name and tensor information
        Called after the module executes the forward method.

        Args:
            module_obj: Pointer to the module object
            in_tensor: Input tensor or list of tensors
            out_tensor: Output tensor of the resulting forward operator

        Returns:
            None:

        Raises:
            None:
        """
        nvtx.range_pop()
        module_name = self.module_to_name_map[module_obj]

        logger.debug(f"FWD hook module {module_name}")
        if module_name == '\'top\'':
            self.iteration = self.iteration + 1
            logger.debug(f"Completed {self.iteration} iterations")

        return

    def _module_fwd_pre_hook(self, module_obj, in_tensor):
        """ Creates an NVTX marker with the module name in it.

        This function is called before the module executes

        Args:
            module_obj: Module object data structure - used to get unique module name
            in_tensor: Input tensor data structure

        Returns:
            None

        Raises:
            None
        """

        marker_dict = {}
        module_name = self.module_to_name_map[module_obj]
        module_params = module_obj.named_parameters(recurse=False)
        logger.debug(f"FWD Pre hook module:{module_name}")
        marker_dict['Module'] = module_name
        marker_dict['TrainableParams'] = {}
        ## Get trainable parameters like weights and bias
        for (param_name, param_obj) in module_params:
            marker_dict['TrainableParams'][param_name] = list(param_obj.size())
            logger.debug(f"Param {param_name} value {list(param_obj.size())}")

        in_tensor_list = LayerwiseNvtxMarker._print_tensor(in_tensor,
                                                           "Input",
                                                           tensor_list=[])
        if in_tensor_list:
            marker_dict['Inputs'] = in_tensor_list
            logger.debug("Input Tensor List-> {in_tensor_list}")

        nvtx.range_push("{}".format(marker_dict))

        return

    def register_hooks(self, network_model, module_prefix="top"):
        """ User level function that activates all the hooks

        The user needs to call this method from the network source code
        The code descends all the modules in the network and registers their
        respective hooks.

        Args:
            network_model: Model object for the network
            module_prefix: (default: top)

        Returns:
            None

        Raises:
            Exception if a module instance is reused
        """
        for name, module in network_model.named_modules(prefix=module_prefix):
            logger.debug(f"Module Name:{name} addr:{hex(id(module))}")
            module.register_forward_pre_hook(self._module_fwd_pre_hook)
            module.register_forward_hook(self._module_fwd_hook)
            if module not in self.module_to_name_map:
                self.module_to_name_map[module] = name
            else:
                raise Exception(
                    "Module instance {} is not unique ".format(module))
        return
