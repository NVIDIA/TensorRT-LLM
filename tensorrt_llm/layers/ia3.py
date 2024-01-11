

class Ia3RuntimeParams(object):

    def __init__(self, ia3_weights_pointers):
        self.ia3_weights_pointers = ia3_weights_pointers
    
    
class Ia3Params(object):

    def __init__(self, ia3_weights_pointers=None): # : List[dict[Tensor]]
        self.ia3_weights_pointers = ia3_weights_pointers

    def get_layer_params(self, layer_idx: int):
        return Ia3Params(ia3_weights_pointers=[self.ia3_weights_pointers[layer_idx]])

    def get_runtime_params(self, layer_idx: int, ia3_module: str):
        if f"{ia3_module}_ia3_weights_pointers" in self.ia3_weights_pointers[layer_idx]:
            return Ia3RuntimeParams(
                ia3_weights_pointers=[
                    self.ia3_weights_pointers[layer_idx]
                    [f"{ia3_module}_ia3_weights_pointers"]
                ]
            )
        else:
            return None
