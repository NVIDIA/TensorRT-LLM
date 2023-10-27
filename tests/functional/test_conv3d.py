import unittest

import numpy as np
import torch
from polygraphy.backend.trt import EngineFromNetwork, TrtRunner

import tensorrt_llm
from tensorrt_llm import Tensor

class TestFunctional(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('error')

    def test_conv3d(self):
        # test data
        dtype = 'float32'
        x_data = torch.randn(8, 4, 5, 5, 5)
        weight_data = torch.randn(8, 4, 3, 3, 3)
        padding = (1, 1, 1)
        # construct trt network
        builder = tensorrt_llm.Builder()
        net = builder.create_network()
        with tensorrt_llm.net_guard(net):
            network = tensorrt_llm.default_trtnet()
            x = Tensor(name='x',
                       shape=x_data.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))

            weight = tensorrt_llm.constant(weight_data.numpy())

            output = tensorrt_llm.functional.conv3d(x, weight,
                                                    padding=padding).trt_tensor
            output.name = 'output'
            network.mark_output(output)

        # trt run
        build_engine = EngineFromNetwork((builder.trt_builder, net.trt_network))
        with TrtRunner(build_engine) as runner:
            outputs = runner.infer(feed_dict={
                'x': x_data.numpy(),
            })

        # pytorch run
        ref = torch.nn.functional.conv3d(x_data, weight_data, padding=padding)

        # compare diff
        np.testing.assert_allclose(ref.cpu().numpy(),
                                   outputs['output'],
                                   atol=1e-5)

if __name__ == "__main__":
    unittest.main()
