import os
import sys
from time import time

import tensorrt as trt

iModelID = int(
    sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else -1

onnxFileList = [
    'onnx/visual_encoder/visual_encoder.onnx', 'onnx/Qformer/Qformer.onnx'
]

planFileList = [
    'plan/visual_encoder/visual_encoder_fp16.plan',
    'plan/Qformer/Qformer_fp16.plan'
]

os.system('mkdir -p ./plan/visual_encoder')
os.system('mkdir -p ./plan/Qformer')

logger = trt.Logger(trt.Logger.ERROR)


def build(iPart, minBS=1, optBS=2, maxBS=4):
    onnxFile = onnxFileList[iPart]
    planFile = planFileList[iPart]

    builder = trt.Builder(logger)

    network_creation_flags = 0
    if "EXPLICIT_BATCH" in trt.NetworkDefinitionCreationFlag.__members__.keys():
        network_creation_flags = 1 << int(
            trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    network = builder.create_network(network_creation_flags)
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)

    parser = trt.OnnxParser(network, logger)

    with open(onnxFile, 'rb') as model:
        if not parser.parse(model.read(), "/".join(onnxFile.split("/"))):
            print("Failed parsing %s" % onnxFile)
            for error in range(parser.num_errors):
                print(parser.get_error(error))
        print("Succeeded parsing %s" % onnxFile)

    nBS = -1
    nMinBS = minBS
    nOptBS = optBS
    nMaxBS = maxBS

    if iPart == 0:
        inputT = network.get_input(0)
        inputT.shape = [nBS, 3, 224, 224]
        profile.set_shape(inputT.name, [nMinBS, 3, 224, 224],
                          [nOptBS, 3, 224, 224], [nMaxBS, 3, 224, 224])
    elif iPart == 1:
        inputT = network.get_input(0)
        inputT.shape = [nBS, 32, 768]
        profile.set_shape(inputT.name, [nMinBS, 32, 768], [nOptBS, 32, 768],
                          [nMaxBS, 32, 768])
        inputT = network.get_input(1)
        inputT.shape = [nBS, 257, 1408]
        profile.set_shape(inputT.name, [nMinBS, 257, 1408], [nOptBS, 257, 1408],
                          [nMaxBS, 257, 1408])
        inputT = network.get_input(2)
        inputT.shape = [nBS, 257]
        profile.set_shape(inputT.name, [nMinBS, 257], [nOptBS, 257],
                          [nMaxBS, 257])
    else:
        raise RuntimeError("iPart should be either 0 (ViT) or 1 (Qformer)")

    config.add_optimization_profile(profile)

    t0 = time()
    engineString = builder.build_serialized_network(network, config)
    t1 = time()
    if engineString == None:
        print("Failed building %s" % planFile)
    else:
        print("Succeeded building %s in %d s" % (planFile, t1 - t0))

    with open(planFile, 'wb') as f:
        f.write(engineString)


if __name__ == "__main__":
    if iModelID != 0 and iModelID != 1:
        print("Error model number, should be in [0, 1]")
        exit()

    build(iModelID, minBS=1, optBS=2, maxBS=4)
