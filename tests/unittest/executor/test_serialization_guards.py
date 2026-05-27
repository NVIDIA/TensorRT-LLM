from pathlib import Path


def test_deserialize_tensor_checks_serialized_byte_count():
    source = Path("cpp/tensorrt_llm/executor/serialization.cpp").read_text()
    function = source.split("Tensor Serialization::deserializeTensor", 1)[1]
    function = function.split("void Serialization::serialize", 1)[0]

    assert "sizeInBytes == tensor.getSizeInBytes()" in function
    assert function.count("checkTensorSize(") == 4
