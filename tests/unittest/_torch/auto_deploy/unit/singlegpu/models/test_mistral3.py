from tensorrt_llm._torch.auto_deploy.models import mistral3


def test_get_extra_inputs_includes_image_sizes():
    factory = mistral3.Mistral3VLM(model="test-model")
    extra_inputs = factory.get_extra_inputs()

    pixel_values = extra_inputs["pixel_values"]
    image_sizes = extra_inputs["image_sizes"]

    pixel_values_dynamic_shape = pixel_values[1]()
    image_sizes_dynamic_shape = image_sizes[1]()

    # Unfortunately, direct object comparisons do not work.
    assert pixel_values_dynamic_shape[0].__dict__ == image_sizes_dynamic_shape[0].__dict__
