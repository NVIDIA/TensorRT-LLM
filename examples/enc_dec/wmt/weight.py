import torch
import math
import numpy

fairseq_wmt_ckpt_path = "/data/wmt"
fairseq_wmt_ckpt = "model.pt"

def generate_position_embedding(num_embeddings: int, embedding_dim: int):
    '''
    https://github.com/facebookresearch/fairseq/blob/main/fairseq/modules/sinusoidal_positional_embedding.py
    fairseq specific position embedding [sin, sin, ... cos, cos...]
    '''
    half_dim = embedding_dim // 2.0
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
    emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
    if embedding_dim % 2 == 1:
        # zero pad
        emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
    return emb.numpy()

def get_qkv_from_in_proj_weight(in_proj_weight):
    # in_proj_weight: (3*hidden_dim, hidden_dim) (3072, 1024)
    # qkv_weight : (hidden_dim, 3*hidden_dim)
    weight_q, weight_k, weight_v = numpy.vsplit(in_proj_weight, 3)
    qkv_weight = numpy.array([weight_q, weight_k, weight_v])
    qkv_weight = numpy.concatenate(qkv_weight)
    return qkv_weight

def get_qkv_from_qkv_proj(weight_q, weight_k, weight_v):
    qkv_weight = numpy.array([weight_q, weight_k, weight_v])
    qkv_weight = numpy.concatenate(qkv_weight)
    return qkv_weight


def load_from_pytorch_wmt(tensorrt_llm_mt, pytorch_ckpt_path, component, max_token_num=1024):
    '''
    loading wmt ckpt to trt_llm. 
    '''
    print(f"load weight for {component}")
    pytorch_ckpt = torch.load(pytorch_ckpt_path + "/" + fairseq_wmt_ckpt)
    pytorch_model = {key : value.numpy() for key, value in pytorch_ckpt['model'].items()}

    if component == "encoder":
        # encoder
        tensorrt_llm_mt.embedding.vocab_embedding.weight.value = pytorch_model['encoder.embed_tokens.weight']
        _, emb_dim = pytorch_model['encoder.embed_tokens.weight'].shape
        
        tensorrt_llm_mt.embedding.position_embedding.weight.value = generate_position_embedding(max_token_num, emb_dim)
        
        for i in range(pytorch_ckpt['args'].encoder_layers):

            tensorrt_llm_mt.encoder_layers[i].attention.qkv.weight.value = get_qkv_from_in_proj_weight(pytorch_model[f'encoder.layers.{i}.self_attn.in_proj_weight'])
            tensorrt_llm_mt.encoder_layers[i].attention.qkv.bias.value = pytorch_model[f'encoder.layers.{i}.self_attn.in_proj_bias']

            tensorrt_llm_mt.encoder_layers[i].attention.dense.weight.value = pytorch_model[f'encoder.layers.{i}.self_attn.out_proj.weight']
            tensorrt_llm_mt.encoder_layers[i].attention.dense.bias.value = pytorch_model[f'encoder.layers.{i}.self_attn.out_proj.bias']
        
            tensorrt_llm_mt.encoder_layers[i].attention_layernorm.weight.value = pytorch_model[f'encoder.layers.{i}.layer_norms.0.weight']
            tensorrt_llm_mt.encoder_layers[i].attention_layernorm.bias.value = pytorch_model[f'encoder.layers.{i}.layer_norms.0.bias']
            
            tensorrt_llm_mt.encoder_layers[i].mlp.fc.weight.value = pytorch_model[f'encoder.layers.{i}.fc1.weight']
            tensorrt_llm_mt.encoder_layers[i].mlp.fc.bias.value = pytorch_model[f'encoder.layers.{i}.fc1.bias']
            tensorrt_llm_mt.encoder_layers[i].mlp.proj.weight.value = pytorch_model[f'encoder.layers.{i}.fc2.weight']
            tensorrt_llm_mt.encoder_layers[i].mlp.proj.bias.value = pytorch_model[f'encoder.layers.{i}.fc2.bias']

            tensorrt_llm_mt.encoder_layers[i].mlp_layernorm.weight.value = pytorch_model[f'encoder.layers.{i}.layer_norms.1.weight']
            tensorrt_llm_mt.encoder_layers[i].mlp_layernorm.bias.value = pytorch_model[f'encoder.layers.{i}.layer_norms.1.bias']

    if component == "decoder":
        # decoder
        tensorrt_llm_mt.embedding.vocab_embedding.weight.value = pytorch_model['decoder.embed_tokens.weight']
        _, emb_dim = pytorch_model['decoder.embed_tokens.weight'].shape
        tensorrt_llm_mt.embedding.position_embedding.weight.value= generate_position_embedding(max_token_num, emb_dim)

        for i in range(pytorch_ckpt['args'].decoder_layers):

            tensorrt_llm_mt.decoder_layers[i].self_attention.qkv.weight.value = get_qkv_from_in_proj_weight(pytorch_model[f'decoder.layers.{i}.self_attn.in_proj_weight'])
            tensorrt_llm_mt.decoder_layers[i].self_attention.qkv.bias.value = pytorch_model[f'decoder.layers.{i}.self_attn.in_proj_bias']
            tensorrt_llm_mt.decoder_layers[i].self_attention.dense.weight.value = pytorch_model[f'decoder.layers.{i}.self_attn.out_proj.weight']
            tensorrt_llm_mt.decoder_layers[i].self_attention.dense.bias.value = pytorch_model[f'decoder.layers.{i}.self_attn.out_proj.bias']

            tensorrt_llm_mt.decoder_layers[i].self_attention_layernorm.weight.value = pytorch_model[f'decoder.layers.{i}.layer_norms.0.weight']
            tensorrt_llm_mt.decoder_layers[i].self_attention_layernorm.bias.value = pytorch_model[f'decoder.layers.{i}.layer_norms.0.bias']

            tensorrt_llm_mt.decoder_layers[i].cross_attention.qkv.weight.value = get_qkv_from_in_proj_weight(pytorch_model[f'decoder.layers.{i}.encoder_attn.in_proj_weight'])
            tensorrt_llm_mt.decoder_layers[i].cross_attention.qkv.bias.value = pytorch_model[f'decoder.layers.{i}.encoder_attn.in_proj_bias']
            tensorrt_llm_mt.decoder_layers[i].cross_attention.dense.weight.value = pytorch_model[f'decoder.layers.{i}.encoder_attn.out_proj.weight']
            tensorrt_llm_mt.decoder_layers[i].cross_attention.dense.bias.value = pytorch_model[f'decoder.layers.{i}.encoder_attn.out_proj.bias']

            tensorrt_llm_mt.decoder_layers[i].cross_attention_layernorm.weight.value = pytorch_model[f'decoder.layers.{i}.layer_norms.1.weight']
            tensorrt_llm_mt.decoder_layers[i].cross_attention_layernorm.bias.value = pytorch_model[f'decoder.layers.{i}.layer_norms.1.bias']

            tensorrt_llm_mt.decoder_layers[i].mlp.fc.weight.value = pytorch_model[f'decoder.layers.{i}.fc1.weight']
            tensorrt_llm_mt.decoder_layers[i].mlp.fc.bias.value = pytorch_model[f'decoder.layers.{i}.fc1.bias']
            tensorrt_llm_mt.decoder_layers[i].mlp.proj.weight.value = pytorch_model[f'decoder.layers.{i}.fc2.weight']
            tensorrt_llm_mt.decoder_layers[i].mlp.proj.bias.value = pytorch_model[f'decoder.layers.{i}.fc2.bias']

            tensorrt_llm_mt.decoder_layers[i].mlp_layernorm.weight.value = pytorch_model[f'decoder.layers.{i}.layer_norms.2.weight']
            tensorrt_llm_mt.decoder_layers[i].mlp_layernorm.bias.value = pytorch_model[f'decoder.layers.{i}.layer_norms.2.bias']

            # share_decoder_input_output_embed=True, output_proj = embed_tokens.transpose()
            # https://github.pie.apple.com/machine-translation/fairseq/blob/5465e33c004c38e94af30e8dd200483a66069c54/fairseq/models/transformer.py#L718-L724
            tensorrt_llm_mt.lm_head.weight.value = pytorch_model['encoder.embed_tokens.weight']
