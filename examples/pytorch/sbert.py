from sentence_transformers import SentenceTransformer
from transformers import BertForSequenceClassification, BertConfig
import torch


my_model_path = './model/my-custom-sbert'

def load_sentence_transformer():
    # Load the model
    model = SentenceTransformer(my_model_path)
    print(model)
    # SentenceTransformer(
    # (0): Transformer({'max_seq_length': 512, 'do_lower_case': False}) with Transformer model: BertModel
    # (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True,
    #               'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False,
    #               'pooling_mode_lasttoken': False, 'include_prompt': True})
    # (2): Dense({'in_features': 768, 'out_features': 192, 'bias': False, 'activation_function': 'torch.nn.modules.linear.Identity'})
    return model


# 1. Load SentenceTransformer model
sentence_model = load_sentence_transformer()

# 2-1. Get the underlying BERT model
bert_model = sentence_model._first_module().auto_model
# 2-2. Extract the Dense layer weights
dense_layer = sentence_model._last_module()  # Gets the final Dense layer
dense_weight = dense_layer.linear.weight  # Shape [192, 768]

# 3. Create config for classification model
config = BertConfig.from_pretrained(
    my_model_path,
    num_labels=192,  # Set number of classes
    hidden_dropout_prob=bert_model.config.hidden_dropout_prob,
    attention_probs_dropout_prob=bert_model.config.attention_probs_dropout_prob
)

# 4. Create new classification model
classification_model = BertForSequenceClassification(config)

# 5. Copy weights
classification_model.bert.load_state_dict(bert_model.state_dict(), strict=False)
classification_model.classifier = torch.nn.Linear(
    in_features=768,
    out_features=192,
    bias=True
)
classification_model.classifier.weight.data = dense_weight.clone()
# sentence-transformer's Dense layer has no bias,
# but trtllm weight loading method enforces a bias,
# so setting all to zero here to be computationally equivalent, though less efficient
classification_model.classifier.bias.data.zero_()
print(classification_model)

# 7. Save if needed
classification_model.save_pretrained('./converted-classification-model')