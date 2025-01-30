import torch


class BaseImageProcessor:

    def __init__(self, tokenizer, device='auto'):
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self, **kwargs):
        return self.tokenizer(**kwargs)

    def preprocess_function(self, examples):
        raise NotImplementedError(
            "Each image processor must implement its own preprocess method")

    def collate_function(self, examples):
        raise NotImplementedError(
            "Each image processor must implement its own colloate method")


# A light Encapsulation for Huggingface MllamaImageProcessor
class MllamaImageProcessor(BaseImageProcessor):

    def preprocess_function(self, examples):
        # Prepare prompts in a generic chat format
        if 'question' in examples:
            question = examples['question']
        else:
            question = "Describe this image."

        if examples['image'] is not None:
            if self.tokenizer.chat_template is not None:
                prompt = self.tokenizer.apply_chat_template(
                    [{
                        "role":
                        "user",
                        "content": [{
                            "type": "image"
                        }, {
                            "type": "text",
                            "text": question
                        }],
                    }],
                    add_generation_prompt=True,
                )
            else:
                prompt = f"<|image|><|begin_of_text|>{question}"

            # Process images using the processor's image processor
            values = self.tokenizer(text=prompt,
                                    images=examples['image'],
                                    return_tensors="pt").to(self.device)
        else:
            if self.tokenizer.chat_template is not None:
                prompt = self.tokenizer.apply_chat_template(
                    [{
                        "role": "user",
                        "content": [{
                            "type": "text",
                            "text": question
                        }],
                    }],
                    add_generation_prompt=True,
                )
            else:
                prompt = question

            values = self.tokenizer(text=prompt,
                                    images=None,
                                    return_tensors="pt").to(self.device)

            values['pixel_values'] = None
            values['aspect_ratio_ids'] = None
            values['aspect_ratio_mask'] = None
            values['cross_attention_mask'] = None

        return values

    # Define a collate function to process images during data loading
    def collate_function(self, batch):
        batch[0]['input_ids'] = torch.LongTensor(batch[0]['input_ids']).to(
            self.device)
        batch[0]['attention_mask'] = torch.LongTensor(
            batch[0]['attention_mask']).to(self.device)

        if batch[0]['pixel_values'] is not None:
            batch[0]['pixel_values'] = torch.Tensor(
                batch[0]['pixel_values']).to(self.device)
            batch[0]['aspect_ratio_ids'] = torch.LongTensor(
                batch[0]['aspect_ratio_ids']).to(self.device)
            batch[0]['aspect_ratio_mask'] = torch.LongTensor(
                batch[0]['aspect_ratio_mask']).to(self.device)
            batch[0]['cross_attention_mask'] = torch.LongTensor(
                batch[0]['cross_attention_mask']).to(self.device)

        return batch[0]
