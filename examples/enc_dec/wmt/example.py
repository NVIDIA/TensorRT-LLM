import torch

# Load an En-Fr Transformer model trained on WMT'14 data :
en2fr = torch.hub.load('pytorch/fairseq', 'transformer.wmt14.en-fr', tokenizer='moses', bpe='subword_nmt')

# Use the GPU (optional):
en2fr.cuda()

# Translate with beam search:
fr = en2fr.translate('Hello world!', beam=5)
assert fr == 'Bonjour à tous !'

# Manually tokenize:
en_toks = en2fr.tokenize('Hello world!')
assert en_toks == 'Hello world !'

# Manually apply BPE:
en_bpe = en2fr.apply_bpe(en_toks)
assert en_bpe == 'H@@ ello world !'

# Manually binarize:
en_bin = en2fr.binarize(en_bpe)
assert en_bin.tolist() == [329, 14044, 682, 812, 2]

# Generate five translations with top-k sampling:
fr_bin = en2fr.generate(en_bin, beam=5, sampling=True, sampling_topk=20)
assert len(fr_bin) == 5

# Convert one of the samples to a string and detokenize
fr_sample = fr_bin[0]['tokens']
fr_bpe = en2fr.string(fr_sample)
fr_toks = en2fr.remove_bpe(fr_bpe)
fr = en2fr.detokenize(fr_toks)
assert fr == en2fr.decode(fr_sample)