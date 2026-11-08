def ppl(logits, output_ids):
    """
    Calculate per-token perplexity.
    """
    nlls = -logits.log_softmax(dim=-1)
    ppls = nlls.gather(-1, output_ids.long().unsqueeze(-1))
    return ppls.mean().exp().item()
