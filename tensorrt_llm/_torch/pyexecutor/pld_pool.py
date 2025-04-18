class PLDPool:  # Ngrams pool for Prompt-Lookup-Decoding

    def __init__(
        self,
        input_batch_size: int,
        prompt_lookup_num_tokens: int,
        max_matching_ngram_size: int,
        end_id: int,
        max_seq_len: list[int],
        is_keep_all: bool = True,
        is_use_oldest: bool = True,
    ):
        self.input_batch_size = input_batch_size
        self.plnt = prompt_lookup_num_tokens  # Shorter name
        self.mmns = max_matching_ngram_size  # Shorter name
        self.end_id = end_id
        self.max_seq_len = max_seq_len
        self.is_keep_all = is_keep_all
        self.is_use_oldest = is_use_oldest
        self.pool = [{} for _ in range(input_batch_size)]
        self.start_index = [0 for _ in range(input_batch_size)]

    # modified from `transformers/generation/candidate_generator.py`
    def get_draft_tokens(self, prefix: list[torch.Tensor],
                         batch_slot: list[int]):
        batch_size = len(prefix)
        prefix_len = [len(prefix[bi]) for bi in range(batch_size)]
        draft_tokens = []  # `logits` is useless yet
        for bi in range(batch_size):
            gbi = batch_slot[bi]  # Global index in the input batch
            chosen_ids = [self.end_id]
            # Skip search if prefix is length of `max_length - 1`
            if prefix_len[bi] >= self.max_seq_len[gbi] - 1:
                draft_tokens.append(chosen_ids)
                continue

            # Update pool
            sequence = prefix[bi][self.start_index[gbi]:] #.tolist()
            for size in range(min(self.mmns, prefix_len[bi] - 1), 0, -1):
                # Find each possible key-value combination, and use tuple for hash
                for l in range(len(sequence) - size):
                    r = min(l + size + self.plnt, len(sequence))
                    key = tuple(sequence[l:l + size])
                    value = tuple(sequence[l + size:r])
                    if key not in self.pool[gbi] or not self.is_keep_all or \
                        len(self.pool[gbi][key][0]) < self.plnt:
                        # Update the value if
                        # 1. the key does not exist
                        # 2. we only keep one value for each key
                        # 3. the length of the value saved before is less than `prompt_lookup_num_tokens`
                        self.pool[gbi][key] = OrderedSet((value, ))
                    elif value not in self.pool[gbi][key]:
                        # Extend the value if the key is already existed and we want to keep all of them
                        self.pool[gbi][key].add(value)

            # Find match
            for size in range(min(self.mmns, prefix_len[bi] - 1), 0, -1):
                pattern = tuple(prefix[bi][-size:])
                if pattern not in self.pool[gbi]:
                    continue
                if self.is_use_oldest:
                    # Always choose the oldest match, aligned with HF
                    chosen_ids = self.pool[gbi][pattern][0]
                else:
                    # Always choose the newest match
                    chosen_ids = self.pool[gbi][pattern][-1]
                break
            draft_tokens.append(chosen_ids)
            self.start_index[gbi] = max(
                0, prefix_len[bi] - (self.plnt + self.mmns - 1))

        return draft_tokens, None

