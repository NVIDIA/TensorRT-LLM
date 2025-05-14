from ordered_set import OrderedSet


class PLDPool:  # Ngrams pool for Prompt-Lookup-Decoding
    """
    This class maintains the pattern-matches pairs for NGram drafter.

    For example, one of the existed pairs could be: ["I","love"] -> [["apple", "because", "it", "is"], ["banana", "and"]]

    Here we call ["I","love"] as `pattern`, and [["apple", "because", "it", "is"], ["banana", "and"]] as `matches`.

    `pattern` is a list of token_ids. The pool will provide corresponding draft tokens from the matches if the pattern appears at the tail of the sentence during generation.

    `matches` is a list of candidate draft token_ids attaching to a pattern.

    Arguments:
        prompt_lookup_num_tokens: int
            The length maximum of tokens for draft (can be understood as length maximum of output draft tokens)

        max_matching_ngram_size: int
            The length maximum of tokens for searching (can be understood as length maximum of input tokens for search)

        is_keep_all (bool) = True
            Whether to save all candidate pattern-matches pairs

        is_use_oldest: bool = True
            Whether to provide the oldest match when pattern is hit

        is_public_pool: bool = True
            Whether to use a common pool for all requests

    Members:
        pool: dict[tuple[int], OrderedSet[int]] or dict[int, dict[tuple[int], OrderedSet[int]]]
            If is_public_pool == True, it maps from patterns to matches
            If is_public_pool == False, it maps from request ID to the request-specific pool

        start_index: dict[int, int]
            It maps from request ID to the index of the prompt to update the pool in the next step
    """

    def __init__(
        self,
        prompt_lookup_num_tokens: int,
        max_matching_ngram_size: int,
        is_keep_all: bool = True,
        is_use_oldest: bool = True,
        is_public_pool: bool = True,
    ):
        self.prompt_lookup_num_tokens = prompt_lookup_num_tokens
        self.max_matching_ngram_size = max_matching_ngram_size
        self.is_keep_all = is_keep_all
        self.is_use_oldest = is_use_oldest
        self.is_public_pool = is_public_pool
        self.pool = {}
        self.start_index = {}

    def print_line(self, local_map, indentation=0):  # For debug
        for pattern, matches in local_map.items():
            output = " " * indentation + str(pattern) + "->"
            for match in matches:
                output += str(match) + ", "
            logger.debug(output)

    def print_pool(self):  # For debug
        if self.is_public_pool:
            logger.debug(f"Using public pool, size = {len(self.pool)}")
            self.print_line(self.pool)
        else:
            logger.debug(f"Using private pool")
            for request_id, request_map in self.pool.items():
                logger.debug(f"Request {request_id}, size={len(request_map)}")
                self.print_line(request_map, 4)

    def remove_pool(self, request_id):
        if self.is_public_pool:
            return  # TODO: need to have a strategy to swap out the pairs
        if request_id in self.pool:
            self.pool.pop(request_id)
            self.start_index.pop(request_id)
        else:
            logger.warning(f"Request {request_id} is not in NGram Pool.")

    def get_draft_tokens(
        self,
        prefix: list[int],
        request_id: int,
        end_id: int,
        max_sequence_length: int,
    ):
        prefix_len = len(prefix)
        # Skip search if prefix is length of `max_length - 1`
        if prefix_len >= max_sequence_length - 1:
            return [end_id]

        if request_id not in self.start_index:  # A new request
            self.start_index[request_id] = 0
            if not self.is_public_pool:
                self.pool[request_id] = {}

        pool = (self.pool if self.is_public_pool else self.pool[request_id])

        # Update pool
        sequence = prefix[self.start_index[request_id]:]
        for size in range(min(self.max_matching_ngram_size, prefix_len - 1), 0,
                          -1):
            # Find each possible pattern-match combination, and use tuple for hash
            for l in range(len(sequence) - size):
                r = min(l + size + self.prompt_lookup_num_tokens, len(sequence))
                pattern = tuple(sequence[l:l + size])
                new_match = tuple(sequence[l + size:r])
                if pattern not in pool or \
                    (not self.is_keep_all and len(match) > pool[pattern][0]):
                    # Replace the match if
                    # 1. the pattern does not exist in the pool
                    # 2. only one match is kept, and the new match is longer (MRU)
                    pool[pattern] = OrderedSet((new_match, ))
                elif new_match not in pool[pattern]:
                    # Update the matches if the pattern is already existed:
                    # TODO: need a strategy to maintain the short candidates, now we just remove them
                    # Drop all existed matches with small length
                    for match in pool[pattern]:
                        if len(match) < len(new_match):
                            pool[pattern].remove(match)
                    pool[pattern].add(new_match)

        # Find match
        draft_tokens = [end_id]
        for size in range(min(self.max_matching_ngram_size, prefix_len - 1), 0,
                          -1):
            pattern = tuple(prefix[-size:])
            if pattern not in pool:
                continue
            draft_tokens = list(pool[pattern][0 if self.is_use_oldest else -1])
            break
        self.start_index[request_id] = max(
            0, prefix_len -
            (self.prompt_lookup_num_tokens + self.max_matching_ngram_size - 1))

        return draft_tokens
