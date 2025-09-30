from tensorrt_llm.bindings.executor import FinishReason


class FinishedState:
    # State flags
    FINISHED_EOS = 1 << 0
    FINISHED_STOP_WORDS = 1 << 1
    FINISHED_MAX_LENGTH = 1 << 2
    FINISHED = FINISHED_EOS | FINISHED_STOP_WORDS | FINISHED_MAX_LENGTH
    SKIP_DECODING = 1 << 3

    def __init__(self, state=0):
        self._state = state

    @classmethod
    def empty(cls):
        return cls(0)

    @classmethod
    def finished(cls):
        return cls(cls.FINISHED)

    @classmethod
    def skip_decoding(cls):
        return cls(cls.SKIP_DECODING)

    @classmethod
    def finished_eos(cls):
        return cls(cls.FINISHED_EOS)

    @classmethod
    def finished_max_length(cls):
        return cls(cls.FINISHED_MAX_LENGTH)

    @classmethod
    def finished_stop_words(cls):
        return cls(cls.FINISHED_STOP_WORDS)

    def set_finished_eos(self):
        self._state |= self.FINISHED_EOS

    @property
    def is_finished_eos(self):
        return self._any_bit_set(self.FINISHED_EOS)

    def set_finished_stop_words(self):
        self._state |= self.FINISHED_STOP_WORDS

    @property
    def is_finished_stop_words(self):
        return self._any_bit_set(self.FINISHED_STOP_WORDS)

    def set_finished_max_length(self):
        self._state |= self.FINISHED_MAX_LENGTH

    @property
    def is_finished_max_length(self):
        return self._any_bit_set(self.FINISHED_MAX_LENGTH)

    def set_finished(self):
        self._state |= self.FINISHED

    @property
    def is_finished(self):
        return self._any_bit_set(self.FINISHED)

    def set_skip_decoding(self):
        self._state |= self.SKIP_DECODING

    @property
    def is_skip_decoding(self):
        return self._any_bit_set(self.SKIP_DECODING)

    def to_finish_reason(self):
        if self.is_finished_eos:
            return FinishReason.END_ID
        if self.is_finished_stop_words:
            return FinishReason.STOP_WORDS
        if self.is_finished_max_length:
            return FinishReason.LENGTH
        return FinishReason.NOT_FINISHED

    def to_underlying(self):
        return self._state

    def _any_bit_set(self, bits):
        return (self._state & bits) != 0
