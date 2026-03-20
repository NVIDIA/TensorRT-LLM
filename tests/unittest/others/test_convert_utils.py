# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for tensorrt_llm.models.convert_utils.load_state_dict.

NVBug: https://nvbugs/5866619
"""

import tempfile
import unittest
import unittest.mock
from pathlib import Path

import torch

from tensorrt_llm.models.convert_utils import load_state_dict


class TestLoadStateDict(unittest.TestCase):
    """Tests for load_state_dict() covering the PEFT safetensors fallback.

    NVBug 5866619: load_state_dict() used safe_open() (low-level API) which
    raised ValueError for PEFT-saved safetensors files containing
    UntypedStorage objects.  The fix added a try/except fallback to
    safetensors.torch.load_file().
    """

    def test_peft_safetensors_fallback(self):
        """load_state_dict() falls back to load_file() when get_tensor() fails.

        NVBug 5866619: PEFT-saved safetensors files trigger a ValueError inside
        the safe_open() context manager at f.get_tensor() — not at safe_open()
        construction time.  The actual traceback was:

            File "tensorrt_llm/models/convert_utils.py", line 184, in load_state_dict
                tensor = f.get_tensor(name)
            ValueError: could not determine the shape of object type
                        'torch.storage.UntypedStorage'

        The fallback to safetensors.torch.load_file() must recover and return
        the correct tensors, optionally with dtype casting.
        """
        import safetensors.torch

        tensor_names = ["lora_A.weight", "lora_B.weight"]
        expected = {
            "lora_A.weight": torch.randn(8, 16, dtype=torch.float32),
            "lora_B.weight": torch.randn(16, 8, dtype=torch.float32),
        }

        peft_error = ValueError(
            "could not determine the shape of object type 'torch.storage.UntypedStorage'"
        )

        # Build a mock context manager that:
        #   - succeeds on safe_open() call (returns a context manager)
        #   - succeeds on __enter__ (returns a mock file handle)
        #   - has working keys() (returns tensor names)
        #   - raises ValueError on get_tensor() — the real failure site
        mock_handle = unittest.mock.MagicMock()
        mock_handle.keys.return_value = tensor_names
        mock_handle.get_tensor.side_effect = peft_error

        mock_ctx = unittest.mock.MagicMock()
        mock_ctx.__enter__ = unittest.mock.Mock(return_value=mock_handle)
        mock_ctx.__exit__ = unittest.mock.Mock(return_value=False)

        mock_safe_open = unittest.mock.Mock(return_value=mock_ctx)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "adapter_model.safetensors"
            safetensors.torch.save_file(expected, str(path))

            with unittest.mock.patch("safetensors.safe_open", mock_safe_open):
                result = load_state_dict(path, dtype=torch.float16)

        # safe_open() must have been entered (not just called) before failing
        mock_ctx.__enter__.assert_called_once()
        mock_handle.get_tensor.assert_called()

        # The fallback must deliver the same keys and shapes, with dtype cast
        self.assertEqual(
            set(result.keys()),
            set(expected.keys()),
            "Fallback path must return the same keys as the saved file",
        )
        for key in expected:
            self.assertIn(key, result, f"Key {key!r} missing after fallback load")
            self.assertEqual(
                result[key].shape,
                expected[key].shape,
                f"Shape mismatch after fallback for key {key!r}",
            )
            self.assertEqual(
                result[key].dtype,
                torch.float16,
                f"dtype cast not applied in fallback path for key {key!r}",
            )


if __name__ == "__main__":
    unittest.main()
