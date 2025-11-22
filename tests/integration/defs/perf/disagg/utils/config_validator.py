"""Configuration Validator for Test Configs.

Validates TestConfig objects at test execution time (not at config loading time).
This ensures that validation failures only affect individual test cases.
"""

import os
from typing import Optional

from utils.common import EnvManager, extract_config_fields
from utils.config_loader import TestConfig
from utils.logger import logger


class ConfigValidator:
    """Configuration validator for test configs."""

    @staticmethod
    def validate_test_config(test_config: TestConfig) -> None:
        """Validate test configuration.

        This method is called at the beginning of each test case.
        If validation fails, it raises an exception that will cause only
        the current test to fail (not all tests).

        Args:
            test_config: TestConfig object to validate

        Raises:
            ValueError: If configuration is invalid
            FileNotFoundError: If required files/directories don't exist
            AssertionError: If assertion-based validation fails
        """
        logger.info("Validating test configuration...")
        extracted_config = extract_config_fields(test_config.config_data)

        ConfigValidator._validate_gen_max_tokens(extracted_config)
        logger.info("Validate generation maximum tokens succeeded!")
        ConfigValidator._validate_streaming_true(extracted_config)
        logger.info("Validate streaming is true succeeded!")
        ConfigValidator._validate_ctx_and_gen_max_seq_length(extracted_config)
        logger.info("Validate context and generation maximum sequence length succeeded!")
        

    @staticmethod
    def _validate_gen_max_tokens(extracted_config: dict) -> None:
        """Validate generation maximum tokens.

        Args:
            extracted_config: Extracted configuration fields

        Raises:
            ValueError: If generation maximum tokens is invalid
        """
        gen_max_tokens = extracted_config["gen_max_tokens"]
        gen_max_batch_size = extracted_config["gen_max_batch_size"]
        mtp_size = extracted_config["mtp_size"]
        if mtp_size > 0:
            assert gen_max_tokens == (gen_max_batch_size * (mtp_size + 1)), \
                "config error: gen_max_tokens != gen_max_batch_size * (mtp_size + 1)"
    
    @staticmethod
    def _validate_streaming_true(extracted_config: dict) -> None:
        """Validate streaming is true.

        Args:
            extracted_config: Extracted configuration fields

        Raises:
            ValueError: If streaming is not true
        """
        streaming = extracted_config["streaming"]
        assert streaming, "config error: streaming is not true"
    
    @staticmethod
    def _validate_ctx_and_gen_max_seq_length(extracted_config: dict) -> None:
        """Validate context and generation maximum sequence length.

        Args:
            extracted_config: Extracted configuration fields

        Raises:
            ValueError: If context and generation maximum sequence length is invalid
        """
        isl = extracted_config["isl"]
        osl = extracted_config["osl"]
        ctx_max_seq_len = extracted_config["ctx_max_seq_len"]
        gen_max_seq_len = extracted_config["gen_max_seq_len"]
        assert ctx_max_seq_len > (isl + osl), "config error: ctx_max_seq_len <= (isl + osl)" 
        assert gen_max_seq_len > (isl + osl), "config error: gen_max_seq_len <= (isl + osl)"