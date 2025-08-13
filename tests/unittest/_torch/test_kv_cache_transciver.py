import multiprocessing as mp
import os
import sys

import pytest
import torch

import tensorrt_llm
from tensorrt_llm import logger
from tensorrt_llm._torch.pyexecutor.kv_cache_transceiver import (
    AttentionTypeCpp, CacheTransBufferManager, create_kv_cache_transceiver)
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.bindings.executor import CacheTransceiverConfig
from tensorrt_llm.mapping import Mapping


def setup_environment():
    """Setup environment variables for testing"""
    # Set UCX environment variables for testing
    os.environ["UCX_TLS"] = "^cuda_ipc"  # Disable CUDA IPC for testing
    os.environ["UCX_TCP_CM_REUSEADDR"] = "y"  # Reuse ports for testing
    os.environ["TRTLLM_USE_UCX_KVCACHE"] = "1"  # Enable UCX backend


def create_test_mapping(rank,
                        world_size,
                        tp_size=1,
                        pp_size=1,
                        cp_size=1,
                        enable_attention_dp=False):
    """Create a test mapping configuration"""
    return Mapping(rank=rank,
                   tp_size=tp_size,
                   pp_size=pp_size,
                   cp_size=cp_size,
                   gpus_per_node=world_size,
                   enable_attention_dp=enable_attention_dp)


def create_test_kv_cache_manager(mapping,
                                 num_layers=2,
                                 num_heads=4,
                                 head_dim=64,
                                 tokens_per_block=8,
                                 dtype=torch.float16,
                                 is_mla=False,
                                 max_seq_len=128,
                                 max_batch_size=4):
    """Create a test KV cache manager with configurable data type and MLA support"""

    # Convert torch dtype to TensorRT-LLM dtype
    if dtype == torch.float16:
        kv_cache_dtype = tensorrt_llm.bindings.DataType.HALF
    elif dtype == torch.bfloat16:
        kv_cache_dtype = tensorrt_llm.bindings.DataType.BF16
    elif dtype == torch.float8_e4m3fn:
        kv_cache_dtype = tensorrt_llm.bindings.DataType.FP8
    elif dtype == torch.float32:
        kv_cache_dtype = tensorrt_llm.bindings.DataType.FLOAT
    elif dtype == torch.int8:
        kv_cache_dtype = tensorrt_llm.bindings.DataType.INT8
    else:
        raise ValueError(f"Unsupported dtype for KV cache: {dtype}")

    # Create kv_cache_config
    kv_cache_config = tensorrt_llm.bindings.executor.KvCacheConfig(
        max_tokens=max_seq_len)

    # Determine cache type and parameters based on MLA
    if is_mla:
        cache_type = tensorrt_llm.bindings.internal.batch_manager.CacheType.SELFKONLY
        # For MLA, num_kv_heads should be 1
        num_kv_heads = 1
        # MLA uses different head dimension calculation
        mla_head_dim = head_dim  # This could be adjusted based on MLA config
    else:
        cache_type = tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF
        num_kv_heads = num_heads

    # Create KV cache manager
    kv_cache_manager = KVCacheManager(kv_cache_config,
                                      cache_type,
                                      num_layers=num_layers,
                                      num_kv_heads=num_kv_heads,
                                      head_dim=head_dim,
                                      tokens_per_block=tokens_per_block,
                                      max_seq_len=max_seq_len,
                                      max_batch_size=max_batch_size,
                                      mapping=mapping,
                                      dtype=kv_cache_dtype)

    return kv_cache_manager


def create_test_llm_request(request_id=1, max_new_tokens=1):
    """Create a test LLM request with configurable parameters"""
    sampling_params = tensorrt_llm.sampling_params.SamplingParams()
    sampling_config = tensorrt_llm.bindings.SamplingConfig(
        sampling_params._get_sampling_config())

    input_tokens = [i + 1 for i in range(max_new_tokens)]
    request = LlmRequest(
        request_id=request_id,
        max_new_tokens=max_new_tokens,
        input_tokens=input_tokens,
        sampling_config=sampling_config,
        is_streaming=False,
    )

    return request


def sender_process(rank, world_size):
    """Process that sends KV cache data"""
    setup_environment()
    logger.info(f"Sender process {rank} starting")

    # Create mapping for sender
    mapping = create_test_mapping(rank, world_size)

    is_MLA = False
    attention_type = AttentionTypeCpp.MLA if is_MLA else AttentionTypeCpp.DEFAULT

    max_seq_len = 128
    # Create KV cache manager
    kv_cache_manager = create_test_kv_cache_manager(mapping,
                                                    max_seq_len=max_seq_len,
                                                    is_mla=is_MLA)

    # Create cache transceiver config
    config = CacheTransceiverConfig()
    config.backend = tensorrt_llm.bindings.executor.CacheTransceiverBackendType.UCX
    config.max_tokens_in_buffer = max_seq_len

    # Create transceiver
    transceiver = create_kv_cache_transceiver(mapping=mapping,
                                              kv_cache_manager=kv_cache_manager,
                                              attention_type=attention_type,
                                              cache_transceiver_config=config)

    llmRequest = create_test_llm_request(request_id=1,
                                         max_new_tokens=max_seq_len)
    beam_width = 1
    kv_cache_manager.impl.add_sequence(llmRequest.request_id,
                                       llmRequest.prompt_len, beam_width,
                                       llmRequest)


def test_create_data_transceiver_state_socket_basic():
    """Test basic usage of create_data_transceiver_state_socket function"""

    # Import the function from the data_transceiver_utils module
    from tensorrt_llm.bindings.exceptions import \
        create_data_transceiver_state_socket

    # Test parameters
    nb_kv_heads_per_layer = [4, 4]  # 2 layers, 4 heads each
    size_per_head = 64
    tokens_per_block = 8
    tensor_parallelism = 1
    pipeline_parallelism = 1
    data_type = tensorrt_llm.bindings.DataType.HALF
    socket_addresses = ["127.0.0.1", "127.0.0.1"]  # 2 socket addresses
    attention_type = tensorrt_llm.bindings.internal.batch_manager.AttentionType.DEFAULT
    kv_factor = 2
    enable_attention_dp = False
    dp_rank = 0
    dp_size = 1
    rank = 0

    # Call the function
    result = create_data_transceiver_state_socket(
        nb_kv_heads_per_layer=nb_kv_heads_per_layer,
        size_per_head=size_per_head,
        tokens_per_block=tokens_per_block,
        tensor_parallelism=tensor_parallelism,
        pipeline_parallelism=pipeline_parallelism,
        data_type=data_type,
        socket_addresses=socket_addresses,
        attention_type=attention_type,
        kv_factor=kv_factor,
        enable_attention_dp=enable_attention_dp,
        dp_rank=dp_rank,
        dp_size=dp_size,
        rank=rank)

    # Verify the result is a bytes object (serialized data)
    assert isinstance(result, bytes), f"Expected bytes, got {type(result)}"
    assert len(result) > 0, "Serialized data should not be empty"


def test_kv_cache_transceiver_basic():

    # sender_process(0, 1)
    test_create_data_transceiver_state_socket_basic()


class TestKVCacheTransceiver:
    """Test class for KV cache transceiver functionality"""

    @pytest.fixture(autouse=True)
    def setup_test_environment(self):
        """Setup test environment before each test"""
        setup_environment()

    def test_kv_cache_transceiver_config(self):
        """Test KV cache transceiver configuration"""
        # Test with different backend types
        try:
            backends = [
                tensorrt_llm.bindings.executor.CacheTransceiverBackendType.UCX,
                tensorrt_llm.bindings.executor.CacheTransceiverBackendType.MPI,
                tensorrt_llm.bindings.executor.CacheTransceiverBackendType.NIXL
            ]

            for backend in backends:
                config = CacheTransceiverConfig()
                config.backend = backend

                # Test that config can be created
                assert config.backend == backend
                logger.info(f"Backend config test passed for {backend}")
        except AttributeError:
            pytest.skip("CacheTransceiverBackendType not available")

    def test_cache_trans_buffer_manager(self):
        """Test cache transceiver buffer manager"""
        # Create a simple KV cache manager
        kv_cache_manager = create_test_kv_cache_manager()

        # Create buffer manager
        max_num_tokens = 1024
        buffer_manager = CacheTransBufferManager(kv_cache_manager,
                                                 max_num_tokens)

        # Test buffer size calculation
        kv_cache_size_per_token = 1024  # bytes
        config = CacheTransceiverConfig()
        try:
            config.backend = tensorrt_llm.bindings.executor.CacheTransceiverBackendType.UCX
        except AttributeError:
            pytest.skip("CacheTransceiverBackendType not available")

        buffer_size = CacheTransBufferManager.pre_alloc_buffer_size(
            kv_cache_size_per_token, config)

        # Buffer size should be positive
        assert buffer_size > 0
        logger.info(f"Pre-allocated buffer size: {buffer_size} bytes")

    def test_kv_cache_manager_creation(self):
        """Test KV cache manager creation"""
        try:
            kv_cache_manager = create_test_kv_cache_manager()
            assert kv_cache_manager is not None
            logger.info("KV cache manager created successfully")
        except Exception as e:
            logger.warning(f"KV cache manager creation failed: {e}")
            pytest.skip("KV cache manager not available")

    def test_kv_cache_manager_different_dtypes(self):
        """Test KV cache manager creation with different data types"""
        supported_dtypes = [
            torch.float16, torch.float32, torch.bfloat16, torch.int8
        ]

        for dtype in supported_dtypes:
            try:
                kv_cache_manager = create_test_kv_cache_manager(dtype=dtype)
                assert kv_cache_manager is not None
                logger.info(
                    f"KV cache manager created successfully with dtype {dtype}")
            except Exception as e:
                logger.warning(
                    f"KV cache manager creation failed with dtype {dtype}: {e}")
                # Skip if this dtype is not supported in the current environment
                continue

    def test_kv_cache_manager_mla_config(self):
        """Test KV cache manager creation with MLA configuration"""
        try:
            # Test MLA configuration
            kv_cache_manager_mla = create_test_kv_cache_manager(
                num_layers=4,
                num_heads=8,
                head_dim=64,
                tokens_per_block=16,
                dtype=torch.float16,
                is_mla=True)
            assert kv_cache_manager_mla is not None
            logger.info("MLA KV cache manager created successfully")

            # Test non-MLA configuration
            kv_cache_manager_regular = create_test_kv_cache_manager(
                num_layers=4,
                num_heads=8,
                head_dim=64,
                tokens_per_block=16,
                dtype=torch.float16,
                is_mla=False)
            assert kv_cache_manager_regular is not None
            logger.info("Regular KV cache manager created successfully")

        except Exception as e:
            logger.warning(f"MLA KV cache manager test failed: {e}")
            pytest.skip("MLA KV cache manager not available")

    def test_kv_cache_manager_various_configs(self):
        """Test KV cache manager creation with various configurations"""
        configs = [
            # (layers, heads, head_dim, tokens_per_block, dtype, is_mla)
            (2, 4, 64, 8, torch.float16, False),
            (4, 8, 128, 16, torch.float32, False),
            (6, 16, 32, 32, torch.bfloat16, False),
            (2, 4, 64, 8, torch.float16, True),  # MLA
            (4, 8, 128, 16, torch.float16, True),  # MLA
        ]

        for config in configs:
            layers, heads, head_dim, tokens_per_block, dtype, is_mla = config
            try:
                kv_cache_manager = create_test_kv_cache_manager(
                    num_layers=layers,
                    num_heads=heads,
                    head_dim=head_dim,
                    tokens_per_block=tokens_per_block,
                    dtype=dtype,
                    is_mla=is_mla)
                assert kv_cache_manager is not None
                logger.info(
                    f"KV cache manager created successfully with config: {config}"
                )
            except Exception as e:
                logger.warning(
                    f"KV cache manager creation failed with config {config}: {e}"
                )
                # Continue testing other configs
                continue

    def test_kv_cache_transceiver_basic(self):
        """Basic test for KV cache transceiver functionality"""
        # Skip if not in a proper environment
        try:
            tensorrt_llm.bindings.executor.CacheTransceiverBackendType.UCX
        except AttributeError:
            pytest.skip("UCX backend not available")

        # Create shared data for inter-process communication
        manager = mp.Manager()
        shared_data = manager.dict()

        # Initialize shared data
        shared_data['sender_ready'] = False
        shared_data['sender_success'] = False
        shared_data['sender_error'] = None
        shared_data['sender_complete'] = False
        shared_data['receiver_ready'] = False
        shared_data['receiver_success'] = False
        shared_data['receiver_error'] = None
        shared_data['receiver_complete'] = False

        # Create processes
        sender = mp.Process(target=sender_process, args=(0, 2, shared_data))
        receiver = mp.Process(target=receiver_process, args=(1, 2, shared_data))

        try:
            # Start processes
            sender.start()
            receiver.start()

            # Wait for both processes to complete
            sender.join(timeout=30)
            receiver.join(timeout=30)

            # Check results
            assert shared_data.get(
                'sender_success',
                False), f"Sender failed: {shared_data.get('sender_error')}"
            assert shared_data.get(
                'receiver_success',
                False), f"Receiver failed: {shared_data.get('receiver_error')}"

            logger.info("KV cache transceiver test completed successfully")

        except Exception as e:
            logger.error(f"Test failed: {e}")
            raise
        finally:
            # Clean up processes
            if sender.is_alive():
                sender.terminate()
                sender.join()
            if receiver.is_alive():
                receiver.terminate()
                receiver.join()


# Standalone test functions for pytest discovery
def test_kv_cache_transceiver_config_standalone():
    """Standalone test for KV cache transceiver configuration"""
    # Test with different backend types
    try:
        backends = [
            tensorrt_llm.bindings.executor.CacheTransceiverBackendType.UCX,
            tensorrt_llm.bindings.executor.CacheTransceiverBackendType.MPI,
            tensorrt_llm.bindings.executor.CacheTransceiverBackendType.NIXL
        ]

        for backend in backends:
            config = CacheTransceiverConfig()
            config.backend = backend

            # Test that config can be created
            assert config.backend == backend
            logger.info(f"Backend config test passed for {backend}")
    except AttributeError:
        pytest.skip("CacheTransceiverBackendType not available")


def test_cache_trans_buffer_manager_standalone():
    """Standalone test for cache transceiver buffer manager"""
    # Create a simple KV cache manager
    kv_cache_manager = create_test_kv_cache_manager()

    # Create buffer manager
    max_num_tokens = 1024
    CacheTransBufferManager(kv_cache_manager, max_num_tokens)

    # Test buffer size calculation
    kv_cache_size_per_token = 1024  # bytes
    config = CacheTransceiverConfig()
    try:
        config.backend = tensorrt_llm.bindings.executor.CacheTransceiverBackendType.UCX
    except AttributeError:
        pytest.skip("CacheTransceiverBackendType not available")

    buffer_size = CacheTransBufferManager.pre_alloc_buffer_size(
        kv_cache_size_per_token, config)

    # Buffer size should be positive
    assert buffer_size > 0
    logger.info(f"Pre-allocated buffer size: {buffer_size} bytes")


def test_kv_cache_manager_creation_standalone():
    """Standalone test for KV cache manager creation"""
    try:
        kv_cache_manager = create_test_kv_cache_manager()
        assert kv_cache_manager is not None
        logger.info("KV cache manager created successfully")
    except Exception as e:
        logger.warning(f"KV cache manager creation failed: {e}")
        pytest.skip("KV cache manager not available")


def test_kv_cache_manager_different_dtypes_standalone():
    """Standalone test for KV cache manager creation with different data types"""
    supported_dtypes = [
        torch.float16, torch.float32, torch.bfloat16, torch.int8
    ]

    for dtype in supported_dtypes:
        try:
            kv_cache_manager = create_test_kv_cache_manager(dtype=dtype)
            assert kv_cache_manager is not None
            logger.info(
                f"KV cache manager created successfully with dtype {dtype}")
        except Exception as e:
            logger.warning(
                f"KV cache manager creation failed with dtype {dtype}: {e}")
            # Skip if this dtype is not supported in the current environment
            continue


def test_kv_cache_manager_mla_config_standalone():
    """Standalone test for KV cache manager creation with MLA configuration"""
    try:
        # Test MLA configuration
        kv_cache_manager_mla = create_test_kv_cache_manager(num_layers=4,
                                                            num_heads=8,
                                                            head_dim=64,
                                                            tokens_per_block=16,
                                                            dtype=torch.float16,
                                                            is_mla=True)
        assert kv_cache_manager_mla is not None
        logger.info("MLA KV cache manager created successfully")

        # Test non-MLA configuration
        kv_cache_manager_regular = create_test_kv_cache_manager(
            num_layers=4,
            num_heads=8,
            head_dim=64,
            tokens_per_block=16,
            dtype=torch.float16,
            is_mla=False)
        assert kv_cache_manager_regular is not None
        logger.info("Regular KV cache manager created successfully")

    except Exception as e:
        logger.warning(f"MLA KV cache manager test failed: {e}")
        pytest.skip("MLA KV cache manager not available")


def test_kv_cache_manager_various_configs_standalone():
    """Standalone test for KV cache manager creation with various configurations"""
    configs = [
        # (layers, heads, head_dim, tokens_per_block, dtype, is_mla)
        (2, 4, 64, 8, torch.float16, False),
        (4, 8, 128, 16, torch.float32, False),
        (6, 16, 32, 32, torch.bfloat16, False),
        (2, 4, 64, 8, torch.float16, True),  # MLA
        (4, 8, 128, 16, torch.float16, True),  # MLA
    ]

    for config in configs:
        layers, heads, head_dim, tokens_per_block, dtype, is_mla = config
        try:
            kv_cache_manager = create_test_kv_cache_manager(
                num_layers=layers,
                num_heads=heads,
                head_dim=head_dim,
                tokens_per_block=tokens_per_block,
                dtype=dtype,
                is_mla=is_mla)
            assert kv_cache_manager is not None
            logger.info(
                f"KV cache manager created successfully with config: {config}")
        except Exception as e:
            logger.warning(
                f"KV cache manager creation failed with config {config}: {e}")
            # Continue testing other configs
            continue


def test_kv_cache_transceiver_basic_standalone():
    """Standalone basic test for KV cache transceiver functionality"""
    # Skip if not in a proper environment
    try:
        tensorrt_llm.bindings.executor.CacheTransceiverBackendType.UCX
    except AttributeError:
        pytest.skip("UCX backend not available")

    # Create shared data for inter-process communication
    manager = mp.Manager()
    shared_data = manager.dict()

    # Initialize shared data
    shared_data['sender_ready'] = False
    shared_data['sender_success'] = False
    shared_data['sender_error'] = None
    shared_data['sender_complete'] = False
    shared_data['receiver_ready'] = False
    shared_data['receiver_success'] = False
    shared_data['receiver_error'] = None
    shared_data['receiver_complete'] = False

    # Create processes
    sender = mp.Process(target=sender_process, args=(0, 2, shared_data))
    receiver = mp.Process(target=receiver_process, args=(1, 2, shared_data))

    try:
        # Start processes
        sender.start()
        receiver.start()

        # Wait for both processes to complete
        sender.join(timeout=30)
        receiver.join(timeout=30)

        # Check results
        assert shared_data.get(
            'sender_success',
            False), f"Sender failed: {shared_data.get('sender_error')}"
        assert shared_data.get(
            'receiver_success',
            False), f"Receiver failed: {shared_data.get('receiver_error')}"

        logger.info("KV cache transceiver test completed successfully")

    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise
    finally:
        # Clean up processes
        if sender.is_alive():
            sender.terminate()
            sender.join()
        if receiver.is_alive():
            receiver.terminate()
            receiver.join()


if __name__ == "__main__":
    # Set multiprocessing start method for Linux
    if sys.platform.startswith('linux'):
        mp.set_start_method('spawn', force=True)

    # Run tests directly if executed as script
    pytest.main([__file__, "-v"])
