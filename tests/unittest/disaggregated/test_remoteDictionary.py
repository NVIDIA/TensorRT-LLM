import unittest

import etcd3

from tensorrt_llm.serve.metadata_server import EtcdDictionary


class TestEtcdDictionary(unittest.TestCase):

    def setUp(self):
        # Setup etcd connection parameters
        self.host = "localhost"
        self.port = 2379

        # Create a clean etcd client for test setup/teardown
        self.cleanup_client = etcd3.client(host=self.host, port=self.port)

        # Create the dictionary under test
        self.etcd_dict = EtcdDictionary(host=self.host, port=self.port)

        # Clean up any existing test keys before each test
        self._cleanup_test_keys()

    def tearDown(self):
        # Clean up test keys after each test
        self._cleanup_test_keys()

    def _cleanup_test_keys(self):
        # Helper method to remove test keys
        test_keys = [
            "trtllm/1/test_key1", "trtllm/1/test_key2", "trtllm/2/test_key3"
        ]
        for key in test_keys:
            self.cleanup_client.delete(key)

    def test_put_and_get(self):
        # Test putting and getting a value
        test_key = "trtllm/1/test_key1"
        test_value = "value1"

        # Put the value
        self.etcd_dict.put(test_key, test_value)

        # Get the value
        value, _ = self.etcd_dict.get(test_key)

        # Assert
        self.assertEqual(value.decode('utf-8'), test_value)

    def test_remove(self):
        # Test removing a key
        test_key = "trtllm/1/test_key2"
        test_value = "value2"

        # Setup: Put a value first
        self.etcd_dict.put(test_key, test_value)

        # Remove the key
        self.etcd_dict.remove(test_key)

        # Verify key is removed by trying to get it
        result = self.cleanup_client.get(test_key)
        self.assertIsNone(
            result[0])  # etcd3 returns (None, None) when key doesn't exist

    def test_keys(self):
        # Test listing all keys
        test_data = {
            "trtllm/1/test_key1": "value1",
            "trtllm/1/test_key2": "value2",
            "trtllm/2/test_key3": "value3"
        }

        prefix_data = {"trtllm/1": "value1", "trtllm/2": "value2"}

        # Setup: Put multiple values
        for key, value in test_data.items():
            self.etcd_dict.put(key, value)

        # Get all keys
        keys = self.etcd_dict.keys()

        # Assert all test keys are present
        prefix_keys = set(prefix_data.keys())
        extract_keys = set(keys)
        self.assertEqual(prefix_keys, extract_keys)

    def test_get_nonexistent_key(self):
        # Test getting a key that doesn't exist
        result, _ = self.etcd_dict.get("nonexistent_key")
        self.assertIsNone(result)

    def test_put_update_existing(self):
        # Test updating an existing key
        test_key = "trtllm/1/test_key1"
        initial_value = "initial_value"
        updated_value = "updated_value"

        # Put initial value
        self.etcd_dict.put(test_key, initial_value)

        # Update value
        self.etcd_dict.put(test_key, updated_value)

        # Get updated value
        value, _ = self.etcd_dict.get(test_key)

        # Assert
        self.assertEqual(value.decode('utf-8'), updated_value)


if __name__ == '__main__':
    unittest.main()
