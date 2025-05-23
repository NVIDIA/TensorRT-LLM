import asyncio
import aiohttp
import json
from datetime import datetime
import random
import base64
from torch.multiprocessing.reductions import rebuild_cuda_tensor
import torch
import time
from collections import OrderedDict
import multiprocessing as mp
from queue import Empty
import signal

# Sample request template
REQUEST_TEMPLATE = {
    "model": "llava-hf/llava-v1.6-vicuna-7b-hf",
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe the natural environment in the image."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/seashore.png"
                    }
                }
            ]
        }
    ],
    "max_tokens": 64,
    "temperature": 0
}

REQUEST_TEMPLATE_TWO_IMAGES = {
    "model": "llava-hf/llava-v1.6-vicuna-7b-hf",
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe the natural environment in the image."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/seashore.png"
                    }
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/seashore.png"
                    }
                }
            ]
        }
    ],
    "max_tokens": 64,
    "temperature": 0
}

REQUEST_TEMPLATE_THREE_IMAGES = {
    "model": "llava-hf/llava-v1.6-vicuna-7b-hf",
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe the natural environment in these images."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/seashore.png"
                    }
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/seashore.png"
                    }
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/seashore.png"
                    }
                }
            ]
        }
    ],
    "max_tokens": 64,
    "temperature": 0
}

# List of different image URLs to simulate different images
IMAGE_URLS = [
    "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/seashore.png",
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png",
    "https://huggingface.co/datasets/Sayali9141/traffic_signal_images/resolve/main/61.jpg",
]

class SharedTensorPool:
    def __init__(self, max_handles=1):
        self.active_handles = OrderedDict()
        self.max_handles = max_handles
        self._lock = mp.Lock()
        
        # Setup cleanup process
        self.cleanup_queue = mp.Queue()
        self.cleanup_process = mp.Process(target=self._cleanup_worker, daemon=True)
        self.cleanup_process.start()
        
    def _cleanup_worker(self):
        """Worker process that handles CUDA IPC cleanup"""
        def signal_handler(signum, frame):
            print("Cleanup worker received signal to terminate")
            return
            
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        
        while True:
            try:
                # Check for cleanup tasks without blocking
                try:
                    task = self.cleanup_queue.get_nowait()  # Non-blocking get
                    if task == "STOP":
                        return
                    elif task == "CLEANUP":
                        
                        t0 = time.time()
                        torch.cuda.ipc_collect()
                        t1 = time.time()
                        print(f"delete handles: {t1 - t0} seconds")
                except Empty:
                    # Small sleep to prevent busy waiting
                    time.sleep(0.1)
                
            except Exception as e:
                print(f"Error in cleanup worker: {e}")
                continue
    
    def add_handle(self, key, tensor_info):
        with self._lock:
            if len(self.active_handles) >= self.max_handles:
                # Remove oldest handle and request cleanup
                oldest_key = next(iter(self.active_handles))
                self.close_handle(oldest_key)
            self.active_handles[key] = tensor_info
    
    def close_handle(self, key):
        if key in self.active_handles:
            del self.active_handles[key]  # Directly delete the key and its value
            # Request cleanup
            self.cleanup_queue.put("CLEANUP")
    
    def stop_cleanup_task(self):
        if self.cleanup_process.is_alive():
            self.cleanup_queue.put("STOP")
            self.cleanup_process.join(timeout=5)
            if self.cleanup_process.is_alive():
                self.cleanup_process.terminate()
    

# Global pool instance
tensor_pool = SharedTensorPool()

async def submit_request(session, user_id):
    # Create a unique request for this user

    # Randomly select an image URL
    request1 = REQUEST_TEMPLATE.copy()
    request1["messages"][1]["content"][1]["image_url"]["url"] = random.choice(IMAGE_URLS)

    # Randomly select 2 images for this request
    request2 = REQUEST_TEMPLATE_TWO_IMAGES.copy()
    request2["messages"][1]["content"][1]["image_url"]["url"] = random.choice(IMAGE_URLS)
    request2["messages"][1]["content"][2]["image_url"]["url"] = random.choice(IMAGE_URLS)

    request3 = REQUEST_TEMPLATE_THREE_IMAGES.copy()
    request3["messages"][1]["content"][1]["image_url"]["url"] = random.choice(IMAGE_URLS)
    request3["messages"][1]["content"][2]["image_url"]["url"] = random.choice(IMAGE_URLS)
    request3["messages"][1]["content"][3]["image_url"]["url"] = random.choice(IMAGE_URLS)

    #request = random.choice([request1, request2, request3])
    request = request1

    start_time = datetime.now()
    try:
        async with session.post(
            "http://localhost:8000/v1/multimodal_encoder",
            json=request,
            headers={"Content-Type": "application/json"}
        ) as response:
            result = await response.json()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            print(f"User {user_id} - Request completed in with response {result}")
            
            # Reconstruct tensor from response if embeddings exist
            if result and 'embeddings' in result and result['embeddings']:
                # Get the first embedding info (assuming single tensor)
                tensor_info = result['embeddings'][0]
                
                # Decode base64 strings back to bytes
                storage_handle = base64.b64decode(tensor_info['storage_handle'])
                ref_counter_handle = base64.b64decode(tensor_info['ref_counter_handle'])
                event_handle = base64.b64decode(tensor_info['event_handle'])
                
                # Reconstruct the tensor
                cuda_tensor_info = {
                    "tensor_cls": torch.Tensor,
                    "tensor_size": tuple(tensor_info['tensor_size']),
                    "tensor_stride": tuple(tensor_info['tensor_stride']),
                    "tensor_offset": tensor_info['tensor_offset'],
                    "storage_cls": torch.storage.TypedStorage,
                    "dtype": eval(tensor_info['dtype']),  # Convert string back to torch dtype
                    "storage_device": tensor_info['storage_device'],
                    "storage_handle": storage_handle,
                    "storage_size_bytes": tensor_info['storage_size_bytes'],
                    "storage_offset_bytes": tensor_info['storage_offset_bytes'],
                    "requires_grad": tensor_info['requires_grad'],
                    "ref_counter_handle": ref_counter_handle,
                    "ref_counter_offset": tensor_info['ref_counter_offset'],
                    "event_handle": event_handle,
                    "event_sync_required": tensor_info['event_sync_required']
                }
                
                # Rebuild the tensor
                reconstructed_tensor = rebuild_cuda_tensor(**cuda_tensor_info)
                local_tensor = torch.empty_like(reconstructed_tensor).cuda()
                local_tensor.copy_(reconstructed_tensor)
                #del reconstructed_tensor
                #torch.cuda.ipc_collect()

                print(f"User {user_id} - Reconstructed tensor values: {local_tensor.reshape(-1)[:5]}")
                
                result['tensor'] = local_tensor
                # Store the handle in the pool
                tensor_pool.add_handle(f"user_{user_id}", reconstructed_tensor)
            
            return result
    except Exception as e:
        print(f"User {user_id} - Error: {str(e)}")
        return None

async def simulate_users(num_users, concurrent_limit=10):
    # Create a semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(concurrent_limit)

    async def bounded_submit(session, user_id):
        async with semaphore:
            return await submit_request(session, user_id)

    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(num_users):
            task = asyncio.create_task(bounded_submit(session, i))
            tasks.append(task)

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        return results

async def main():
    try:
        # Number of users to simulate
        num_users = 100 
        # Maximum number of concurrent requests
        concurrent_limit = 100 

        print(f"Starting simulation with {num_users} users, max {concurrent_limit} concurrent requests")
        start_time = datetime.now()

        results = await simulate_users(num_users, concurrent_limit)

        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()

        print(f"\nSimulation completed:")
        print(f"Total time: {total_duration:.2f}s")
        print(f"Average time per request: {total_duration/num_users:.2f}s")
        print(f"Successful requests: {sum(1 for r in results if r is not None)}/{num_users}")
        
        # Optional: Print some statistics
        print(f"Total requests completed: {len(results)}")
        print(f"Active handles in pool: {len(tensor_pool.active_handles)}")
        
    finally:
        # Stop the cleanup process
        tensor_pool.stop_cleanup_task()


if __name__ == "__main__":
    asyncio.run(main())
# 