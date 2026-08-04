import asyncio
import threading
from contextlib import contextmanager
from typing import Callable, Optional

from tensorrt_llm._utils import print_all_stacks
from tensorrt_llm.logger import logger


class HangDetector:
    def __init__(
        self, timeout: Optional[int] = None, on_detected: Optional[Callable[[], None]] = None
    ):
        self.timeout = timeout if timeout is not None else 300
        assert self.timeout > 0, "timeout must be greater than 0"
        self.on_detected = on_detected or (lambda: None)
        self.task = None
        self.loop = None
        self.loop_thread = None
        self.lock = threading.Lock()
        self.active = False
        self._detected = False

    def start(self):
        """Enable hang detection."""

        def run_loop():
            asyncio.set_event_loop(self.loop)
            self.loop.run_forever()

        self.active = True
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=run_loop, daemon=True, name="hang_detector_loop")
        self.loop_thread.start()

    async def _detect_hang(self):
        await asyncio.sleep(self.timeout)
        with self.lock:
            self._detected = True
            logger.error(f"Hang detected after {self.timeout} seconds.")
            print_all_stacks()
            self.on_detected()

    def detected(self):
        """Return True if hang is detected."""
        with self.lock:
            return self._detected

    def checkpoint(self):
        """Reset hang detection timer."""
        self.cancel_task()
        if self.active:
            self.task = asyncio.run_coroutine_threadsafe(self._detect_hang(), self.loop)

    def cancel_task(self):
        """Cancel the hang detection task."""
        if self.task is not None and not self.task.done():
            self.task.cancel()
            self.task = None

    @contextmanager
    def pause(self):
        """Pause hang detection in scope."""
        try:
            self.cancel_task()
            yield
        finally:
            self.checkpoint()

    def stop(self):
        """Stop hang detection."""
        self.active = False
        self.cancel_task()
        if self.loop is not None:
            # Cancel all pending tasks before stopping the loop
            def cancel_all_tasks():
                for task in asyncio.all_tasks(self.loop):
                    if not task.done():
                        task.cancel()
                self.loop.call_soon(self.loop.stop)

            self.loop.call_soon_threadsafe(cancel_all_tasks)

            if self.loop_thread is not None and self.loop_thread.is_alive():
                self.loop_thread.join()

            self.loop = None
            self.loop_thread = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
        return False
