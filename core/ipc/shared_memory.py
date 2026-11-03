"""ResiMem shared memory module."""

import struct
import json
import time
import logging
import tempfile
import os
from multiprocessing import shared_memory
from typing import Optional, List
from threading import Lock

try:
    import fcntl

    _HAS_FCNTL = True
except Exception:
    fcntl = None
    _HAS_FCNTL = False

from .protocol import Message, WorkerInfo, WorkerStatus


class GRMSharedMemory:
    """GRMSharedMemory implementation."""

    MAGIC_NUMBER = 0x5245534D454D  # "RESMEM"
    VERSION = 2
    MAX_VLLM_WORKERS = 8
    MAX_DNN_WORKERS = 8
    MAX_MESSAGES = 64
    MESSAGE_SIZE = 512  # 512 bytes per message

    # Offsets
    HEADER_SIZE = 128
    VLLM_WORKER_SIZE = 512
    DNN_WORKER_SIZE = 512
    QUEUE_HEADER_SIZE = 16  # head(4) + tail(4) + size(4) + capacity(4)
    MESSAGE_QUEUE_SIZE = (
        QUEUE_HEADER_SIZE + MESSAGE_SIZE * MAX_MESSAGES
    )  # 16 + 512*64 = 32784 bytes

    REQUEST_QUEUE_OFFSET = (
        HEADER_SIZE
        + VLLM_WORKER_SIZE * MAX_VLLM_WORKERS
        + DNN_WORKER_SIZE * MAX_DNN_WORKERS
    )
    RESPONSE_QUEUE_OFFSET = REQUEST_QUEUE_OFFSET + MESSAGE_QUEUE_SIZE
    TOTAL_SIZE = RESPONSE_QUEUE_OFFSET + MESSAGE_QUEUE_SIZE  # ~73KB

    def __init__(self, name: str = "resimem_grm", create: bool = True):

        self.name = name
        self.logger = logging.getLogger(f"GRMSharedMemory_{name}")
        self.local_lock = Lock()
        self.lock_file_path = os.path.join(tempfile.gettempdir(), f"{name}.lock")
        self._lock_file = None

        try:

            lock_dir = os.path.dirname(self.lock_file_path)
            os.makedirs(lock_dir, exist_ok=True)

            self._lock_file = open(self.lock_file_path, "a+b")
        except Exception as e:
            self.logger.error(
                f"Failed to initialize lock file at {self.lock_file_path}: {e}"
            )
            raise

        try:
            if create:

                try:
                    self.shm = shared_memory.SharedMemory(
                        name=name, create=True, size=self.TOTAL_SIZE
                    )
                except FileExistsError:

                    self.logger.warning(f"init {name} ,." ":,.")
                    try:
                        old_shm = shared_memory.SharedMemory(name=name)
                        old_shm.close()
                        old_shm.unlink()
                    except (FileNotFoundError, OSError) as e:
                        self.logger.warning(f"init: failed: {e}")

                    self.shm = shared_memory.SharedMemory(
                        name=name, create=True, size=self.TOTAL_SIZE
                    )
                self._init_memory()
                self.logger.info(
                    f"Created shared memory: {name} ({self.TOTAL_SIZE} bytes)"
                )
            else:
                self.shm = shared_memory.SharedMemory(name=name)
                self.logger.info(f"Connected to shared memory: {name}")

                magic = self._read_uint64(0)
                if magic != self.MAGIC_NUMBER:
                    raise ValueError(f"Invalid magic number: {magic:016x}")

        except Exception as e:
            self.logger.error(f"Failed to initialize shared memory: {e}")
            raise

    def _init_memory(self):

        self.shm.buf[:] = b"\x00" * len(self.shm.buf)

        self._write_uint64(0, self.MAGIC_NUMBER)  # magic_number
        self._write_uint32(8, self.VERSION)  # version
        self._write_uint8(12, 0)  # lock_flag
        self._write_uint32(13, 0)  # num_vllm_workers
        self._write_uint32(17, 0)  # num_dnn_workers
        self._write_int64(21, int(time.time() * 1e6))  # last_update_time

        self._write_uint32(self.REQUEST_QUEUE_OFFSET, 0)  # head
        self._write_uint32(self.REQUEST_QUEUE_OFFSET + 4, 0)  # tail
        self._write_uint32(self.REQUEST_QUEUE_OFFSET + 8, 0)  # size
        self._write_uint32(
            self.REQUEST_QUEUE_OFFSET + 12, self.MAX_MESSAGES
        )  # capacity

        self._write_uint32(self.RESPONSE_QUEUE_OFFSET, 0)  # head
        self._write_uint32(self.RESPONSE_QUEUE_OFFSET + 4, 0)  # tail
        self._write_uint32(self.RESPONSE_QUEUE_OFFSET + 8, 0)  # size
        self._write_uint32(
            self.RESPONSE_QUEUE_OFFSET + 12, self.MAX_MESSAGES
        )  # capacity

        self.logger.info(
            f"Shared memory initialized (dual-queue design, v{self.VERSION})"
        )

    def _read_uint8(self, offset: int) -> int:
        return struct.unpack_from("B", self.shm.buf, offset)[0]

    def _write_uint8(self, offset: int, value: int):
        struct.pack_into("B", self.shm.buf, offset, value)

    def _read_uint32(self, offset: int) -> int:
        return struct.unpack_from("I", self.shm.buf, offset)[0]

    def _write_uint32(self, offset: int, value: int):
        struct.pack_into("I", self.shm.buf, offset, value)

    def _read_int32(self, offset: int) -> int:
        return struct.unpack_from("i", self.shm.buf, offset)[0]

    def _write_int32(self, offset: int, value: int):
        struct.pack_into("i", self.shm.buf, offset, value)

    def _read_uint64(self, offset: int) -> int:
        return struct.unpack_from("Q", self.shm.buf, offset)[0]

    def _write_uint64(self, offset: int, value: int):
        struct.pack_into("Q", self.shm.buf, offset, value)

    def _read_int64(self, offset: int) -> int:
        return struct.unpack_from("q", self.shm.buf, offset)[0]

    def _write_int64(self, offset: int, value: int):
        struct.pack_into("q", self.shm.buf, offset, value)

    def _read_float32(self, offset: int) -> float:
        return struct.unpack_from("f", self.shm.buf, offset)[0]

    def _write_float32(self, offset: int, value: float):
        struct.pack_into("f", self.shm.buf, offset, value)

    def _read_string(self, offset: int, max_len: int) -> str:

        data = bytes(self.shm.buf[offset : offset + max_len])

        null_idx = data.find(b"\x00")
        if null_idx != -1:
            data = data[:null_idx]
        return data.decode("utf-8", errors="ignore")

    def _write_string(self, offset: int, value: str, max_len: int):

        data = value.encode("utf-8")[:max_len]
        self.shm.buf[offset : offset + len(data)] = data

        if len(data) < max_len:
            self.shm.buf[offset + len(data) : offset + max_len] = b"\x00" * (
                max_len - len(data)
            )

    def acquire_lock(self, timeout: float = 1.0) -> bool:
        """Acquire lock."""
        if _HAS_FCNTL and self._lock_file is not None:
            start_time = time.time()
            while (time.time() - start_time) < timeout:
                with self.local_lock:
                    try:
                        fcntl.flock(
                            self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
                        )

                        self._write_uint8(12, 1)
                        return True
                    except BlockingIOError:
                        pass
                    except Exception:
                        pass
                time.sleep(0.0001)

            self.logger.warning(f"Lock acquisition timeout after {timeout}s")
            return False

        self.logger.debug("acquire lock fallback (fcntl )")
        start_time = time.time()
        while (time.time() - start_time) < timeout:
            with self.local_lock:
                lock_flag = self._read_uint8(12)
                if lock_flag == 0:
                    self._write_uint8(12, 1)
                    return True
            time.sleep(0.0001)

        self.logger.warning(f"Lock acquisition timeout after {timeout}s")
        return False

    def release_lock(self):
        """Release lock."""
        if _HAS_FCNTL and self._lock_file is not None:
            with self.local_lock:
                try:

                    self._write_uint8(12, 0)
                except Exception:
                    pass
                try:
                    fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass
            return

        with self.local_lock:
            self._write_uint8(12, 0)

    def register_vllm_worker(self, worker_info: WorkerInfo) -> bool:
        """Register vllm worker."""
        if not self.acquire_lock():
            return False

        try:
            num_workers = self._read_uint32(13)
            if num_workers >= self.MAX_VLLM_WORKERS:
                self.logger.error(f"Max vLLM workers ({self.MAX_VLLM_WORKERS}) reached")
                return False

            offset = self.HEADER_SIZE + num_workers * self.VLLM_WORKER_SIZE
            self._write_worker_info(offset, worker_info, is_vllm=True)

            self._write_uint32(13, num_workers + 1)
            self._write_int64(21, int(time.time() * 1e6))

            self.logger.info(f"Registered vLLM worker: {worker_info.worker_id}")
            return True
        finally:
            self.release_lock()

    def register_dnn_worker(self, worker_info: WorkerInfo) -> bool:
        """Register dnn worker."""
        if not self.acquire_lock():
            return False

        try:
            num_workers = self._read_uint32(17)
            if num_workers >= self.MAX_DNN_WORKERS:
                self.logger.error(f"Max DNN workers ({self.MAX_DNN_WORKERS}) reached")
                return False

            offset = (
                self.HEADER_SIZE
                + self.VLLM_WORKER_SIZE * self.MAX_VLLM_WORKERS
                + num_workers * self.DNN_WORKER_SIZE
            )
            self._write_worker_info(offset, worker_info, is_vllm=False)

            self._write_uint32(17, num_workers + 1)
            self._write_int64(21, int(time.time() * 1e6))

            self.logger.info(f"Registered DNN worker: {worker_info.worker_id}")
            return True
        finally:
            self.release_lock()

    def _write_worker_info(self, offset: int, worker_info: WorkerInfo, is_vllm: bool):

        self._write_string(offset, worker_info.worker_id, 64)
        self._write_uint8(offset + 64, worker_info.status.value)
        self._write_float32(offset + 65, worker_info.memory_used_gb)

        if is_vllm:
            self._write_int32(offset + 69, worker_info.num_blocks or 0)
            self._write_float32(offset + 73, worker_info.kv_cache_size_gb or 0.0)
            self._write_int64(offset + 77, int(worker_info.last_heartbeat * 1e6))

            json_data = json.dumps(worker_info.to_dict())
            self._write_string(offset + 85, json_data, 427)
        else:
            self._write_float32(offset + 69, worker_info.memory_releasable_gb or 0.0)
            self._write_int64(offset + 73, int(worker_info.last_heartbeat * 1e6))

            json_data = json.dumps(worker_info.to_dict())
            self._write_string(offset + 81, json_data, 431)

    def get_all_vllm_workers(self) -> List[WorkerInfo]:
        """Return all vllm workers."""
        if not self.acquire_lock(timeout=0.1):
            return []

        try:
            num_workers = self._read_uint32(13)
            workers = []

            for i in range(num_workers):
                offset = self.HEADER_SIZE + i * self.VLLM_WORKER_SIZE
                json_str = self._read_string(offset + 85, 427)
                try:
                    worker_dict = json.loads(json_str)
                    workers.append(WorkerInfo.from_dict(worker_dict))
                except json.JSONDecodeError:
                    self.logger.warning(f"Failed to parse vLLM worker {i}")

            return workers
        finally:
            self.release_lock()

    def get_all_dnn_workers(self) -> List[WorkerInfo]:
        """Return all dnn workers."""
        if not self.acquire_lock(timeout=0.1):
            return []

        try:
            num_workers = self._read_uint32(17)
            workers = []

            for i in range(num_workers):
                offset = (
                    self.HEADER_SIZE
                    + self.VLLM_WORKER_SIZE * self.MAX_VLLM_WORKERS
                    + i * self.DNN_WORKER_SIZE
                )
                json_str = self._read_string(offset + 81, 431)
                try:
                    worker_dict = json.loads(json_str)
                    workers.append(WorkerInfo.from_dict(worker_dict))
                except json.JSONDecodeError:
                    self.logger.warning(f"Failed to parse DNN worker {i}")

            return workers
        finally:
            self.release_lock()

    def push_request(self, message: Message) -> bool:
        """Push request."""
        return self._push_to_queue(self.REQUEST_QUEUE_OFFSET, message)

    def pop_request(self) -> Optional[Message]:
        """Pop request."""
        return self._pop_from_queue(self.REQUEST_QUEUE_OFFSET)

    def peek_request(self) -> Optional[Message]:
        """Peek request."""
        return self._peek_from_queue(self.REQUEST_QUEUE_OFFSET)

    def push_response(self, message: Message) -> bool:
        """Push response."""
        return self._push_to_queue(self.RESPONSE_QUEUE_OFFSET, message)

    def pop_response(self) -> Optional[Message]:
        """Pop response."""
        return self._pop_from_queue(self.RESPONSE_QUEUE_OFFSET)

    def pop_response_for_worker(self, worker_id: str) -> Optional[Message]:
        """Pop response for worker."""
        return self._pop_from_queue_for_worker(self.RESPONSE_QUEUE_OFFSET, worker_id)

    def peek_response(self) -> Optional[Message]:
        """Peek response."""
        return self._peek_from_queue(self.RESPONSE_QUEUE_OFFSET)

    def _push_to_queue(self, queue_offset: int, message: Message) -> bool:

        if not self.acquire_lock(timeout=0.01):
            return False

        try:
            head = self._read_uint32(queue_offset)
            tail = self._read_uint32(queue_offset + 4)
            size = self._read_uint32(queue_offset + 8)
            capacity = self._read_uint32(queue_offset + 12)

            if size >= capacity:
                self.logger.warning(f"Message queue full (offset={queue_offset})")
                return False

            msg_offset = queue_offset + 16 + tail * self.MESSAGE_SIZE
            msg_json = json.dumps(message.to_dict())
            self._write_string(msg_offset, msg_json, self.MESSAGE_SIZE)

            new_tail = (tail + 1) % capacity
            self._write_uint32(queue_offset + 4, new_tail)
            self._write_uint32(queue_offset + 8, size + 1)

            return True
        finally:
            self.release_lock()

    def _pop_from_queue(self, queue_offset: int) -> Optional[Message]:

        if not self.acquire_lock(timeout=0.01):
            return None

        try:
            head = self._read_uint32(queue_offset)
            size = self._read_uint32(queue_offset + 8)
            capacity = self._read_uint32(queue_offset + 12)

            if size == 0:
                return None

            msg_offset = queue_offset + 16 + head * self.MESSAGE_SIZE
            msg_json = self._read_string(msg_offset, self.MESSAGE_SIZE)

            try:
                msg_dict = json.loads(msg_json)
                message = Message.from_dict(msg_dict)
            except json.JSONDecodeError:
                self.logger.warning("Failed to parse message")
                return None

            new_head = (head + 1) % capacity
            self._write_uint32(queue_offset, new_head)
            self._write_uint32(queue_offset + 8, size - 1)

            return message
        finally:
            self.release_lock()

    def _peek_from_queue(self, queue_offset: int) -> Optional[Message]:

        if not self.acquire_lock(timeout=0.01):
            return None

        try:
            head = self._read_uint32(queue_offset)
            size = self._read_uint32(queue_offset + 8)

            if size == 0:
                return None

            msg_offset = queue_offset + 16 + head * self.MESSAGE_SIZE
            msg_json = self._read_string(msg_offset, self.MESSAGE_SIZE)

            try:
                msg_dict = json.loads(msg_json)
                return Message.from_dict(msg_dict)
            except json.JSONDecodeError:
                return None
        finally:
            self.release_lock()

    def _pop_from_queue_for_worker(
        self, queue_offset: int, worker_id: str
    ) -> Optional[Message]:

        if not self.acquire_lock(timeout=0.01):
            return None

        try:
            head = self._read_uint32(queue_offset)
            tail = self._read_uint32(queue_offset + 4)
            size = self._read_uint32(queue_offset + 8)
            capacity = self._read_uint32(queue_offset + 12)

            if size == 0:
                return None

            target_rel_index = -1
            target_message: Optional[Message] = None

            for rel_index in range(size):
                queue_index = (head + rel_index) % capacity
                msg_offset = queue_offset + 16 + queue_index * self.MESSAGE_SIZE
                msg_json = self._read_string(msg_offset, self.MESSAGE_SIZE)

                try:
                    msg_dict = json.loads(msg_json)
                    message = Message.from_dict(msg_dict)
                except json.JSONDecodeError:
                    continue

                payload = message.payload or {}
                target_worker = payload.get("target_worker")
                payload_worker = payload.get("worker_id")

                if target_worker == worker_id or payload_worker == worker_id:
                    target_rel_index = rel_index
                    target_message = message
                    break

            if target_rel_index < 0 or target_message is None:
                return None

            if target_rel_index == 0:
                current_head_offset = queue_offset + 16 + head * self.MESSAGE_SIZE
                self._write_string(current_head_offset, "", self.MESSAGE_SIZE)
                new_head = (head + 1) % capacity
                self._write_uint32(queue_offset, new_head)
                self._write_uint32(queue_offset + 8, size - 1)
                return target_message

            for rel_index in range(target_rel_index, size - 1):
                src_index = (head + rel_index + 1) % capacity
                dst_index = (head + rel_index) % capacity
                src_offset = queue_offset + 16 + src_index * self.MESSAGE_SIZE
                dst_offset = queue_offset + 16 + dst_index * self.MESSAGE_SIZE
                self.shm.buf[dst_offset : dst_offset + self.MESSAGE_SIZE] = (
                    self.shm.buf[src_offset : src_offset + self.MESSAGE_SIZE]
                )

            last_index = (head + size - 1) % capacity
            last_offset = queue_offset + 16 + last_index * self.MESSAGE_SIZE
            self._write_string(last_offset, "", self.MESSAGE_SIZE)

            new_tail = (tail - 1 + capacity) % capacity
            self._write_uint32(queue_offset + 4, new_tail)
            self._write_uint32(queue_offset + 8, size - 1)

            return target_message
        finally:
            self.release_lock()

    def close(self):
        """Close."""
        self.shm.close()
        if self._lock_file is not None:
            try:
                self._lock_file.close()
            except Exception:
                pass
            self._lock_file = None
        self.logger.info(f"Closed shared memory: {self.name}")

    def unlink(self):
        """Unlink."""
        self.shm.unlink()
        self.logger.info(f"Unlinked shared memory: {self.name}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
