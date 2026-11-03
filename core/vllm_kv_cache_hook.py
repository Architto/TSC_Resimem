"""ResiMem vllm kv cache hook module."""

import logging
import threading
from typing import Dict, Optional, List, Any, Tuple
from dataclasses import dataclass, field

import torch

logger = logging.getLogger(__name__)


@dataclass
class KVCacheInfo:
    """KVCacheInfo implementation."""

    num_layers: int = 0
    num_blocks: int = 0
    block_size: int = 0
    num_heads: int = 0
    head_dim: int = 0
    dtype: torch.dtype = torch.float16

    kv_caches: Optional[List[torch.Tensor]] = None

    shapes: List[Tuple[int, ...]] = field(default_factory=list)


_global_kv_cache_info: Optional[KVCacheInfo] = None
_global_lock = threading.Lock()


def get_global_kv_cache_info() -> Optional[KVCacheInfo]:
    """Return global kv cache info."""
    return _global_kv_cache_info


def set_global_kv_cache_info(info: KVCacheInfo):
    """Set global kv cache info."""
    global _global_kv_cache_info
    with _global_lock:
        _global_kv_cache_info = info


def _extract_kv_cache_info_from_model(model) -> Dict[str, Any]:

    info = {
        "success": False,
        "error": None,
        "num_layers": 0,
        "shapes": [],
        "dtypes": [],
        "devices": [],
        "model_type": str(type(model)),
        "model_attrs": [],
    }

    try:

        info["model_attrs"] = [attr for attr in dir(model) if not attr.startswith("_")][
            :20
        ]

        kv_caches = None

        if hasattr(model, "kv_caches"):
            kv_caches = model.kv_caches

        elif hasattr(model, "model") and hasattr(model.model, "layers"):

            for layer in model.model.layers:
                if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "kv_cache"):
                    if kv_caches is None:
                        kv_caches = []
                    kv_caches.append(layer.self_attn.kv_cache)

        if kv_caches is None or len(kv_caches) == 0:

            info["error"] = (
                f"extract kv cache info from model: unavailable kv_caches, : {type(model).__name__}"
            )
            return info

        info["num_layers"] = len(kv_caches)

        for i, cache in enumerate(kv_caches):
            if cache is not None:
                info["shapes"].append(tuple(cache.shape))
                info["dtypes"].append(str(cache.dtype))
                info["devices"].append(str(cache.device))

        info["success"] = True

    except Exception as e:
        info["error"] = str(e)

    return info


def _get_kv_cache_tensors(model) -> Tuple[bool, List[torch.Tensor], str]:

    try:
        if not hasattr(model, "kv_caches"):
            return False, [], "model get kv cache tensors kv_caches "

        kv_caches = model.kv_caches

        if not kv_caches:
            return False, [], "kv_caches get kv cache tensors"

        return True, kv_caches, ""

    except Exception as e:
        return False, [], str(e)


# KV Cache Hook
class VLLMKVCacheHook:
    """VLLMKVCacheHook implementation."""

    def __init__(self, llm: Any):

        self.llm = llm
        self.kv_cache_info: Optional[KVCacheInfo] = None
        self._initialized = False
        self._lock = threading.RLock()

        logger.info("VLLMKVCacheHook init: created")

    def initialize(self) -> bool:
        """Initialize."""
        if self._initialized:
            return True

        logger.info("initialize KV Cache Hook...")

        try:

            results = self.llm.apply_model(_extract_kv_cache_info_from_model)

            if not results:
                logger.error("apply_model initialize")
                return False

            result = results[0] if isinstance(results, list) else results

            if not result.get("success"):
                logger.warning(f"initialize: failed: {result.get('error')}")
                logger.info(f"initialize: {result.get('model_type', 'unknown')}")
                logger.info(f"initialize: {result.get('model_attrs', [])[:10]}")

                logger.info("initialize:completed successfully KV Cache...")

                self.kv_cache_info = KVCacheInfo(
                    num_layers=0,
                    num_blocks=0,
                )
                self._initialized = True
                logger.warning(
                    "⚠️ KV Cache Hook initialize: num_layers  num_blocks  0,"
                    ".,completed successfully "
                    "VLLMKVCacheHook.set_kv_cache_info() ."
                )
                return True

            self.kv_cache_info = KVCacheInfo(
                num_layers=result["num_layers"],
                shapes=[tuple(s) for s in result["shapes"]],
            )

            logger.info(f"✅ KV Cache initialize: completed successfully:")
            logger.info(f"   initialize: {self.kv_cache_info.num_layers}")
            if self.kv_cache_info.shapes:
                logger.info(f"   initialize: {self.kv_cache_info.shapes[0]}")
                logger.info(f"   initialize: {result['devices'][0]}")

            self._initialized = True

            set_global_kv_cache_info(self.kv_cache_info)

            return True

        except Exception as e:
            logger.error(f"initialize KV Cache Hook failed: {e}", exc_info=True)
            return False

    def get_kv_cache_shape(self) -> Optional[Tuple[int, ...]]:
        """Return kv cache shape."""
        if self.kv_cache_info and self.kv_cache_info.shapes:
            return self.kv_cache_info.shapes[0]
        return None

    def get_num_layers(self) -> int:
        """Return num layers."""
        if self.kv_cache_info:
            return self.kv_cache_info.num_layers
        return 0

    def get_num_blocks(self) -> int:
        """Return num blocks."""
        shape = self.get_kv_cache_shape()
        if shape and len(shape) >= 1:

            return shape[0]
        return 0

    def execute_on_kv_cache(self, func) -> Any:
        """Execute on kv cache."""

        def wrapper(model):
            if hasattr(model, "kv_caches"):
                return func(model, model.kv_caches)
            return None

        return self.llm.apply_model(wrapper)

    def set_kv_cache_info(
        self,
        num_layers: int = 0,
        num_blocks: int = 0,
        block_size: int = 0,
        num_heads: int = 0,
        head_dim: int = 0,
    ):
        """Set kv cache info."""
        with self._lock:
            if self.kv_cache_info is None:
                self.kv_cache_info = KVCacheInfo()
            if num_layers > 0:
                self.kv_cache_info.num_layers = num_layers
            if num_blocks > 0:
                self.kv_cache_info.num_blocks = num_blocks
            if block_size > 0:
                self.kv_cache_info.block_size = block_size
            if num_heads > 0:
                self.kv_cache_info.num_heads = num_heads
            if head_dim > 0:
                self.kv_cache_info.head_dim = head_dim
            logger.info(
                f"set kv cache info KV Cache : layers={num_layers}, blocks={num_blocks}"
            )

    @property
    def is_initialized(self) -> bool:
        return self._initialized


class RealSwapManager:
    """RealSwapManager implementation."""

    def __init__(
        self,
        llm: Any,
        cpu_swap_space_gb: float = 32.0,
    ):

        self.llm = llm
        self.hook = VLLMKVCacheHook(llm)
        self.cpu_swap_space_gb = cpu_swap_space_gb

        self.user_sessions: Dict[str, Dict] = {}  # user_id -> {prompts, responses, ...}

        self.user_blocks: Dict[str, List[int]] = {}  # user_id -> [block_ids]

        self.swapped_out_users: set = set()

        self.stats = {
            "total_swap_outs": 0,
            "total_swap_ins": 0,
            "total_sessions_saved": 0,
            "total_sessions_restored": 0,
        }

        self._lock = threading.RLock()

        logger.info("RealSwapManager init: created( Drop & Recompute )")

    def initialize(self) -> bool:
        """Initialize."""
        return self.hook.initialize()

    def register_user_blocks(self, user_id: str, block_ids: List[int]):
        """Register user blocks."""
        with self._lock:
            self.user_blocks[user_id] = list(block_ids)
            logger.info(
                f"register user blocks: registered {user_id}  {len(block_ids)} "
            )

    def save_user_session(self, user_id: str, prompt: str, response: str):
        """Save user session."""
        with self._lock:
            if user_id not in self.user_sessions:
                self.user_sessions[user_id] = {
                    "prompts": [],
                    "responses": [],
                    "total_tokens": 0,
                }

            session = self.user_sessions[user_id]
            session["prompts"].append(prompt)
            session["responses"].append(response)
            session["total_tokens"] += len(prompt) + len(response)

    def swap_out_user(self, user_id: str) -> bool:
        """Swap out user."""
        with self._lock:
            if user_id in self.swapped_out_users:
                logger.warning(f"swap out user {user_id} already active")
                return False

            logger.info(f"🔴 Swap Out swap out user {user_id}(Drop & Recompute )")

            self.swapped_out_users.add(user_id)

            self.stats["total_swap_outs"] += 1
            if user_id in self.user_sessions:
                self.stats["total_sessions_saved"] += 1
                session = self.user_sessions[user_id]
                logger.info(f"   swap out user: saved: {len(session['prompts'])} ")

            block_count = len(self.user_blocks.get(user_id, []))
            logger.info(
                f"✅ Swap Out swap out user: completed successfully:  {user_id}, {block_count} released"
            )

            return True

    def swap_in_user(self, user_id: str) -> bool:
        """Swap in user."""
        with self._lock:
            if user_id not in self.swapped_out_users:
                logger.warning(f"swap in user {user_id} evicted")
                return False

            logger.info(f"🟢 Swap In swap in user {user_id}(Drop & Recompute )")

            self.swapped_out_users.remove(user_id)

            self.stats["total_swap_ins"] += 1
            if user_id in self.user_sessions:
                self.stats["total_sessions_restored"] += 1
                session = self.user_sessions[user_id]
                logger.info(f"   swap in user: restored: {len(session['prompts'])} ")

            logger.info(
                f"✅ Swap In swap in user: completed successfully:  {user_id} request"
            )

            return True

    def is_user_swapped_out(self, user_id: str) -> bool:
        """Return whether user swapped out."""
        with self._lock:
            return user_id in self.swapped_out_users

    def get_user_session(self, user_id: str) -> Optional[Dict]:
        """Return user session."""
        with self._lock:
            session = self.user_sessions.get(user_id)
            if session is None:
                return None

            return dict(session)

    def get_statistics(self) -> Dict:
        """Return statistics."""
        with self._lock:
            return {
                **self.stats,
                "swapped_out_users": len(self.swapped_out_users),
                "total_users": len(self.user_blocks),
                "sessions_saved": len(self.user_sessions),
            }


def create_kv_cache_hook(llm: Any) -> VLLMKVCacheHook:
    """Create kv cache hook."""
    hook = VLLMKVCacheHook(llm)
    hook.initialize()
    return hook


def create_real_swap_manager(
    llm: Any, cpu_swap_space_gb: float = 32.0
) -> RealSwapManager:
    """Create real swap manager."""
    manager = RealSwapManager(llm, cpu_swap_space_gb)
    manager.initialize()
    return manager
