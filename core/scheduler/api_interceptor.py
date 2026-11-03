"""ResiMem api interceptor module."""

import asyncio
import logging
import threading
from typing import Dict, Optional, Callable, Any
from functools import wraps

from .multi_user_types import extract_user_id, extract_session_id
from .focus_manager import FocusManager


logger = logging.getLogger(__name__)
_global_focus_manager: Optional[FocusManager] = None
_global_scheduler: Optional[Any] = None


def set_global_focus_manager(focus_manager: FocusManager):
    """Set global focus manager."""
    global _global_focus_manager
    _global_focus_manager = focus_manager
    logger.info("set global focus manager FocusManager ")


def set_global_scheduler(scheduler: Any):
    """Set global scheduler."""
    global _global_scheduler
    _global_scheduler = scheduler
    logger.info("set global scheduler ResiMemScheduler ")


def get_global_scheduler() -> Optional[Any]:
    """Return global scheduler."""
    return _global_scheduler


def get_global_focus_manager() -> Optional[FocusManager]:
    """Return global focus manager."""
    return _global_focus_manager


def intercept_request(request_dict: Dict) -> str:
    """Intercept request."""

    user_id = extract_user_id(request_dict)

    if _global_focus_manager is not None:
        try:
            _global_focus_manager.notify_activity(user_id)
            logger.debug(f"intercept request: notified FocusManager : {user_id}")
        except Exception as e:
            logger.error(
                f"intercept request: notified FocusManager failed: {e}", exc_info=True
            )
    else:
        logger.debug(f"FocusManager intercept request,skipped {user_id} notified")

    return user_id


async def intercept_request_async(request_dict: Dict) -> str:
    """Intercept request async."""

    loop = asyncio.get_running_loop()
    user_id = extract_user_id(request_dict)

    if _global_focus_manager is not None:

        await loop.run_in_executor(
            None, lambda: _global_focus_manager.notify_activity(user_id)
        )

    return user_id


def intercept_chat_completion(func: Callable) -> Callable:
    """Intercept chat completion."""

    @wraps(func)
    async def wrapper(*args, **kwargs):

        request = None
        if args:
            request = args[0]
        elif "request" in kwargs:
            request = kwargs["request"]

        if request is not None:
            try:

                if hasattr(request, "user"):
                    user_id = request.user or "default_user"
                elif hasattr(request, "dict"):
                    request_dict = request.dict()
                    user_id = request_dict.get("user", "default_user")
                else:
                    user_id = "default_user"

                if _global_focus_manager is not None:
                    _global_focus_manager.notify_activity(user_id)
                    logger.debug(f"wrapper: request,: {user_id}")
            except Exception as e:
                logger.error(f"wrapper: failed: {e}", exc_info=True)

        return await func(*args, **kwargs)

    return wrapper


def intercept_completion(func: Callable) -> Callable:
    """Intercept completion."""

    @wraps(func)
    async def wrapper(*args, **kwargs):

        request = None
        if args:
            request = args[0]
        elif "request" in kwargs:
            request = kwargs["request"]

        if request is not None:
            try:
                if hasattr(request, "user"):
                    user_id = request.user or "default_user"
                else:
                    user_id = "default_user"

                if _global_focus_manager is not None:
                    _global_focus_manager.notify_activity(user_id)
            except Exception as e:
                logger.error(f"wrapper: failed: {e}", exc_info=True)

        return await func(*args, **kwargs)

    return wrapper


class APIServerPatch:
    """APIServerPatch implementation."""

    def __init__(self, focus_manager: FocusManager):

        self.focus_manager = focus_manager
        self._original_handlers: Dict[str, Callable] = {}
        self._patched = False

        set_global_focus_manager(focus_manager)

    def patch_api_server(self, api_server_module: Any):
        """Patch api server."""
        if self._patched:
            logger.warning("API Server patch api server: already active,skipped")
            return

        try:

            if hasattr(api_server_module, "create_chat_completion"):
                self._original_handlers["create_chat_completion"] = (
                    api_server_module.create_chat_completion
                )
                api_server_module.create_chat_completion = intercept_chat_completion(
                    api_server_module.create_chat_completion
                )
                logger.info("patch api server create_chat_completion")

            if hasattr(api_server_module, "create_completion"):
                self._original_handlers["create_completion"] = (
                    api_server_module.create_completion
                )
                api_server_module.create_completion = intercept_completion(
                    api_server_module.create_completion
                )
                logger.info("patch api server create_completion")

            self._patched = True
            logger.info("vLLM API Server patch api server: completed successfully")

        except Exception as e:
            logger.error(f"patch api server API Server failed: {e}", exc_info=True)

    def unpatch_api_server(self, api_server_module: Any):
        """Unpatch api server."""
        if not self._patched:
            return

        try:
            for name, handler in self._original_handlers.items():
                if hasattr(api_server_module, name):
                    setattr(api_server_module, name, handler)
                    logger.info(f"unpatch api server: restored {name}")

            self._original_handlers.clear()
            self._patched = False
            logger.info("vLLM API Server unpatch api server")

        except Exception as e:
            logger.error(f"unpatch api server API Server failed: {e}", exc_info=True)


class FocusMiddleware:
    """FocusMiddleware implementation."""

    def __init__(self, app, focus_manager: FocusManager):

        self.app = app
        self.focus_manager = focus_manager

    async def __call__(self, scope, receive, send):

        if scope["type"] == "http" and self.focus_manager is not None:

            try:
                headers = dict(scope.get("headers", []))

                user_id_bytes = headers.get(b"x-user-id")
                if user_id_bytes:
                    user_id = user_id_bytes.decode("utf-8", errors="ignore")
                    self.focus_manager.notify_activity(user_id)
                    logger.debug(f"FocusMiddleware: call header  user_id={user_id}")
            except Exception as e:
                logger.debug(f"FocusMiddleware: call user_id failed: {e}")

        await self.app(scope, receive, send)


async def handle_focus_notification(user_id: str) -> Dict:
    """Handle focus notification."""
    if _global_focus_manager is None:
        return {
            "success": False,
            "error": "FocusManager handle focus notification: not initialized",
        }

    try:
        _global_focus_manager.notify_activity(user_id)
        stats = _global_focus_manager.get_statistics()

        return {
            "success": True,
            "user_id": user_id,
            "current_active_user": stats.get("current_active_user"),
            "user_state": (lambda s: s.value if s else None)(
                _global_focus_manager.get_user_state(user_id)
            ),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


async def handle_focus_status() -> Dict:
    """Handle focus status."""
    if _global_focus_manager is None:
        return {
            "success": False,
            "error": "FocusManager handle focus status: not initialized",
        }

    try:
        stats = _global_focus_manager.get_statistics()
        return {"success": True, **stats}
    except Exception as e:
        return {"success": False, "error": str(e)}


def create_api_server_patch(focus_manager: FocusManager) -> APIServerPatch:
    """Create api server patch."""
    return APIServerPatch(focus_manager)


def setup_request_interception(focus_manager: FocusManager):
    """Setup request interception."""
    set_global_focus_manager(focus_manager)
    logger.info("setup request interception: request")


__all__ = [
    "set_global_focus_manager",
    "get_global_focus_manager",
    "intercept_request",
    "intercept_request_async",
    "intercept_chat_completion",
    "intercept_completion",
    "APIServerPatch",
    "FocusMiddleware",
    "handle_focus_notification",
    "handle_focus_status",
    "create_api_server_patch",
    "setup_request_interception",
]
