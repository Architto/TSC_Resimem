"""ResiMem focus manager module."""

import threading
import time
import logging
from typing import Dict, Optional, Callable, Set
from datetime import datetime

from .multi_user_types import (
    UserState,
    UserStateInfo,
    FocusManagerConfig,
    extract_user_id,
)


logger = logging.getLogger(__name__)


class FocusManager:
    """FocusManager implementation."""

    def __init__(
        self,
        config: FocusManagerConfig,
        swap_out_callback: Optional[Callable[[str], bool]] = None,
        swap_in_callback: Optional[Callable[[str], bool]] = None,
    ):

        self.config = config
        self.swap_out_callback = swap_out_callback
        self.swap_in_callback = swap_in_callback

        self.users: Dict[str, UserStateInfo] = {}
        self.active_user: Optional[str] = None

        self._lock = threading.RLock()

        self._cooling_thread: Optional[threading.Thread] = None
        self._stop_cooling = threading.Event()

        self.stats = {
            "total_focus_switches": 0,
            "total_swap_outs": 0,
            "total_swap_ins": 0,
            "total_cooling_transitions": 0,
        }

        logger.info(f"FocusManager init: completed successfully,: {config}")

    def start(self):
        """Start the service."""
        if self._cooling_thread is not None:
            logger.warning("FocusManager start: already active")
            return

        self._stop_cooling.clear()
        self._cooling_thread = threading.Thread(
            target=self._background_cooling_loop,
            daemon=True,
            name="FocusManager-Cooling",
        )
        self._cooling_thread.start()
        logger.info("FocusManager start: started")

    def stop(self):
        """Stop the service."""
        if self._cooling_thread is None:
            return

        self._stop_cooling.set()
        self._cooling_thread.join(timeout=5.0)
        self._cooling_thread = None
        logger.info("FocusManager stop: stopped")

    def notify_activity(self, user_id: str):
        """Notify activity."""
        with self._lock:

            if user_id not in self.users:
                self._register_new_user(user_id)

            user_info = self.users[user_id]
            user_info.update_activity()

            if self.active_user != user_id:
                self._switch_focus(old_user_id=self.active_user, new_user_id=user_id)
            else:

                if self.config.enable_logging:
                    logger.debug(f"notify activity {user_id} (already active)")

    def _register_new_user(self, user_id: str):

        user_info = UserStateInfo(user_id=user_id, state=UserState.COLD)
        self.users[user_id] = user_info
        logger.info(f"register new user: registered: {user_id}")

    def _switch_focus(self, old_user_id: Optional[str], new_user_id: str):

        logger.info(f"switch focus: switched: {old_user_id} → {new_user_id}")
        self.stats["total_focus_switches"] += 1

        new_user = self.users[new_user_id]
        old_state = new_user.state
        switched = False

        if old_state == UserState.COLD:

            logger.info(f"switch focus {new_user_id}: COLD → ACTIVE( Swap In)")
            if self._trigger_swap_in(new_user_id):
                new_user.change_state(UserState.ACTIVE)
                self.stats["total_swap_ins"] += 1
                switched = True
            else:
                logger.warning(f"switch focus {new_user_id} Swap In failed, COLD")

        elif old_state == UserState.WARM:

            logger.info(f"switch focus {new_user_id}: WARM → ACTIVE( GPU)")
            new_user.change_state(UserState.ACTIVE)
            switched = True

        else:

            logger.warning(f"switch focus {new_user_id} already active ACTIVE")
            switched = True

        if not switched:
            return

        if (
            old_user_id is not None
            and old_user_id in self.users
            and old_user_id != new_user_id
        ):
            old_user = self.users[old_user_id]
            if old_user.state == UserState.ACTIVE:
                old_user.change_state(UserState.WARM)
                logger.info(f"switch focus {old_user_id}: ACTIVE → WARM(started)")

        self.active_user = new_user_id

    def _background_cooling_loop(self):

        logger.info("background cooling loop: started")

        while not self._stop_cooling.is_set():
            try:
                self._check_cooling_transitions()
            except Exception as e:
                logger.error(f"background cooling loop: {e}", exc_info=True)

            self._stop_cooling.wait(timeout=self.config.check_interval_s)

        logger.info("background cooling loop: stopped")

    def _check_cooling_transitions(self):

        candidates = []
        with self._lock:
            for user_id, user_info in self.users.items():
                if user_info.state == UserState.WARM:
                    idle_time = user_info.get_idle_time_seconds()
                    if idle_time >= self.config.cooling_off_period_s:
                        candidates.append(user_id)

        for user_id in candidates:
            logger.info(
                f"check cooling transitions {user_id}: WARM → COLD " f"( Swap Out)"
            )

            if self._trigger_swap_out(user_id):
                with self._lock:
                    user_info = self.users.get(user_id)
                    if user_info and user_info.state == UserState.WARM:
                        user_info.change_state(UserState.COLD)
                        self.stats["total_cooling_transitions"] += 1
                        self.stats["total_swap_outs"] += 1
                    else:
                        logger.warning(
                            f"check cooling transitions {user_id} state,skipped COLD "
                        )
            else:
                logger.warning(
                    f"check cooling transitions {user_id} Swap Out failed, WARM"
                )

    def _trigger_swap_out(self, user_id: str) -> bool:

        if not self.config.enable_swap:
            logger.debug(f"Swap trigger swap out: disabled,skipped {user_id}  Swap Out")
            return True

        if self.swap_out_callback is not None:
            try:
                logger.info(f"trigger swap out {user_id}  Swap Out")
                return bool(self.swap_out_callback(user_id))
            except Exception as e:
                logger.error(
                    f"trigger swap out {user_id}  swap_out_callback : {e}",
                    exc_info=True,
                )
                return False
        else:
            logger.warning(
                f"trigger swap out: registered swap_out_callback,unavailable {user_id}"
            )

        return False

    def _trigger_swap_in(self, user_id: str) -> bool:

        if not self.config.enable_swap:
            logger.debug(f"Swap trigger swap in: disabled,skipped {user_id}  Swap In")
            return True

        if self.swap_in_callback is not None:
            try:
                logger.info(f"trigger swap in {user_id}  Swap In")
                return bool(self.swap_in_callback(user_id))
            except Exception as e:
                logger.error(
                    f"trigger swap in {user_id}  swap_in_callback : {e}", exc_info=True
                )
                return False
        else:
            logger.warning(
                f"trigger swap in: registered swap_in_callback,unavailable {user_id}"
            )

        return False

    def get_user_state(self, user_id: str) -> Optional[UserState]:
        """Return user state."""
        with self._lock:
            if user_id in self.users:
                return self.users[user_id].state
            return None

    def get_active_user(self) -> Optional[str]:
        """Return active user."""
        with self._lock:
            return self.active_user

    def get_all_users(self) -> Set[str]:
        """Return all users."""
        with self._lock:
            return set(self.users.keys())

    def get_users_by_state(self, state: UserState) -> Set[str]:
        """Return users by state."""
        with self._lock:
            return {
                user_id
                for user_id, user_info in self.users.items()
                if user_info.state == state
            }

    def get_statistics(self) -> Dict:
        """Return statistics."""
        with self._lock:
            return {
                **self.stats,
                "total_users": len(self.users),
                "active_users": len(self.get_users_by_state(UserState.ACTIVE)),
                "warm_users": len(self.get_users_by_state(UserState.WARM)),
                "cold_users": len(self.get_users_by_state(UserState.COLD)),
                "current_active_user": self.active_user,
            }

    def force_swap_out(self, user_id: str) -> bool:
        """Force swap out."""
        with self._lock:
            if user_id not in self.users:
                logger.warning(f"force swap out: unavailable: {user_id}")
                return False

            user_info = self.users[user_id]
            if user_info.state == UserState.COLD:
                logger.warning(f"force swap out {user_id} already active COLD,skipped")
                return False

            if not self._trigger_swap_out(user_id):
                logger.warning(f"force swap out {user_id} Swap Out failed,state")
                return False

            user_info.change_state(UserState.COLD)

            if self.active_user == user_id:
                self.active_user = None

            logger.info(f"force swap out: evicted {user_id}")
            return True

    def reset(self):
        """Reset."""
        with self._lock:
            self.users.clear()
            self.active_user = None
            self.stats = {
                "total_focus_switches": 0,
                "total_swap_outs": 0,
                "total_swap_ins": 0,
                "total_cooling_transitions": 0,
            }
            logger.info("FocusManager reset")

    def __repr__(self) -> str:
        return (
            f"FocusManager(users={len(self.users)}, "
            f"active={self.active_user}, "
            f"cooling_period={self.config.cooling_off_period_s}s)"
        )


def create_focus_manager(
    cooling_off_period_s: float = 7.0,
    enable_swap: bool = True,
    enable_logging: bool = True,
    swap_out_callback: Optional[Callable[[str], bool]] = None,
    swap_in_callback: Optional[Callable[[str], bool]] = None,
) -> FocusManager:
    """Create focus manager."""
    config = FocusManagerConfig(
        cooling_off_period_s=cooling_off_period_s,
        enable_swap=enable_swap,
        enable_logging=enable_logging,
    )

    manager = FocusManager(
        config=config,
        swap_out_callback=swap_out_callback,
        swap_in_callback=swap_in_callback,
    )

    manager.start()

    return manager
