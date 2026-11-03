"""ResiMem test focus manager module."""

import os
import sys
import time
import unittest
import logging
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, MagicMock


_root_env = os.environ.get("RESIMEM_ROOT_DIR", "").strip()
project_root = (
    Path(_root_env).expanduser().resolve()
    if _root_env
    else Path(__file__).resolve().parents[2]
)
sys.path.insert(0, str(project_root))

from core.scheduler.multi_user_types import UserState, FocusManagerConfig
from core.scheduler.focus_manager import FocusManager, create_focus_manager


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TestFocusManagerBasic(unittest.TestCase):
    """TestFocusManagerBasic implementation."""

    def setUp(self):

        self.config = FocusManagerConfig(
            cooling_off_period_s=2.0,
            check_interval_s=0.5,
            enable_swap=True,
            enable_logging=True,
        )
        self.swap_out_mock = Mock(return_value=True)
        self.swap_in_mock = Mock(return_value=True)

        self.fm = FocusManager(
            config=self.config,
            swap_out_callback=self.swap_out_mock,
            swap_in_callback=self.swap_in_mock,
        )

    def tearDown(self):

        self.fm.stop()

    def test_init(self):

        self.assertEqual(len(self.fm.users), 0)
        self.assertIsNone(self.fm.active_user)
        logger.info("✅ test init: completed successfully")

    def test_register_new_user(self):

        self.fm.notify_activity("user_1")

        self.assertEqual(len(self.fm.users), 1)
        self.assertIn("user_1", self.fm.users)
        self.assertEqual(self.fm.active_user, "user_1")
        self.assertEqual(self.fm.get_user_state("user_1"), UserState.ACTIVE)
        logger.info("✅ test register new user: completed successfully")

    def test_focus_switch(self):

        self.fm.notify_activity("user_1")
        self.assertEqual(self.fm.active_user, "user_1")
        self.assertEqual(self.fm.get_user_state("user_1"), UserState.ACTIVE)

        self.fm.notify_activity("user_2")
        self.assertEqual(self.fm.active_user, "user_2")
        self.assertEqual(self.fm.get_user_state("user_2"), UserState.ACTIVE)
        self.assertEqual(self.fm.get_user_state("user_1"), UserState.WARM)

        logger.info("✅ test focus switch: completed successfully")

    def test_same_user_no_switch(self):

        self.fm.notify_activity("user_1")
        self.fm.notify_activity("user_1")
        self.fm.notify_activity("user_1")

        stats = self.fm.get_statistics()
        self.assertEqual(stats["total_focus_switches"], 1)
        logger.info("✅ test same user no switch: completed successfully")

    def test_warm_to_active_no_swap(self):

        self.fm.notify_activity("user_1")
        self.fm.notify_activity("user_2")

        self.assertEqual(self.fm.get_user_state("user_1"), UserState.WARM)

        self.swap_in_mock.reset_mock()
        self.fm.notify_activity("user_1")

        self.swap_in_mock.assert_not_called()
        self.assertEqual(self.fm.get_user_state("user_1"), UserState.ACTIVE)

        logger.info(
            "✅ WARM → ACTIVE test warm to active no swap Swap completed successfully"
        )


class TestFocusManagerCooling(unittest.TestCase):
    """TestFocusManagerCooling implementation."""

    def setUp(self):

        self.config = FocusManagerConfig(
            cooling_off_period_s=1.0,
            check_interval_s=0.2,
            enable_swap=True,
            enable_logging=True,
        )
        self.swap_out_mock = Mock(return_value=True)
        self.swap_in_mock = Mock(return_value=True)

        self.fm = FocusManager(
            config=self.config,
            swap_out_callback=self.swap_out_mock,
            swap_in_callback=self.swap_in_mock,
        )
        self.fm.start()

    def tearDown(self):

        self.fm.stop()

    def test_cooling_transition(self):

        self.fm.notify_activity("user_1")

        self.fm.notify_activity("user_2")
        self.assertEqual(self.fm.get_user_state("user_1"), UserState.WARM)

        logger.info("test cooling transition: waiting...")
        time.sleep(1.5)

        self.assertEqual(self.fm.get_user_state("user_1"), UserState.COLD)

        self.swap_out_mock.assert_called_once_with("user_1")

        logger.info("✅ test cooling transition: completed successfully")

    def test_activity_prevents_cooling(self):

        self.fm.notify_activity("user_1")

        self.fm.notify_activity("user_2")

        time.sleep(0.5)
        self.fm.notify_activity("user_1")

        self.assertEqual(self.fm.get_user_state("user_1"), UserState.ACTIVE)

        self.swap_out_mock.assert_not_called()

        logger.info("✅ test activity prevents cooling: completed successfully")


class TestFocusManagerSwapCallbacks(unittest.TestCase):
    """TestFocusManagerSwapCallbacks implementation."""

    def setUp(self):

        self.config = FocusManagerConfig(
            cooling_off_period_s=1.0,
            check_interval_s=0.2,
            enable_swap=True,
            enable_logging=True,
        )
        self.swap_out_mock = Mock(return_value=True)
        self.swap_in_mock = Mock(return_value=True)

        self.fm = FocusManager(
            config=self.config,
            swap_out_callback=self.swap_out_mock,
            swap_in_callback=self.swap_in_mock,
        )
        self.fm.start()

    def tearDown(self):

        self.fm.stop()

    def test_swap_in_on_cold_user(self):

        self.fm.notify_activity("user_1")

        self.fm.notify_activity("user_2")

        time.sleep(1.5)
        self.assertEqual(self.fm.get_user_state("user_1"), UserState.COLD)

        self.swap_in_mock.reset_mock()
        self.fm.notify_activity("user_1")

        self.swap_in_mock.assert_called_once_with("user_1")

        logger.info(
            "✅ COLD → ACTIVE Swap In test swap in on cold user: completed successfully"
        )

    def test_force_swap_out(self):

        self.fm.notify_activity("user_1")

        result = self.fm.force_swap_out("user_1")

        self.assertTrue(result)
        self.assertEqual(self.fm.get_user_state("user_1"), UserState.COLD)
        self.swap_out_mock.assert_called_with("user_1")

        logger.info("✅ test force swap out Swap Out completed successfully")

    def test_swap_failure_and_success_paths(self):

        self.swap_in_mock.side_effect = lambda user_id: user_id != "user_fail"
        self.fm.notify_activity("user_ok")
        self.assertEqual(self.fm.get_active_user(), "user_ok")
        self.assertEqual(self.fm.get_user_state("user_ok"), UserState.ACTIVE)

        self.fm.notify_activity("user_fail")
        self.assertEqual(self.fm.get_active_user(), "user_ok")
        self.assertEqual(self.fm.get_user_state("user_fail"), UserState.COLD)

        self.swap_out_mock.side_effect = lambda user_id: user_id != "user_ok"
        self.fm.notify_activity("user_warm")
        self.assertEqual(self.fm.get_active_user(), "user_warm")
        self.assertEqual(self.fm.get_user_state("user_ok"), UserState.WARM)

        self.fm.users["user_ok"].last_activity_time = datetime.now() - timedelta(
            seconds=5
        )
        self.fm._check_cooling_transitions()
        self.assertEqual(self.fm.get_user_state("user_ok"), UserState.WARM)

        self.swap_out_mock.side_effect = lambda user_id: True
        self.fm.users["user_ok"].last_activity_time = datetime.now() - timedelta(
            seconds=5
        )
        self.fm._check_cooling_transitions()
        self.assertEqual(self.fm.get_user_state("user_ok"), UserState.COLD)

        self.swap_in_mock.side_effect = lambda user_id: True
        self.fm.notify_activity("user_ok")
        self.assertEqual(self.fm.get_active_user(), "user_ok")
        self.assertEqual(self.fm.get_user_state("user_ok"), UserState.ACTIVE)

        logger.info(
            "✅ Swap test swap failure and success paths: failed/completed successfully"
        )


class TestFocusManagerStatistics(unittest.TestCase):
    """TestFocusManagerStatistics implementation."""

    def setUp(self):

        self.config = FocusManagerConfig(
            cooling_off_period_s=1.0,
            check_interval_s=0.2,
            enable_swap=True,
        )
        self.swap_in_mock = Mock(return_value=True)
        self.fm = FocusManager(config=self.config, swap_in_callback=self.swap_in_mock)
        self.fm.start()

    def tearDown(self):

        self.fm.stop()

    def test_statistics_tracking(self):

        self.fm.notify_activity("user_1")
        self.fm.notify_activity("user_2")
        self.fm.notify_activity("user_3")

        stats = self.fm.get_statistics()

        self.assertEqual(stats["total_users"], 3)
        self.assertEqual(stats["total_focus_switches"], 3)
        self.assertEqual(stats["current_active_user"], "user_3")

        logger.info(f"✅ test statistics tracking: statistics: {stats}")

    def test_users_by_state(self):

        self.fm.notify_activity("user_1")
        self.fm.notify_activity("user_2")
        self.fm.notify_activity("user_3")

        active_users = self.fm.get_users_by_state(UserState.ACTIVE)
        warm_users = self.fm.get_users_by_state(UserState.WARM)

        self.assertEqual(active_users, {"user_3"})
        self.assertEqual(warm_users, {"user_1", "user_2"})

        logger.info("✅ test users by state: completed successfully")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("FocusManager run all tests")
    print("=" * 60 + "\n")

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestFocusManagerBasic))
    suite.addTests(loader.loadTestsFromTestCase(TestFocusManagerCooling))
    suite.addTests(loader.loadTestsFromTestCase(TestFocusManagerSwapCallbacks))
    suite.addTests(loader.loadTestsFromTestCase(TestFocusManagerStatistics))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("✅ run all tests: completed successfully!")
    else:
        print(
            f"❌ run all tests: failed: {len(result.failures)}, error: {len(result.errors)}"
        )
    print("=" * 60)

    return result.wasSuccessful()


if __name__ == "__main__":
    run_all_tests()
