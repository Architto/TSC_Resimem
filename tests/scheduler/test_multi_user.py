"""ResiMem test multi user module."""

import os
import sys
import time
import unittest
import logging
from pathlib import Path
from unittest.mock import Mock, patch


_root_env = os.environ.get("RESIMEM_ROOT_DIR", "").strip()
project_root = (
    Path(_root_env).expanduser().resolve()
    if _root_env
    else Path(__file__).resolve().parents[2]
)
sys.path.insert(0, str(project_root))

import torch

from core.scheduler.multi_user_types import UserState
from core.scheduler.resimem_scheduler import (
    ResiMemScheduler,
    ResiMemSchedulerConfig,
    create_resimem_scheduler,
)


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TestMultiUserScenario(unittest.TestCase):
    """TestMultiUserScenario implementation."""

    def setUp(self):

        num_blocks = 500
        block_size = 16
        num_heads = 8
        head_dim = 64

        self.kv_cache_gpu = torch.zeros(
            num_blocks,
            block_size,
            num_heads,
            head_dim,
            dtype=torch.float16,
            device="cpu",
        )
        self.kv_cache_cpu = torch.zeros(
            num_blocks,
            block_size,
            num_heads,
            head_dim,
            dtype=torch.float16,
            device="cpu",
        )

        config = ResiMemSchedulerConfig(
            cooling_off_period_s=1.0,
            enable_swap=True,
            enable_logging=True,
        )

        self.scheduler = ResiMemScheduler(
            config=config,
            kv_cache_gpu=self.kv_cache_gpu,
            kv_cache_cpu=self.kv_cache_cpu,
        )
        self.scheduler.start()

    def tearDown(self):

        self.scheduler.stop()

    def test_scheduler_init(self):

        self.assertIsNotNone(self.scheduler.focus_manager)
        self.assertIsNotNone(self.scheduler.group_swap_manager)

        logger.info("✅ test scheduler init: completed successfully")

    def test_user_registration(self):

        self.scheduler.on_user_input("user_1")
        self.scheduler.on_user_input("user_2")
        self.scheduler.on_user_input("user_3")

        all_users = self.scheduler.get_all_users()
        self.assertEqual(len(all_users), 3)
        self.assertIn("user_1", all_users)
        self.assertIn("user_2", all_users)
        self.assertIn("user_3", all_users)

        logger.info("✅ test user registration: completed successfully")

    def test_focus_switching(self):

        self.scheduler.on_user_input("user_1")
        self.assertEqual(self.scheduler.get_active_user(), "user_1")
        self.assertEqual(self.scheduler.get_user_state("user_1"), UserState.ACTIVE)

        self.scheduler.on_user_input("user_2")
        self.assertEqual(self.scheduler.get_active_user(), "user_2")
        self.assertEqual(self.scheduler.get_user_state("user_2"), UserState.ACTIVE)
        self.assertEqual(self.scheduler.get_user_state("user_1"), UserState.WARM)

        logger.info("✅ test focus switching: completed successfully")

    def test_three_users_rotation(self):

        logger.info("test three users rotation 3 ...")

        for i in range(1, 4):
            self.scheduler.on_user_input(f"user_{i}")
            time.sleep(0.1)

        self.assertEqual(self.scheduler.get_active_user(), "user_3")

        self.scheduler.on_user_input("user_1")
        self.assertEqual(self.scheduler.get_active_user(), "user_1")

        stats = self.scheduler.get_statistics()
        focus_stats = stats["focus_manager"]

        self.assertEqual(focus_stats["total_users"], 3)
        self.assertGreater(focus_stats["total_focus_switches"], 0)

        logger.info(f"test three users rotation: statistics: {focus_stats}")
        logger.info("✅ 3 test three users rotation: completed successfully")

    def test_cooling_and_swap_out(self):

        for i in range(10):
            self.scheduler.register_block("user_1", f"seq_{i}", block_id=i)

        self.scheduler.on_user_input("user_1")

        self.scheduler.on_user_input("user_2")
        self.assertEqual(self.scheduler.get_user_state("user_1"), UserState.WARM)

        logger.info("test cooling and swap out: waiting...")
        time.sleep(1.5)

        self.scheduler.focus_manager._check_cooling_transitions()

        self.assertEqual(self.scheduler.get_user_state("user_1"), UserState.COLD)

        stats = self.scheduler.get_statistics()
        self.assertGreaterEqual(stats["focus_manager"]["total_swap_outs"], 1)

        logger.info("✅ test cooling and swap out Swap Out completed successfully")

    def test_swap_in_on_return(self):

        for i in range(10):
            self.scheduler.register_block("user_1", f"seq_{i}", block_id=i)

        self.scheduler.on_user_input("user_1")

        self.scheduler.on_user_input("user_2")

        time.sleep(1.5)
        self.scheduler.focus_manager._check_cooling_transitions()
        self.assertEqual(self.scheduler.get_user_state("user_1"), UserState.COLD)

        self.scheduler.on_user_input("user_1")

        self.assertEqual(self.scheduler.get_user_state("user_1"), UserState.ACTIVE)

        stats = self.scheduler.get_statistics()
        self.assertGreaterEqual(stats["focus_manager"]["total_swap_ins"], 1)

        logger.info("✅ Swap In test swap in on return: completed successfully")


class TestFiveUsersScenario(unittest.TestCase):
    """TestFiveUsersScenario implementation."""

    def setUp(self):

        num_blocks = 1000
        block_size = 16
        num_heads = 8
        head_dim = 64

        self.kv_cache_gpu = torch.zeros(
            num_blocks,
            block_size,
            num_heads,
            head_dim,
            dtype=torch.float16,
            device="cpu",
        )
        self.kv_cache_cpu = torch.zeros(
            num_blocks,
            block_size,
            num_heads,
            head_dim,
            dtype=torch.float16,
            device="cpu",
        )

        config = ResiMemSchedulerConfig(
            cooling_off_period_s=0.5,
            enable_swap=True,
        )

        self.scheduler = ResiMemScheduler(
            config=config,
            kv_cache_gpu=self.kv_cache_gpu,
            kv_cache_cpu=self.kv_cache_cpu,
        )
        self.scheduler.start()

    def tearDown(self):

        self.scheduler.stop()

    def test_five_users_rotation(self):

        logger.info("test five users rotation 5 ...")

        blocks_per_user = 50
        for user_idx in range(5):
            user_id = f"user_{user_idx}"
            for block_idx in range(blocks_per_user):
                block_id = user_idx * blocks_per_user + block_idx
                self.scheduler.register_block(user_id, f"seq_{block_idx}", block_id)

        num_rounds = 3
        for round_idx in range(num_rounds):
            logger.info(f"test five users rotation {round_idx + 1} ...")
            for user_idx in range(5):
                user_id = f"user_{user_idx}"
                self.scheduler.on_user_input(user_id)

                time.sleep(0.2)

        time.sleep(1.0)

        stats = self.scheduler.get_statistics()
        logger.info(f"test five users rotation: statistics: {stats}")

        self.assertGreater(
            stats["focus_manager"]["total_focus_switches"],
            0,
            "test five users rotation: switched",
        )

        logger.info("✅ 5 test five users rotation: completed successfully")

    def test_rapid_switching(self):

        logger.info("test rapid switching: switched...")

        for _ in range(20):
            for user_idx in range(5):
                self.scheduler.on_user_input(f"user_{user_idx}")

        stats = self.scheduler.get_statistics()
        focus_stats = stats["focus_manager"]

        logger.info(f"test rapid switching: switched: {focus_stats}")

        self.assertEqual(focus_stats["total_users"], 5)

        logger.info("✅ test rapid switching: completed successfully")


class TestMemoryPressure(unittest.TestCase):
    """TestMemoryPressure implementation."""

    def setUp(self):

        num_blocks = 200
        block_size = 16
        num_heads = 8
        head_dim = 64

        self.kv_cache_gpu = torch.zeros(
            num_blocks,
            block_size,
            num_heads,
            head_dim,
            dtype=torch.float16,
            device="cpu",
        )
        self.kv_cache_cpu = torch.zeros(
            num_blocks,
            block_size,
            num_heads,
            head_dim,
            dtype=torch.float16,
            device="cpu",
        )

        self.scheduler = create_resimem_scheduler(
            cooling_off_period_s=0.5,
            enable_swap=True,
            kv_cache_gpu=self.kv_cache_gpu,
            kv_cache_cpu=self.kv_cache_cpu,
            auto_start=True,
        )

    def tearDown(self):

        self.scheduler.stop()

    def test_handle_memory_pressure_with_warm_users(self):

        for i in range(10):
            self.scheduler.register_block("user_1", f"seq_{i}", block_id=i)

        self.scheduler.on_user_input("user_1")
        self.scheduler.on_user_input("user_2")

        self.assertEqual(self.scheduler.get_user_state("user_1"), UserState.WARM)

        success = self.scheduler.handle_memory_pressure(needed_blocks=5)

        self.assertTrue(success)

        self.assertEqual(self.scheduler.get_user_state("user_1"), UserState.COLD)

        logger.info(
            "✅ test handle memory pressure with warm users: completed successfully"
        )

    def test_three_layer_memory_pressure_cascade_order(self):

        call_order = []

        def _mock_l1(needed_blocks: int) -> bool:
            call_order.append("L1")
            return False

        def _mock_l2(needed_blocks: int) -> bool:
            call_order.append("L2")
            return False

        def _mock_l3(needed_blocks: int) -> bool:
            call_order.append("L3")
            return True

        self.scheduler._try_l1_swap_out = _mock_l1
        self.scheduler._try_l2_dnn_degrade = _mock_l2
        self.scheduler._try_l3_sandwich_eviction = _mock_l3

        success = self.scheduler.handle_memory_pressure(needed_blocks=8)

        self.assertTrue(success)
        self.assertEqual(call_order, ["L1", "L2", "L3"])

        logger.info(
            "✅ test three layer memory pressure cascade order: completed successfully"
        )

    def test_three_layer_stops_after_l2_success(self):

        call_order = []

        def _mock_l1(needed_blocks: int) -> bool:
            call_order.append("L1")
            return False

        def _mock_l2(needed_blocks: int) -> bool:
            call_order.append("L2")
            return True

        def _mock_l3(needed_blocks: int) -> bool:
            call_order.append("L3")
            return True

        self.scheduler._try_l1_swap_out = _mock_l1
        self.scheduler._try_l2_dnn_degrade = _mock_l2
        self.scheduler._try_l3_sandwich_eviction = _mock_l3

        success = self.scheduler.handle_memory_pressure(needed_blocks=6)

        self.assertTrue(success)
        self.assertEqual(call_order, ["L1", "L2"])

        logger.info(
            "✅ L2 test three layer stops after l2 success: completed successfully"
        )

    def test_l2_dnn_callback_updates_statistics(self):

        self.scheduler.dnn_degrade_callback = lambda needed_blocks: True

        before = self.scheduler.stats["l2_degradations"]
        success = self.scheduler._try_l2_dnn_degrade(needed_blocks=4)
        after = self.scheduler.stats["l2_degradations"]

        self.assertTrue(success)
        self.assertEqual(after, before + 1)

        logger.info(
            "✅ L2 test l2 dnn callback updates statistics: completed successfully"
        )


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("run all tests")
    print("=" * 60 + "\n")

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestMultiUserScenario))
    suite.addTests(loader.loadTestsFromTestCase(TestFiveUsersScenario))
    suite.addTests(loader.loadTestsFromTestCase(TestMemoryPressure))

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
