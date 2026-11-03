"""ResiMem test group swap module."""

import os
import sys
import time
import unittest
import logging
from pathlib import Path


_root_env = os.environ.get("RESIMEM_ROOT_DIR", "").strip()
project_root = (
    Path(_root_env).expanduser().resolve()
    if _root_env
    else Path(__file__).resolve().parents[2]
)
sys.path.insert(0, str(project_root))

import torch

from core.scheduler.group_swap import (
    GroupSwapConfig,
    GroupSwapManager,
    UserBlockTracker,
    create_group_swap_manager,
)


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TestUserBlockTracker(unittest.TestCase):
    """TestUserBlockTracker implementation."""

    def setUp(self):

        self.tracker = UserBlockTracker()

    def test_register_block(self):

        self.tracker.register_block("user_1", "seq_1", block_id=0, num_tokens=16)
        self.tracker.register_block("user_1", "seq_1", block_id=1, num_tokens=16)
        self.tracker.register_block("user_1", "seq_2", block_id=2, num_tokens=8)

        user_blocks = self.tracker.get_user_blocks("user_1")
        self.assertEqual(user_blocks, {0, 1, 2})

        logger.info("✅ test register block: completed successfully")

    def test_unregister_block(self):

        self.tracker.register_block("user_1", "seq_1", block_id=0)
        self.tracker.register_block("user_1", "seq_1", block_id=1)

        self.tracker.unregister_block(0)

        user_blocks = self.tracker.get_user_blocks("user_1")
        self.assertEqual(user_blocks, {1})

        logger.info("✅ test unregister block: completed successfully")

    def test_multiple_users(self):

        self.tracker.register_block("user_1", "seq_1", block_id=0)
        self.tracker.register_block("user_1", "seq_1", block_id=1)

        self.tracker.register_block("user_2", "seq_2", block_id=2)
        self.tracker.register_block("user_2", "seq_2", block_id=3)

        self.assertEqual(self.tracker.get_user_blocks("user_1"), {0, 1})
        self.assertEqual(self.tracker.get_user_blocks("user_2"), {2, 3})

        logger.info("✅ test multiple users: completed successfully")

    def test_block_info(self):

        self.tracker.register_block("user_1", "seq_1", block_id=0, num_tokens=16)

        info = self.tracker.get_block_info(0)

        self.assertIsNotNone(info)
        self.assertEqual(info.block_id, 0)
        self.assertEqual(info.user_id, "user_1")
        self.assertEqual(info.sequence_id, "seq_1")
        self.assertEqual(info.num_tokens, 16)
        self.assertTrue(info.in_gpu)

        logger.info("✅ test block info: completed successfully")


class TestGroupSwapManager(unittest.TestCase):
    """TestGroupSwapManager implementation."""

    def setUp(self):

        num_blocks = 100
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

        self.manager = GroupSwapManager(
            config=GroupSwapConfig(),
            kv_cache_gpu=self.kv_cache_gpu,
            kv_cache_cpu=self.kv_cache_cpu,
            block_size=block_size,
        )

    def test_swap_out_empty_user(self):

        result = self.manager.swap_out_user("nonexistent_user")

        self.assertIsNone(result)

        logger.info("✅ test swap out empty user: completed successfully")

    def test_swap_out_user(self):

        for i in range(5):
            self.manager.register_block_for_user("user_1", f"seq_{i}", block_id=i)

        metadata = self.manager.swap_out_user("user_1")

        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.user_id, "user_1")
        self.assertEqual(metadata.num_blocks_swapped, 5)
        self.assertTrue(self.manager.is_user_swapped_out("user_1"))

        logger.info(
            f"✅ test swap out user: completed successfully,: {metadata.swap_out_latency_ms:.2f}ms"
        )

    def test_swap_in_user(self):

        for i in range(5):
            self.manager.register_block_for_user("user_1", f"seq_{i}", block_id=i)

        self.manager.swap_out_user("user_1")

        success = self.manager.swap_in_user("user_1")

        self.assertTrue(success)
        self.assertFalse(self.manager.is_user_swapped_out("user_1"))

        logger.info("✅ test swap in user: completed successfully")

    def test_swap_in_without_metadata(self):

        success = self.manager.swap_in_user("nonexistent_user")

        self.assertFalse(success)

        logger.info("✅ test swap in without metadata: completed successfully")

    def test_statistics(self):

        for i in range(10):
            self.manager.register_block_for_user("user_1", f"seq_{i}", block_id=i)

        self.manager.swap_out_user("user_1")
        self.manager.swap_in_user("user_1")

        stats = self.manager.get_statistics()

        self.assertEqual(stats["total_swap_outs"], 1)
        self.assertEqual(stats["total_swap_ins"], 1)
        self.assertEqual(stats["total_blocks_swapped_out"], 10)
        self.assertEqual(stats["total_blocks_swapped_in"], 10)

        logger.info(f"✅ test statistics: completed successfully: {stats}")


class TestGroupSwapManagerPerformance(unittest.TestCase):
    """TestGroupSwapManagerPerformance implementation."""

    def setUp(self):

        num_blocks = 1000
        block_size = 16
        num_heads = 32
        head_dim = 128

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

        self.manager = GroupSwapManager(
            config=GroupSwapConfig(target_latency_ms=100.0),
            kv_cache_gpu=self.kv_cache_gpu,
            kv_cache_cpu=self.kv_cache_cpu,
            block_size=block_size,
        )

    def test_swap_latency(self):

        num_user_blocks = 100
        for i in range(num_user_blocks):
            self.manager.register_block_for_user("user_1", f"seq_{i}", block_id=i)

        # Swap Out
        start = time.time()
        metadata = self.manager.swap_out_user("user_1")
        swap_out_time = (time.time() - start) * 1000

        # Swap In
        start = time.time()
        self.manager.swap_in_user("user_1")
        swap_in_time = (time.time() - start) * 1000

        logger.info(f"Swap Out: {swap_out_time:.2f}ms, Swap In: {swap_in_time:.2f}ms")
        logger.info(f"test swap latency: {num_user_blocks}")

        self.assertLess(swap_out_time, 1000, "Swap Out test swap latency")
        self.assertLess(swap_in_time, 1000, "Swap In test swap latency")

        logger.info("✅ Swap test swap latency: completed successfully")


class TestMultipleUsers(unittest.TestCase):
    """TestMultipleUsers implementation."""

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

        self.manager = GroupSwapManager(
            config=GroupSwapConfig(),
            kv_cache_gpu=self.kv_cache_gpu,
            kv_cache_cpu=self.kv_cache_cpu,
            block_size=block_size,
        )

    def test_multiple_users_swap(self):

        for user_idx in range(3):
            user_id = f"user_{user_idx}"
            for block_idx in range(50):
                block_id = user_idx * 50 + block_idx
                self.manager.register_block_for_user(
                    user_id, f"seq_{block_idx}", block_id
                )

        for user_idx in range(3):
            user_id = f"user_{user_idx}"
            metadata = self.manager.swap_out_user(user_id)
            self.assertIsNotNone(metadata)
            self.assertEqual(metadata.num_blocks_swapped, 50)

        swapped_users = self.manager.get_swapped_out_users()
        self.assertEqual(len(swapped_users), 3)

        for user_idx in range(3):
            user_id = f"user_{user_idx}"
            success = self.manager.swap_in_user(user_id)
            self.assertTrue(success)

        swapped_users = self.manager.get_swapped_out_users()
        self.assertEqual(len(swapped_users), 0)

        logger.info("✅ test multiple users swap Swap completed successfully")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("GroupSwapManager run all tests")
    print("=" * 60 + "\n")

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestUserBlockTracker))
    suite.addTests(loader.loadTestsFromTestCase(TestGroupSwapManager))
    suite.addTests(loader.loadTestsFromTestCase(TestGroupSwapManagerPerformance))
    suite.addTests(loader.loadTestsFromTestCase(TestMultipleUsers))

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
