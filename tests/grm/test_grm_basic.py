"""ResiMem test grm basic module."""

import pytest
import time
import sys
from pathlib import Path


project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.grm import GRMServer, GRMClient, GRMConfig
from core.ipc import ExpansionRequest, WorkerStatus


class TestGRMBasic:
    """TestGRMBasic implementation."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup teardown."""

        try:
            from multiprocessing import shared_memory

            old_shm = shared_memory.SharedMemory(name="resimem_grm")
            old_shm.close()
            old_shm.unlink()
        except:
            pass

        yield

        time.sleep(0.1)

    def test_grm_server_startup(self):

        print("\n" + "=" * 60)
        print("test grm server startup: GRM Server started")
        print("=" * 60)

        grm = GRMServer(GRMConfig(total_gpu_memory_gb=24.0))

        grm.start_in_background()
        time.sleep(0.2)

        assert grm.is_running(), "GRM should be running"
        print("✅ GRM Server started successfully")

        grm.stop()
        time.sleep(0.1)

        assert not grm.is_running(), "GRM should be stopped"
        print("✅ GRM Server stopped successfully")

    def test_vllm_worker_registration(self):

        print("\n" + "=" * 60)
        print("test vllm worker registration: vLLM Worker registered")
        print("=" * 60)

        grm = GRMServer()
        grm.start_in_background()
        time.sleep(0.2)

        try:

            client = GRMClient("vllm_worker_0", worker_type="vllm")
            success = client.register()

            assert success, "Registration should succeed"
            print("✅ vLLM Worker registered")

            time.sleep(0.3)

            assert "vllm_worker_0" in grm.vllm_workers, "Worker should be in GRM"
            print(f"✅ GRM confirmed registration")
            print(f"   Worker ID: {grm.vllm_workers['vllm_worker_0'].worker_id}")
            print(f"   Status: {grm.vllm_workers['vllm_worker_0'].status}")

            client.unregister()
            client.close()

        finally:
            grm.stop()

    def test_multiple_workers_registration(self):

        print("\n" + "=" * 60)
        print("test multiple workers registration:  Worker registered")
        print("=" * 60)

        grm = GRMServer()
        grm.start_in_background()
        time.sleep(0.2)

        try:

            clients = []
            for i in range(3):
                client = GRMClient(f"vllm_worker_{i}", worker_type="vllm")
                success = client.register()
                assert success, f"Worker {i} registration failed"
                clients.append(client)
                print(f"✅ Registered vLLM Worker {i}")

            time.sleep(0.3)

            assert len(grm.vllm_workers) == 3, "Should have 3 vLLM workers"
            print(f"✅ GRM has {len(grm.vllm_workers)} vLLM workers")

            for client in clients:
                client.unregister()
                client.close()

        finally:
            grm.stop()

    def test_expansion_with_free_memory(self):

        print("\n" + "=" * 60)
        print("test expansion with free memory: expanded")
        print("=" * 60)

        grm = GRMServer(GRMConfig(total_gpu_memory_gb=24.0))
        grm.start_in_background()
        time.sleep(0.5)

        try:

            client = GRMClient("vllm_worker_0", worker_type="vllm")
            client.register()
            time.sleep(1.0)

            print(f"📊 GRM has {len(grm.vllm_workers)} vLLM workers registered")
            assert "vllm_worker_0" in grm.vllm_workers, "Worker should be registered"

            request = ExpansionRequest(
                num_blocks=100,
                size_bytes=1 * 1024**3,  # 1 GB
                priority=5,
                timeout_ms=2000,
            )

            print(f"📤 Requesting expansion: {request.size_bytes / 1024**3:.2f} GB")

            approved = client.request_expansion(request)

            time.sleep(0.5)

            print(f"📈 GRM Stats: {grm.get_stats()}")
            assert (
                approved
            ), f"Expansion should be approved when memory is available (stats: {grm.get_stats()})"
            print("✅ Expansion approved by GRM")
            print(f"   Total requests: {grm.stats['total_expansion_requests']}")
            print(f"   Approved: {grm.stats['approved_expansions']}")

            client.unregister()
            client.close()

        finally:
            grm.stop()

    def test_expansion_without_free_memory(self):

        print("\n" + "=" * 60)
        print(
            "test expansion without free memory: insufficient capacity DNN Worker expanded"
        )
        print("=" * 60)

        grm = GRMServer(GRMConfig(total_gpu_memory_gb=2.0, min_free_memory_gb=0.5))
        grm.start_in_background()
        time.sleep(0.2)

        try:

            client = GRMClient(
                "vllm_worker_0", worker_type="vllm", heartbeat_interval_s=0.2
            )
            client.register()
            client.update_memory_usage(1.0)
            time.sleep(1.5)

            request = ExpansionRequest(
                num_blocks=150,
                size_bytes=int(1.5 * 1024**3),  # 1.5 GB
                priority=5,
                timeout_ms=200,
            )

            print(f"📤 Requesting expansion: {request.size_bytes / 1024**3:.2f} GB")
            print(
                f"   Available memory: ~0.5 GB (2GB total - 1GB used - 0.5GB reserved)"
            )

            approved = client.request_expansion(request)

            assert not approved, "Expansion should be denied when no memory available"
            print("✅ Expansion correctly denied (no DNN worker to release memory)")
            print(f"   Total requests: {grm.stats['total_expansion_requests']}")
            print(f"   Denied: {grm.stats['denied_expansions']}")

            client.unregister()
            client.close()

        finally:
            grm.stop()

    def test_heartbeat_mechanism(self):

        print("\n" + "=" * 60)
        print("test heartbeat mechanism: ")
        print("=" * 60)

        grm = GRMServer(GRMConfig(heartbeat_timeout_s=2.0))
        grm.start_in_background()
        time.sleep(0.2)

        try:

            client = GRMClient(
                "vllm_worker_0", worker_type="vllm", heartbeat_interval_s=0.5
            )
            client.register()
            time.sleep(0.3)

            assert "vllm_worker_0" in grm.vllm_workers, "Worker should be registered"
            print("✅ Worker registered and sending heartbeats")

            initial_heartbeat = grm.vllm_workers["vllm_worker_0"].last_heartbeat

            time.sleep(1.0)

            updated_heartbeat = grm.vllm_workers["vllm_worker_0"].last_heartbeat
            assert updated_heartbeat > initial_heartbeat, "Heartbeat should be updated"
            print(
                f"✅ Heartbeat updated (Δt = {updated_heartbeat - initial_heartbeat:.2f}s)"
            )

            print("⏰ Stopping heartbeat, waiting for timeout...")
            client._stop_heartbeat()
            time.sleep(3.0)

            assert (
                "vllm_worker_0" not in grm.vllm_workers
            ), "Worker should be removed after timeout"
            print("✅ Worker correctly removed after heartbeat timeout")

            client.close()

        finally:
            grm.stop()

    def test_grm_statistics(self):

        print("\n" + "=" * 60)
        print("test grm statistics: GRM statistics")
        print("=" * 60)

        grm = GRMServer()
        grm.start_in_background()
        time.sleep(0.5)

        try:

            client = GRMClient("vllm_worker_0", worker_type="vllm")
            client.register()
            time.sleep(1.0)

            print(f"📊 Worker registered: {len(grm.vllm_workers)} workers")

            for i in range(3):
                print(f"  📤 Sending expansion request {i+1}/3...")
                request = ExpansionRequest(
                    num_blocks=50, size_bytes=int(0.5 * 1024**3), timeout_ms=2000
                )
                approved = client.request_expansion(request)
                print(
                    f"     Result: {'✅ Approved' if approved else '❌ Denied/Timeout'}"
                )
                time.sleep(0.1)

            time.sleep(0.5)

            stats = grm.get_stats()

            print("\n📊 GRM Statistics:")
            print(f"   Total Expansion Requests: {stats['total_expansion_requests']}")
            print(f"   Approved: {stats['approved_expansions']}")
            print(f"   Denied: {stats['denied_expansions']}")
            print(f"   DNN Releases: {stats['dnn_releases']}")
            print(f"   vLLM Workers: {stats['num_vllm_workers']}")
            print(f"   DNN Workers: {stats['num_dnn_workers']}")

            assert (
                stats["total_expansion_requests"] == 3
            ), f"Should have 3 requests, got {stats['total_expansion_requests']}"
            assert (
                stats["approved_expansions"] == 3
            ), f"All 3 requests should be approved, got {stats['approved_expansions']}"
            print("✅ Statistics tracking working correctly")

            client.unregister()
            client.close()

        finally:
            grm.stop()


if __name__ == "__main__":

    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    pytest.main([__file__, "-v", "-s"])
