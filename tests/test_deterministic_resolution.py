"""Tests for deterministic step resolution ordering during replay.

During replay, concurrent steps should resolve in function ID order to ensure
deterministic behavior, even though asyncio scheduling is non-deterministic.
"""

import asyncio
import uuid
from typing import List

import pytest

from dbos import DBOS, SetWorkflowID


@pytest.mark.asyncio
async def test_resolution_order_during_replay(dbos: DBOS) -> None:
    """Test that concurrent steps resolve in function ID order during replay.

    This test creates a workflow with multiple concurrent steps that track their
    resolution order. During replay, we expect steps to resolve in the order of
    their function IDs (which are assigned based on initiation order), regardless
    of asyncio scheduling.

    The key insight is that during replay, the step function body doesn't execute -
    only the await returns. So we track when the await completes, not when the
    function body runs.
    """
    resolution_order: List[int] = []

    @DBOS.step()
    async def tracked_step(step_num: int) -> int:
        """A step that returns its step_num."""
        return step_num

    @DBOS.workflow()
    async def concurrent_workflow() -> List[int]:
        """Workflow that runs multiple steps concurrently and tracks resolution order."""
        resolution_order.clear()

        async def track_resolution(step_num: int) -> int:
            """Wrapper that tracks when the step's await completes."""
            result = await tracked_step(step_num)
            # This runs when the await completes (both fresh and replay)
            resolution_order.append(step_num)
            return result

        # Create tasks in order 0, 1, 2, 3, 4
        # Function IDs will be assigned as 1, 2, 3, 4, 5 (1-indexed)
        tasks = [asyncio.create_task(track_resolution(i)) for i in range(5)]
        results = await asyncio.gather(*tasks)
        return list(results)

    # First run - execute fresh, order may vary based on asyncio scheduling
    wfid = str(uuid.uuid4())
    with SetWorkflowID(wfid):
        first_results = await concurrent_workflow()

    # Results should contain all values regardless of order
    assert sorted(first_results) == [0, 1, 2, 3, 4]
    first_resolution_order = resolution_order.copy()

    # Replay - should be deterministic with steps resolving in function ID order
    # Function IDs are assigned in task creation order: step(0)->1, step(1)->2, etc.
    # So during replay, step_num 0 (func_id 1) should resolve first, then 1, etc.
    resolution_order.clear()
    with SetWorkflowID(wfid):
        replay_results = await concurrent_workflow()

    # Verify results are the same
    assert replay_results == first_results

    # The key assertion: during replay, resolution order should match function ID order
    # Step with step_num=0 has function_id=1 and should resolve first, etc.
    assert resolution_order == [0, 1, 2, 3, 4], (
        f"Expected resolution order [0, 1, 2, 3, 4] during replay, "
        f"but got {resolution_order}. Steps should resolve in function ID order. "
        f"(First run order was {first_resolution_order})"
    )


@pytest.mark.asyncio
async def test_fresh_execution_not_constrained(dbos: DBOS) -> None:
    """Verify that fresh execution order is NOT artificially constrained.

    Fresh execution should complete steps as fast as possible without waiting
    for ordering. This test uses varying delays to show that faster steps
    can complete before slower ones during fresh execution.
    """
    resolution_order: List[int] = []

    @DBOS.step()
    async def delayed_step(step_num: int, delay: float) -> int:
        """A step with configurable delay."""
        await asyncio.sleep(delay)
        resolution_order.append(step_num)
        return step_num

    @DBOS.workflow()
    async def concurrent_workflow_with_delays() -> List[int]:
        """Workflow with steps that have different delays."""
        resolution_order.clear()
        # Create tasks with delays: step 0 is slowest, step 4 is fastest
        tasks = [asyncio.create_task(delayed_step(i, 0.1 - i * 0.02)) for i in range(5)]
        results = await asyncio.gather(*tasks)
        return list(results)

    # Fresh execution - steps should complete based on their delays
    wfid = str(uuid.uuid4())
    with SetWorkflowID(wfid):
        results = await concurrent_workflow_with_delays()

    assert sorted(results) == [0, 1, 2, 3, 4]

    # During fresh execution, faster steps (higher step_num) should complete first
    # Step 4 has delay 0.02, step 0 has delay 0.1
    # So we expect something like [4, 3, 2, 1, 0] or close to it
    # We just verify it's NOT in creation order to show we're not constraining fresh execution
    # Note: This is probabilistic but with these delays should be reliable
    assert resolution_order != [0, 1, 2, 3, 4], (
        "Fresh execution appears to be constrained to function ID order. "
        f"Got {resolution_order} which matches creation order despite varying delays."
    )
