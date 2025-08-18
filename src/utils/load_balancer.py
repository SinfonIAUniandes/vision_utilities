from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from typing import Callable, Generic, List, Tuple, Dict, Optional, TypeVar
from math import lcm

T = TypeVar("T")

@dataclass(frozen=True)
class Subscriber(Generic[T]):
    callback: Callable[[T], None]
    wait: int  # w_j >= 1


class LoadBalancer(Generic[T]):
    def __init__(self) -> None:
        self.subs: List[Subscriber[T]] = []

    def subscribe(self, callback: Callable[[T], None], wait_turns_for_call: int) -> None:
        if wait_turns_for_call < 1:
            raise ValueError("wait_turns_for_call must be >= 1")
        self.subs.append(Subscriber(callback=callback, wait=wait_turns_for_call))

    # ---------- Scheduling core (DP + min–max via binary search) ----------

    def _round_calls(self, H: int) -> List[int]:
        # k_j = round(H / w_j)
        ks = []
        for s in self.subs:
            f = H / s.wait
            k = int(f // 1)
            if f - k >= 0.5:
                k += 1
            ks.append(k)
        return ks

    def _feasible(self, H: int, M: int, K_target: Tuple[int, ...]) -> Optional[List[List[int]]]:
        W = tuple(s.wait for s in self.subs)
        N = len(W)
        R0 = sum(K_target)

        # Precompute per-subscriber upper bound of calls achievable from any state for quick pruning
        @lru_cache(maxsize=None)
        def max_calls_from_state(cool: Tuple[int, ...], remain: Tuple[int, ...], turns_left: int) -> int:
            total = 0
            for j in range(N):
                r = remain[j]
                if r <= 0:
                    continue
                w = W[j]
                c = max(cool[j], 0)
                # First opportunity in [0..]
                first = 0 if c == 0 else c
                if first >= turns_left:
                    continue
                # Greedy upper bound: 1 call at `first`, then every w turns
                possible = 1 + (turns_left - 1 - first) // w
                total += min(possible, r)
            return total

        # DP with memoization; returns whether feasible and stores one witness via backpointers
        @lru_cache(maxsize=None)
        def F(i: int, cool: Tuple[int, ...], remain: Tuple[int, ...]) -> bool:
            turns_left = H - i
            rem_total = sum(remain)
            if rem_total == 0:
                return True
            if turns_left <= 0:
                return False
            # Global pruning: even with best packing, cannot finish
            if max_calls_from_state(cool, remain, turns_left) < rem_total:
                return False

            # Eligible set this turn
            eligible = [j for j in range(N) if cool[j] == 0 and remain[j] > 0]
            if not eligible:
                # Advance time without calling anyone; decrease cooldowns
                new_cool = tuple(max(c - 1, 0) for c in cool)
                return F(i + 1, new_cool, remain)

            # We may call up to M this turn
            max_take = min(M, len(eligible))

            # Heuristic: try larger groups first to meet remaining demand
            # Lower bound on how many we must take now:
            # If we take t now, the theoretical max future including now is computed inside max_calls_from_state,
            # so we derive a crude bound by trying t from max_take down to 0.
            # We also prioritize subscribers with largest remaining quotas.
            eligible_sorted = sorted(eligible, key=lambda j: (-remain[j], -W[j]))

            for take in range(max_take, -1, -1):
                # combinations over the sorted list to improve early success
                for U in combinations(eligible_sorted, take):
                    # Apply transition
                    cool_next = list(cool)
                    remain_next = list(remain)
                    for j in range(N):
                        if j in U:
                            cool_next[j] = W[j] - 1 if W[j] > 0 else 0
                            remain_next[j] -= 1
                        else:
                            cool_next[j] = max(cool_next[j] - 1, 0)
                    cool_next = tuple(cool_next)
                    remain_next = tuple(remain_next)

                    # Prune: if even after this choice the upper bound fails, skip
                    if max_calls_from_state(cool_next, remain_next, turns_left - 1) < sum(remain_next):
                        continue

                    if F(i + 1, cool_next, remain_next):
                        parent[(i, cool, remain)] = (U, cool_next, remain_next)
                        return True
            return False

        # Reconstruct schedule if feasible
        parent: Dict[Tuple[int, Tuple[int, ...], Tuple[int, ...]], Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]] = {}
        cool0 = tuple(0 for _ in W)
        remain0 = tuple(K_target)

        ok = F(0, cool0, remain0)
        if not ok:
            return None

        # Build per-turn list of indices called
        schedule: List[List[int]] = [[] for _ in range(H)]
        i, cool, remain = 0, cool0, remain0
        while i < H and sum(remain) > 0:
            key = (i, cool, remain)
            if key not in parent:
                # No calls this turn; advance cooldowns
                for j in range(N):
                    cool = tuple(max(c - 1, 0) if idx == j else c for idx, c in enumerate(cool))
                i += 1
                continue
            U, cool_next, remain_next = parent[key]
            schedule[i] = list(U)
            i += 1
            cool, remain = cool_next, remain_next

        # Fill trailing turns with empty lists
        for t in range(i, H):
            schedule[t] = []
        return schedule

    def _minimize_peak(self, H: int, counts: List[int]) -> List[List[int]]:
        N = len(self.subs)
        if N == 0:
            return [[] for _ in range(H)]

        K_target = tuple(counts)
        total_calls = sum(K_target)
        # Lower bound on M: at least average concurrency
        lb = 0 if H == 0 else (total_calls + H - 1) // H
        lb = max(lb, 0)
        ub = min(N, max(1, total_calls))  # cannot exceed N; at least 1 if there are calls

        best_schedule: Optional[List[List[int]]] = None
        best_M = None

        while lb <= ub:
            mid = (lb + ub) // 2
            sched = self._feasible(H, mid, K_target)
            if sched is not None:
                best_schedule = sched
                best_M = mid
                ub = mid - 1
            else:
                lb = mid + 1

        if best_schedule is None:
            raise RuntimeError("No feasible schedule for given horizon and counts")
        self._last_peak = best_M  # for inspection
        return best_schedule

    # ---------- Public API ----------

    def rebuild_schedule(self) -> None:
        if not self.subs:
            self._period_schedule = [[]]
            self._period = 1
            return
        period = 1
        for s in self.subs:
            period = lcm(period, s.wait)
        counts = [round(period / s.wait) for s in self.subs]
        sched = self._minimize_peak(period, counts)
        self._period_schedule = sched
        self._period = period
        self._turn = 0


    def run_turn(self, data: T) -> None:
        if self._period_schedule is None or self._period is None:
            self.rebuild_schedule()
        idxs = self._period_schedule[self._turn % self._period]
        for j in idxs:
            self.subs[j].callback(data)
        self._turn += 1


__all__ = ["LoadBalancer"]