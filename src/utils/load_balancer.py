from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from typing import Callable, Generic, List, Tuple, Dict, Optional, TypeVar
from math import lcm

T = TypeVar("T")


@dataclass(frozen=True)
class Subscriber(Generic[T]):
    id: int
    callback: Callable[[T], None]
    wait: int  # w_j >= 1


class LoadBalancer(Generic[T]):
    def __init__(self) -> None:
        self.subs: List[Subscriber[T]] = []
        self._sub_map: Dict[int, Subscriber[T]] = {}
        self._next_id: int = 0

        self._period_schedule: Optional[List[List[int]]] = None  # stores subscriber ids per turn
        self._period: Optional[int] = None
        self._turn: int = 0
        self._last_peak: Optional[int] = None

    def subscribe(self, callback: Callable[[T], None], wait_turns_for_call: int) -> int:
        if wait_turns_for_call < 1:
            raise ValueError("wait_turns_for_call must be >= 1")
        sid = self._next_id
        self._next_id += 1
        sub = Subscriber(id=sid, callback=callback, wait=wait_turns_for_call)
        self.subs.append(sub)
        self._sub_map[sid] = sub
        return sid

    def unsubscribe(self, subscriber_id: int) -> None:
        sub = self._sub_map.get(subscriber_id)
        if sub is None:
            raise ValueError(f"no subscriber with id {subscriber_id}")
        # remove from list and map
        for i, s in enumerate(self.subs):
            if s.id == subscriber_id:
                self.subs.pop(i)
                break
        del self._sub_map[subscriber_id]
        # invalidate any existing period schedule so future run_turn will rebuild
        self._period_schedule = None
        self._period = None

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

        @lru_cache(maxsize=None)
        def max_calls_from_state(cool: Tuple[int, ...], remain: Tuple[int, ...], turns_left: int) -> int:
            total = 0
            for j in range(N):
                r = remain[j]
                if r <= 0:
                    continue
                w = W[j]
                c = max(cool[j], 0)
                first = 0 if c == 0 else c
                if first >= turns_left:
                    continue
                possible = 1 + (turns_left - 1 - first) // w
                total += min(possible, r)
            return total

        @lru_cache(maxsize=None)
        def F(i: int, cool: Tuple[int, ...], remain: Tuple[int, ...]) -> bool:
            turns_left = H - i
            rem_total = sum(remain)
            if rem_total == 0:
                return True
            if turns_left <= 0:
                return False
            if max_calls_from_state(cool, remain, turns_left) < rem_total:
                return False

            eligible = [j for j in range(N) if cool[j] == 0 and remain[j] > 0]
            if not eligible:
                new_cool = tuple(max(c - 1, 0) for c in cool)
                return F(i + 1, new_cool, remain)

            max_take = min(M, len(eligible))
            eligible_sorted = sorted(eligible, key=lambda j: (-remain[j], -W[j]))

            for take in range(max_take, -1, -1):
                for U in combinations(eligible_sorted, take):
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

                    if max_calls_from_state(cool_next, remain_next, turns_left - 1) < sum(remain_next):
                        continue

                    if F(i + 1, cool_next, remain_next):
                        parent[(i, cool, remain)] = (U, cool_next, remain_next)
                        return True
            return False

        parent: Dict[Tuple[int, Tuple[int, ...], Tuple[int, ...]], Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]] = {}
        cool0 = tuple(0 for _ in W)
        remain0 = tuple(K_target)

        ok = F(0, cool0, remain0)
        if not ok:
            return None

        schedule: List[List[int]] = [[] for _ in range(H)]
        i, cool, remain = 0, cool0, remain0
        while i < H and sum(remain) > 0:
            key = (i, cool, remain)
            if key not in parent:
                # No calls this turn; advance cooldowns
                cool = tuple(max(c - 1, 0) for c in cool)
                i += 1
                continue
            U, cool_next, remain_next = parent[key]
            schedule[i] = list(U)
            i += 1
            cool, remain = cool_next, remain_next

        for t in range(i, H):
            schedule[t] = []
        return schedule

    def _minimize_peak(self, H: int, counts: List[int]) -> List[List[int]]:
        N = len(self.subs)
        if N == 0:
            return [[] for _ in range(H)]

        K_target = tuple(counts)
        total_calls = sum(K_target)
        lb = 0 if H == 0 else (total_calls + H - 1) // H
        lb = max(lb, 0)
        ub = min(N, max(1, total_calls))

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
        self._last_peak = best_M
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
        # convert schedule of indices to schedule of subscriber ids
        sched_ids: List[List[int]] = [[self.subs[idx].id for idx in turn] for turn in sched]
        self._period_schedule = sched_ids
        self._period = period
        self._turn = 0

    def run_turn(self, data: T) -> None:
        if not self.subs:
            return
        if self._period_schedule is None or self._period is None:
            self.rebuild_schedule()
        idxs = self._period_schedule[self._turn % self._period]
        for sid in idxs:
            sub = self._sub_map.get(sid)
            if sub is not None:
                sub.callback(data)
        self._turn += 1
        
        if self._turn == self._period:
            self._turn = 0


__all__ = ["LoadBalancer"]
