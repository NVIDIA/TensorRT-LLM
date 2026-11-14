# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Online output-length prediction for the generation admission order in disaggregated serving.

Enabled by ``output_length_prediction`` in the disaggregated server config. After a request's context
phase, the orchestrator predicts its output length and writes
``priority = 1 / (1 + tokens / PRIORITY_SCALE_TOKENS)`` (strictly decreasing in the prediction) into the
generation request's ``priority``. Generation servers that run ``waiting_queue_policy: priority`` then
admit the shortest predicted reply first, and a pooled generation router places in the same order.

The predictor learns from the replies of the running server only; nothing is loaded or saved. It lives
in the coordinator (one per orchestrator), so it sees the replies of every frontend. Its inputs are the
conversation's earlier replies (keyed by ``conversation_params.conversation_id``) and the context
phase's prompt length; the request's ``max_tokens`` is never read.

Per conversation it keeps the recorded replies' count, last / EMA(0.5) / mean / max output tokens and
the last prompt length. A reply without a conversation id trains the global models (median, ridge,
trees) with an empty history. A request's 7 features: log1p of the history count, of its prompt tokens, and
of the last / EMA / mean / max output tokens (NaN without history), and the signed log1p of the tokens
appended since the last reply (NaN without history). Rules: a history of exactly one reply with a
prompt under 1,500 tokens is dropped when the next request appends over 10,000 tokens (a new prefix),
and replies of at most 2 tokens (cache primers) are not learned.

Predictors, each taking over at its first fit:
  * a conversation with history gets its EMA(0.5); otherwise the median of the last
    ``MEDIAN_WINDOW`` learned replies, or ``PRIOR_TOKENS`` before the first;
  * ``ridge``: a ridge regression of log1p(output tokens) on the features (NaN as -1 plus a missing
    indicator per feature), refit every ``RIDGE_REFIT_S`` on the replies learned so far;
  * ``ridge_gbm``: also a gradient-boosted tree ensemble on the same target (scikit-learn
    ``HistGradientBoostingRegressor``), refit every ``GBM_REFIT_S`` on the last ``GBM_MAX_ROWS``
    learned replies in a separate Python process off the request path; the coordinator evaluates
    its trees itself.

In a fleet (``num_workers > 1``) a generation router placed by the coordinator gets the prediction
from ``/select`` (the routing key carries ``OUTPUT_LENGTH_KEY``), so it costs no extra round trip;
otherwise the frontend asks the coordinator's ``/output_length``. Finished replies go back in
batches. Only requests with a ``priority`` field (completions and chat completions) are predicted;
``/v1/responses`` requests are not.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import subprocess  # nosec B404 - runs sys.executable -I with a fixed argv, no shell
import sys
import threading
import time
from array import array
from bisect import bisect_left, insort
from collections import OrderedDict, deque
from typing import TYPE_CHECKING, Any, AsyncIterator, List, Optional, Protocol, Tuple, get_args

from tensorrt_llm.llmapi.disagg_utils import OutputLengthPredictorName as PredictorName
from tensorrt_llm.logger import logger

if TYPE_CHECKING:
    from tensorrt_llm.serve.disagg_coordinator import DisaggCoordinator
    from tensorrt_llm.serve.openai_protocol import (
        ChatCompletionRequest,
        CompletionRequest,
        ResponsesRequest,
    )
    from tensorrt_llm.serve.responses_utils import UCompletionResponseOrGenerator

    # A request the orchestrator sends to a generation server.
    GenerationRequest = CompletionRequest | ChatCompletionRequest | ResponsesRequest

PREDICTORS = get_args(PredictorName)

# priority = 1 / (1 + tokens / scale); the admission order does not depend on the scale.
PRIORITY_SCALE_TOKENS = 1024.0
# Neutral starting value: the prediction before any reply is learned (the learned median replaces it
# after the first reply) and for a request the coordinator could not predict.
PRIOR_TOKENS = 400.0
PREDICT_TIMEOUT_S = 1.0
RECORD_TIMEOUT_S = 5.0
STATS_LOG_PERIOD_S = 30.0
# Routing-key and route-info entry of a /select that predicts the output length.
OUTPUT_LENGTH_KEY = "output_length"
# Finished replies go to the coordinator in batches: at most this many, at most this late.
RECORD_BATCH_MAX = 512
RECORD_FLUSH_S = 0.05
# Warn if none of this many reported replies was learned.
NOTHING_LEARNED_WARN_AFTER = 1000

FEATURES = ("turn", "isl", "prev_out", "ema_out", "mean_out", "max_out", "appended")
RESET_MAX_PREV_ISL = 1500.0
RESET_MIN_APPENDED = 10000.0
SKIP_MAX_OUT = 2.0
MAX_CONVERSATIONS = 50000
# A longer conversation id is keyed by its digest, which bounds the table's memory.
MAX_CONVERSATION_ID_CHARS = 128
# Features of predicted requests whose reply has not been learned yet. A row leaves when its reply
# is learned or reported failed, when it is older than PENDING_MAX_AGE_S (the coordinator's pooled
# seat expiry: no generation runs longer), or, past this many rows, oldest first.
MAX_PENDING_ROWS = 200000
PENDING_MAX_AGE_S = 6 * 3600.0
# The median prior covers the last this many learned replies.
MEDIAN_WINDOW = 10000

RIDGE_REFIT_S = 120.0
RIDGE_ALPHA = 1.0
RIDGE_MIN_ROWS = 200
GBM_REFIT_S = 300.0
GBM_MIN_ROWS = 1000
GBM_THREADS = 4
GBM_FIT_TIMEOUT_S = 600.0
# The tree ensemble is fitted on the last this many learned replies (8 doubles each).
GBM_MAX_ROWS = 250000
# A fitted ensemble goes live only if the coordinator's tree walk matches scikit-learn's predict.
GBM_EXPORT_TOL = 1e-6
GBM_PARAMS = {
    "max_iter": 300,
    "learning_rate": 0.05,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 200,
    "l2_regularization": 1.0,
    "early_stopping": True,
    "validation_fraction": 0.1,
    "random_state": 0,
}

NAN = float("nan")
# Prompt and reply lengths are valid up to this many tokens, far beyond any model's context window. It bounds the
# log1p features and targets the models learn (<= 16.7), and a model's log prediction is clamped to it.
MAX_TOKENS = float(2**24)
_LOG1P_MAX_TOKENS = math.log1p(MAX_TOKENS)

# A conversation's history as passed to features(): (count, last out, EMA out, sum out, max out, last prompt).
Snapshot = Tuple[float, float, float, float, float, float]


def is_token_count(value: Any) -> bool:
    """Whether ``value`` is a number in [0, ``MAX_TOKENS``] (not a bool or NaN)."""
    return (
        isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= MAX_TOKENS
    )


def _tokens_of(log1p_tokens: float) -> float:
    """A model's log1p prediction as tokens, clamped to [0, ``MAX_TOKENS``] (NaN stays NaN)."""
    return math.expm1(min(max(log1p_tokens, 0.0), _LOG1P_MAX_TOKENS))


def priority_of(tokens: float) -> float:
    """Map predicted output tokens into [0, 1], decreasing: shorter replies go first.

    A NaN prediction gets the prior's priority.
    """
    tokens = float(tokens)
    if tokens != tokens:
        tokens = PRIOR_TOKENS
    return 1.0 / (1.0 + max(tokens, 0.0) / PRIORITY_SCALE_TOKENS)


class _History:
    __slots__ = ("n", "last_out", "ema", "sum_out", "max_out", "last_isl")

    def __init__(self) -> None:
        self.n = 0
        self.last_out = NAN
        self.ema = NAN
        self.sum_out = 0.0
        self.max_out = NAN
        self.last_isl = NAN

    def record(self, isl: float, out: float) -> None:
        self.ema = out if self.n == 0 else 0.5 * out + 0.5 * self.ema
        self.sum_out += out
        self.max_out = out if self.n == 0 else max(self.max_out, out)
        self.last_out = out
        self.last_isl = isl
        self.n += 1

    def snapshot(self) -> Snapshot:
        return (float(self.n), self.last_out, self.ema, self.sum_out, self.max_out, self.last_isl)


def _table_key(conv_id: str) -> str | bytes:
    """``conv_id``, or a 16-byte digest of it when longer than ``MAX_CONVERSATION_ID_CHARS``."""
    if len(conv_id) <= MAX_CONVERSATION_ID_CHARS:
        return conv_id
    return hashlib.blake2b(conv_id.encode(errors="surrogatepass"), digest_size=16).digest()


class ConversationLengthTable:
    """Per-conversation reply history (least recently used conversations evicted)."""

    def __init__(self, max_conversations: int = MAX_CONVERSATIONS) -> None:
        self.max_conversations = max_conversations
        self._histories: OrderedDict[str | bytes, _History] = OrderedDict()
        self.resets = 0
        self.recorded = 0
        self.skipped = 0

    def __len__(self) -> int:
        return len(self._histories)

    def snapshot_for(self, conv_id: Optional[str], isl: float) -> Optional[Snapshot]:
        """The conversation's history for a new request, after the reset rule; None without one."""
        if conv_id is None:
            return None
        key = _table_key(conv_id)
        history = self._histories.get(key)
        if history is None:
            return None
        self._histories.move_to_end(key)
        if (
            history.n == 1
            and history.last_isl < RESET_MAX_PREV_ISL
            and isl - history.last_isl - history.last_out > RESET_MIN_APPENDED
        ):
            del self._histories[key]
            self.resets += 1
            return None
        return history.snapshot()

    def record(self, conv_id: str, isl: float, out: float) -> None:
        """Add a finished reply to its conversation's history.

        :meth:`OnlineLengthPredictor.record` decides which replies are learned (and counts the
        skipped cache primers in ``skipped``).
        """
        key = _table_key(conv_id)
        history = self._histories.get(key)
        if history is None:
            history = self._histories[key] = _History()
            if len(self._histories) > self.max_conversations:
                self._histories.popitem(last=False)
        else:
            self._histories.move_to_end(key)
        history.record(float(isl), float(out))
        self.recorded += 1


def _log1p_nonneg(value: float) -> float:
    return NAN if value != value else math.log1p(max(value, 0.0))


def _signed_log1p(value: float) -> float:
    if value != value:
        return NAN
    return math.copysign(math.log1p(abs(value)), value) if value != 0.0 else 0.0


def features(snapshot: Optional[Snapshot], isl: float) -> list:
    """The predictors' input row for a request with ``isl`` prompt tokens and a history (None: none)."""
    isl = float(isl)
    if snapshot is None or snapshot[0] <= 0:
        return [0.0, _log1p_nonneg(isl), NAN, NAN, NAN, NAN, NAN]
    n, last_out, ema, sum_out, max_out, last_isl = (float(v) for v in snapshot)
    return [
        math.log1p(n),
        _log1p_nonneg(isl),
        _log1p_nonneg(last_out),
        _log1p_nonneg(ema),
        _log1p_nonneg(sum_out / n),
        _log1p_nonneg(max_out),
        _signed_log1p(isl - last_isl - last_out),
    ]


class TreeEnsemble:
    """Raw prediction of a fitted ``HistGradientBoostingRegressor``: baseline + one leaf per tree.

    A node is ``(is_leaf, value, feature, threshold, missing_go_to_left, left, right)``.
    """

    def __init__(self, baseline: float, trees: list) -> None:
        self.baseline = float(baseline)
        self.trees = [[tuple(node) for node in tree] for tree in trees]

    def raw(self, x: list) -> float:
        acc = self.baseline
        for nodes in self.trees:
            i = 0
            while True:
                is_leaf, value, feature, threshold, missing_left, left, right = nodes[i]
                if is_leaf:
                    acc += value
                    break
                v = x[feature]
                if v != v:
                    i = left if missing_left else right
                else:
                    i = left if v <= threshold else right
        return acc


class RunningMedian:
    """Median of the last ``window`` values added (the mean of the two middle values for an even count).

    ``n`` counts every value added.
    """

    __slots__ = ("_recent", "_sorted", "window", "n")

    def __init__(self, window: int = MEDIAN_WINDOW) -> None:
        self.window = window
        self._recent: deque = deque()
        self._sorted: list = []
        self.n = 0

    def add(self, value: float) -> None:
        value = float(value)
        self._recent.append(value)
        insort(self._sorted, value)
        if len(self._recent) > self.window:
            del self._sorted[bisect_left(self._sorted, self._recent.popleft())]
        self.n += 1

    def value(self) -> float:
        k = len(self._sorted)
        if not k:
            return NAN
        if k % 2:
            return self._sorted[k // 2]
        return (self._sorted[k // 2 - 1] + self._sorted[k // 2]) / 2.0


def ridge_row(x: list) -> list:
    """The ridge's design row: the features with NaN as -1, then a 0/1 missing indicator per feature."""
    return [-1.0 if v != v else float(v) for v in x] + [1.0 if v != v else 0.0 for v in x]


def _solve(a: list, b: list) -> list:
    """Solve ``a @ w = b`` by Gaussian elimination with partial pivoting (``a`` and ``b`` are consumed)."""
    n = len(b)
    for k in range(n):
        p = max(range(k, n), key=lambda r: abs(a[r][k]))
        if a[p][k] == 0.0:
            raise ZeroDivisionError("singular ridge system")
        if p != k:
            a[k], a[p] = a[p], a[k]
            b[k], b[p] = b[p], b[k]
        row_k, pivot = a[k], a[k][k]
        for r in range(k + 1, n):
            f = a[r][k] / pivot
            if f != 0.0:
                row_r = a[r]
                for c in range(k, n):
                    row_r[c] -= f * row_k[c]
                b[r] -= f * b[k]
    w = [0.0] * n
    for k in range(n - 1, -1, -1):
        w[k] = (b[k] - sum(a[k][c] * w[c] for c in range(k + 1, n))) / a[k][k]
    return w


class OnlineRidge:
    """Ridge regression kept as running means and centered co-moments.

    ``fit`` solves ``(Cxx + alpha I) w = Cxy`` with ``intercept = mean(y) - mean(x) @ w``: scikit-learn's
    ``Ridge(alpha)`` on the same rows (the intercept is not penalized).
    """

    def __init__(self, dim: int, alpha: float = RIDGE_ALPHA) -> None:
        self.dim = dim
        self.alpha = float(alpha)
        self.n = 0
        self.mean_x = [0.0] * dim
        self.mean_y = 0.0
        self.cxx = [[0.0] * dim for _ in range(dim)]
        self.cxy = [0.0] * dim
        self.coef: Optional[list] = None
        self.intercept = NAN
        self.fit_rows = 0

    def add(self, z: list, y: float) -> None:
        """Welford update: ``C += (z - old mean) (z - new mean)^T``."""
        self.n += 1
        n = self.n
        dx = [zi - mi for zi, mi in zip(z, self.mean_x)]
        self.mean_x = [mi + di / n for mi, di in zip(self.mean_x, dx)]
        self.mean_y += (y - self.mean_y) / n
        ex = [zi - mi for zi, mi in zip(z, self.mean_x)]
        ey = y - self.mean_y
        for i in range(self.dim):
            di = dx[i]
            if di != 0.0:
                row = self.cxx[i]
                for j in range(self.dim):
                    row[j] += di * ex[j]
                self.cxy[i] += di * ey

    def fit(self) -> None:
        d = self.dim
        a = [
            [
                0.5 * (self.cxx[i][j] + self.cxx[j][i]) + (self.alpha if i == j else 0.0)
                for j in range(d)
            ]
            for i in range(d)
        ]
        w = _solve(a, list(self.cxy))
        self.coef = w
        self.intercept = self.mean_y - sum(wi * mi for wi, mi in zip(w, self.mean_x))
        self.fit_rows = self.n

    def raw(self, z: list) -> float:
        return self.intercept + sum(wi * zi for wi, zi in zip(self.coef, z))


# The fit process: numpy + scikit-learn only. The header (with the server's sys.path, which -I drops)
# is argv[1], the rows come in on stdin, the trees go out as JSON on stdout, with the largest difference
# between its tree walk and scikit-learn's predict on up to 2,000 rows.
_GBM_FIT_SCRIPT = r"""
import json, os, sys, time
header = json.loads(sys.argv[1])
sys.path[:0] = header["sys_path"]
try:
    os.nice(10)
except OSError:
    pass
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
n, d = header["n"], header["d"]
data = np.frombuffer(sys.stdin.buffer.read(), dtype=np.float64).reshape(n, d + 1)
X, y = data[:, :d], data[:, d]
t0 = time.perf_counter()
model = HistGradientBoostingRegressor(**header["params"]).fit(X, y)
fit_s = time.perf_counter() - t0
trees = []
for predictors in model._predictors:
    nodes = predictors[0].nodes
    if len(predictors) != 1 or bool(np.any(nodes["is_categorical"])):
        raise SystemExit("unsupported tree")
    trees.append([[bool(r["is_leaf"]), float(r["value"]), int(r["feature_idx"]), float(r["num_threshold"]),
                   bool(r["missing_go_to_left"]), int(r["left"]), int(r["right"])] for r in nodes])
baseline = float(np.asarray(model._baseline_prediction).ravel()[0])
def walk(x):
    acc = baseline
    for nodes in trees:
        i = 0
        while True:
            leaf, v, f, thr, ml, le, ri = nodes[i]
            if leaf:
                acc += v
                break
            xv = x[f]
            i = (le if ml else ri) if xv != xv else (le if xv <= thr else ri)
    return acc
k = min(n, 2000)
ref = model.predict(X[:k]) if k else []
diff = max((abs(walk([float(v) for v in X[j]]) - float(ref[j])) for j in range(k)), default=0.0)
json.dump({"baseline": baseline, "trees": trees, "n_iter": int(model.n_iter_), "fit_s": fit_s,
           "export_max_abs_diff": diff}, sys.stdout)
"""


def child_sys_path() -> List[str]:
    """The server's import path for a ``-I`` child: ``sys.path`` without the working directory.

    ``-I`` ignores ``PYTHONPATH`` and the user site (where scikit-learn may be installed) and keeps
    the working directory off the path; the child prepends this list instead.
    """
    cwd = os.getcwd()
    return [
        entry
        for entry in sys.path
        if entry and os.path.isabs(entry) and os.path.realpath(entry) != os.path.realpath(cwd)
    ]


# Rows of the startup check, enough for trees with splits under GBM_PARAMS' min_samples_leaf.
GBM_CHECK_ROWS = 1000
GBM_CHECK_TIMEOUT_S = 120.0


def _gbm_check_rows(n: int) -> array:
    """``n`` deterministic synthetic training rows: the features (some missing), then the target."""
    rows = array("d")
    for i in range(n):
        rows.extend(
            NAN if (i + j) % 7 == 0 else float((i * (j + 3)) % 97) for j in range(len(FEATURES))
        )
        rows.append(math.log1p(float((i * 37) % 2000)))
    return rows


def _gbm_fit_problem(result: dict) -> Optional[str]:
    """Why a tree fit's result cannot go live (a failed fit, or a tree walk unlike predict), or None."""
    if "error" in result:
        return str(result["error"])
    diff = result.get("export_max_abs_diff")
    if diff is None or not diff <= GBM_EXPORT_TOL:
        return f"export check {diff}"
    return None


def check_gbm_dependencies(python: Optional[str] = None) -> None:
    """Raise ImportError unless the fit process can fit and export trees with scikit-learn.

    Runs the refit itself (:class:`SubprocessGbmFitter`) once on synthetic rows and applies the
    checks that put a refit live, so what passes here also passes at every refit.
    """
    fitter = SubprocessGbmFitter(timeout_s=GBM_CHECK_TIMEOUT_S, python=python, sync=True)
    fitter.start(_gbm_check_rows(GBM_CHECK_ROWS), GBM_CHECK_ROWS, {})
    problem = _gbm_fit_problem(fitter.poll() or {"error": "no result"})
    if problem is not None:
        raise ImportError(
            "output_length_prediction.predictor='ridge_gbm' fits its trees with numpy and "
            "scikit-learn and reads them from HistGradientBoostingRegressor's fitted predictors, "
            f"which {fitter.python} cannot do ({problem.strip()[-300:]}); "
            "install a scikit-learn that provides them or use predictor='ridge'"
        )


class SubprocessGbmFitter:
    """Fits the tree ensemble off the request path, in a separate Python process.

    ``start`` hands the rows to a background thread, which serializes them and runs the process;
    ``poll`` returns its result once. ``sync=True`` fits inside ``start`` (tests).
    """

    def __init__(
        self,
        threads: int = GBM_THREADS,
        timeout_s: float = GBM_FIT_TIMEOUT_S,
        python: Optional[str] = None,
        sync: bool = False,
    ) -> None:
        self.threads = threads
        self.timeout_s = timeout_s
        self.python = python or sys.executable
        self.sync = sync
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._result: Optional[dict] = None

    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, rows: array, n: int, meta: dict) -> None:
        """Fit on ``rows`` (``n`` rows of the features and the target; not modified afterwards)."""
        header = {"n": n, "d": len(FEATURES), "params": GBM_PARAMS, "sys_path": child_sys_path()}
        if self.sync:
            self._finish(self._run(header, rows), meta)
            return
        self._thread = threading.Thread(
            target=lambda: self._finish(self._run(header, rows), meta),
            name="output-length-gbm-fit",
            daemon=True,
        )
        self._thread.start()

    def _run(self, header: dict, rows: array) -> dict:
        env = dict(os.environ)
        env.update(OMP_NUM_THREADS=str(self.threads), OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        try:
            result = subprocess.run(
                [self.python, "-I", "-c", _GBM_FIT_SCRIPT, json.dumps(header)],
                input=rows.tobytes(),
                capture_output=True,
                timeout=self.timeout_s,
                env=env,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            return {"error": f"{type(e).__name__}: {e}"}
        if result.returncode != 0:
            return {
                "error": f"exit {result.returncode}: {result.stderr.decode(errors='replace')[-1500:]}"
            }
        try:
            return json.loads(result.stdout)
        except ValueError as e:
            return {"error": f"unreadable fit output: {e}"}

    def _finish(self, result: dict, meta: dict) -> None:
        result.update(meta)
        with self._lock:
            self._result = result

    def poll(self) -> Optional[dict]:
        with self._lock:
            result, self._result = self._result, None
        return result


class GbmFitter(Protocol):
    """What :class:`OnlineLengthPredictor` needs of a tree fitter (:class:`SubprocessGbmFitter`)."""

    def busy(self) -> bool: ...

    def start(self, rows: array, n: int, meta: dict) -> None: ...

    def poll(self) -> Optional[dict]: ...


class OnlineLengthPredictor:
    """The coordinator's predictor: the conversation table and the online models.

    Events carry their time ``t`` (``time.monotonic()`` in the coordinator), so the same events give the
    same predictions, except that a tree ensemble goes live at the first event after its fit returns.
    Its memory is bounded: ``MAX_CONVERSATIONS`` histories, ``MAX_PENDING_ROWS`` predicted rows,
    ``MEDIAN_WINDOW`` replies for the median and ``GBM_MAX_ROWS`` training rows for the trees.
    """

    def __init__(self, predictor: PredictorName, gbm_fitter: Optional[GbmFitter] = None) -> None:
        if predictor not in PREDICTORS:
            raise ValueError(f"predictor must be one of {PREDICTORS}, got {predictor!r}")
        self.predictor = predictor
        self.table = ConversationLengthTable()
        self.median = RunningMedian()
        self.ridge = OnlineRidge(2 * len(FEATURES))
        self.gbm_on = predictor == "ridge_gbm"
        self.gbm_fitter = gbm_fitter
        if self.gbm_on and self.gbm_fitter is None:
            check_gbm_dependencies()
            self.gbm_fitter = SubprocessGbmFitter()
        # The last GBM_MAX_ROWS training rows (features + target), a ring once full.
        self.gbm_rows = array("d")
        self.gbm_max_rows = GBM_MAX_ROWS
        self.gbm_n = 0
        self.gbm: Optional[TreeEnsemble] = None
        self.gbm_fits = 0
        self.gbm_errors = 0
        self.gbm_busy_skips = 0
        self.ridge_fits = 0
        self.t0: Optional[float] = None
        self.next_ridge_fit = math.inf
        self.next_gbm_fit = math.inf
        # Prediction key -> (time of the prediction, features), oldest first.
        self._pending: OrderedDict[int, Tuple[float, list]] = OrderedDict()
        self._next_key = 0
        self.unmatched = 0
        self.dropped = 0
        self.learned = 0
        self.without_conversation = 0
        self.invalid = 0
        self.reported = 0
        self.sources = {"prior": 0, "median": 0, "ema": 0, "ridge": 0, "gbm": 0}
        self._last_stats = time.monotonic()

    def _tick(self, t: float) -> None:
        """Start the clock at the first event, then run every refit boundary at or before ``t``."""
        if self.t0 is None:
            self.t0 = t
            self.next_ridge_fit = t + RIDGE_REFIT_S
            if self.gbm_on:
                self.next_gbm_fit = t + GBM_REFIT_S
            return
        if t >= self.next_ridge_fit:
            # One fit for every boundary passed: no rows arrive between them, so it is the same.
            if self.ridge.n >= RIDGE_MIN_ROWS:
                self.ridge.fit()
                self.ridge_fits += 1
            self.next_ridge_fit += RIDGE_REFIT_S * (
                1 + math.floor((t - self.next_ridge_fit) / RIDGE_REFIT_S)
            )
        while t >= self.next_gbm_fit:
            rows = min(self.gbm_n, self.gbm_max_rows)
            if rows >= GBM_MIN_ROWS:
                if self.gbm_fitter.busy():
                    self.gbm_busy_skips += 1
                else:
                    meta = {"rows": rows, "boundary_s": self.next_gbm_fit - self.t0}
                    # A bounded copy: later replies overwrite the ring while the fit runs.
                    self.gbm_fitter.start(self.gbm_rows[: rows * (len(FEATURES) + 1)], rows, meta)
            self.next_gbm_fit += GBM_REFIT_S
        if self.gbm_on:
            result = self.gbm_fitter.poll()
            if result is not None:
                self._install_gbm(result)

    def _add_gbm_row(self, x: list, y: float) -> None:
        width = len(FEATURES) + 1
        row = array("d", x)
        row.append(y)
        if self.gbm_n < self.gbm_max_rows:
            self.gbm_rows.extend(row)
        else:
            i = (self.gbm_n % self.gbm_max_rows) * width
            self.gbm_rows[i : i + width] = row
        self.gbm_n += 1

    def _install_gbm(self, result: dict) -> None:
        problem = _gbm_fit_problem(result)
        if problem is not None:
            self.gbm_errors += 1
            logger.warning(
                f"[output_length] tree fit on {result.get('rows')} replies failed, the current model "
                f"stays: {problem}"
            )
            return
        diff = result["export_max_abs_diff"]
        self.gbm = TreeEnsemble(result["baseline"], result["trees"])
        self.gbm_fits += 1
        logger.info(
            f"[output_length] tree fit #{self.gbm_fits} live: {result['rows']} replies, {result['n_iter']} "
            f"iterations, fit {result['fit_s']:.2f} s, export check {diff:.1e}"
        )

    def predict(
        self, conv_id: Optional[str], isl: float, t: float, key: Optional[int] = None
    ) -> Tuple[float, int]:
        """(predicted output tokens, key): the key goes back with the reply, so its features can be learned.

        A ``key`` still pending (the same request predicted again, e.g. a repeated ``/select``) is
        reused, so the request keeps one row. Raises ValueError unless ``isl`` is a finite number
        in [0, ``MAX_TOKENS``].
        """
        if not is_token_count(isl):
            raise ValueError(
                f"prompt tokens must be a finite number in [0, {MAX_TOKENS:.0f}], got {isl!r}"
            )
        self._tick(t)
        snapshot = self.table.snapshot_for(conv_id, isl)
        x = features(snapshot, isl)
        tokens, source = NAN, "prior"
        if self.gbm is not None:
            tokens, source = _tokens_of(self.gbm.raw(x)), "gbm"
        elif self.ridge.coef is not None:
            tokens, source = _tokens_of(self.ridge.raw(ridge_row(x))), "ridge"
        elif snapshot is not None:
            tokens, source = snapshot[2], "ema"
        elif self.median.n:
            tokens, source = self.median.value(), "median"
        if not math.isfinite(tokens):
            tokens, source = PRIOR_TOKENS, "prior"
        self.sources[source] += 1
        if key is None or key not in self._pending:
            key = self._next_key
            self._next_key += 1
        self._pending[key] = (t, x)
        self._pending.move_to_end(key)
        pending = self._pending
        while pending and (
            len(pending) > MAX_PENDING_ROWS
            or t - next(iter(pending.values()))[0] > PENDING_MAX_AGE_S
        ):
            pending.popitem(last=False)
        self._maybe_log_stats()
        return tokens, key

    def record(
        self,
        conv_id: Optional[str],
        isl: float,
        out: Optional[float],
        key: Optional[int],
        t: float,
    ) -> bool:
        """Learn a finished reply of ``out`` tokens; returns whether it was learned.

        ``out`` None reports a request that ended without a reply length (failed, interrupted):
        its row is dropped, nothing is learned. Not learned either: an ``isl`` / ``out`` outside
        [0, ``MAX_TOKENS``] (or NaN), and a reply of at most ``SKIP_MAX_OUT`` tokens (a cache
        primer). A reply without a conversation id trains the global models only.
        """
        self._tick(t)
        row = self._pending.pop(key, None) if key is not None else None
        if out is None:
            self.dropped += row is not None
            return False
        self.reported += 1
        x = row[1] if row is not None else None
        if not (is_token_count(isl) and is_token_count(out)):
            self.invalid += 1
            return self._learned(False)
        if out <= SKIP_MAX_OUT:
            self.table.skipped += 1
            return self._learned(False)
        if conv_id is None:
            self.without_conversation += 1
            if x is None:
                # Without a conversation the features never depend on history.
                x = features(None, isl)
        else:
            self.table.record(conv_id, isl, out)
        self.median.add(out)
        if x is None:
            self.unmatched += 1
        else:
            y = math.log1p(out)
            self.ridge.add(ridge_row(x), y)
            if self.gbm_on:
                self._add_gbm_row(x, y)
        return self._learned(True)

    def _learned(self, learned: bool) -> bool:
        self.learned += learned
        if self.reported == NOTHING_LEARNED_WARN_AFTER and not self.learned:
            logger.warning(
                f"[output_length] none of the first {self.reported} replies was learned "
                f"({self.invalid} invalid, {self.table.skipped} of at most {SKIP_MAX_OUT:g} tokens): "
                f"every generation request keeps the prior's priority"
            )
        self._maybe_log_stats()
        return learned

    def stats_line(self) -> str:
        sources = ", ".join(f"{k} {v}" for k, v in self.sources.items() if v) or "none"
        line = (
            f"[output_length] {self.predictor}: {len(self.table)} conversations, replies learned "
            f"{self.learned} ({self.without_conversation} without a conversation id; skipped "
            f"{self.table.skipped}, invalid {self.invalid}, resets {self.table.resets}), median "
            f"{self.median.value():.1f} tokens; predictions by source: {sources}; ridge fits {self.ridge_fits}"
        )
        if self.gbm_on:
            line += f"; tree fits {self.gbm_fits}, errors {self.gbm_errors}, busy skips {self.gbm_busy_skips}"
        return (
            line
            + f"; pending {len(self._pending)}, replies without their features {self.unmatched}, "
            f"failed requests dropped {self.dropped}"
        )

    def _maybe_log_stats(self) -> None:
        now = time.monotonic()
        if now - self._last_stats >= STATS_LOG_PERIOD_S:
            self._last_stats = now
            logger.info(self.stats_line())


def request_inputs(request: Any) -> Tuple[Optional[str], Optional[float]]:
    """(conversation id, prompt tokens) of a generation request after its context phase.

    Reads ``conversation_params.conversation_id`` and ``disaggregated_params.ctx_usage.prompt_tokens``
    only; in particular never ``max_tokens``. An id the server generated for one request (a
    sub-agent call with only its parent's header) names no conversation: None.
    """
    conversation = getattr(request, "conversation_params", None)
    conv_id = None
    if conversation is not None and not getattr(conversation, "generated_for_one_request", False):
        conv_id = getattr(conversation, "conversation_id", None)
    params = getattr(request, "disaggregated_params", None)
    usage = getattr(params, "ctx_usage", None) if params is not None else None
    isl = getattr(usage, "prompt_tokens", None) if usage is not None else None
    return (str(conv_id) if conv_id is not None else None, float(isl) if isl is not None else None)


_USAGE_MARK = b'"completion_tokens"'
# A line with a ``choices`` key that is an empty list or null; a line with a non-empty one is
# not a usage-only chunk, so it is not decoded (with continuous usage every chunk has usage).
_NO_CHOICES = re.compile(rb'"choices"\s*:\s*(?:\[\s*\]|null)')


def _sse_json(event: bytes) -> Optional[dict]:
    """The JSON object of an SSE event's ``data:`` line, else None."""
    for line in event.split(b"\n"):
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == b"[DONE]":
            return None
        try:
            obj = json.loads(payload)
        except ValueError:
            return None
        return obj if isinstance(obj, dict) else None
    return None


def completion_tokens_in(data: bytes) -> Tuple[Optional[int], bool]:
    """Scan SSE ``data:`` lines for a usage-only chunk's ``usage.completion_tokens``: (value, True).

    (None, False) without one. Only lines that may be such a chunk are decoded.
    """
    if _USAGE_MARK not in data:
        return None, False
    found, final = None, False
    for line in data.split(b"\n"):
        if (b'"choices"' in line and not _NO_CHOICES.search(line)) or _USAGE_MARK not in line:
            continue
        obj = _sse_json(line)
        usage = obj.get("usage") if obj is not None else None
        if isinstance(usage, dict) and usage.get("completion_tokens") is not None:
            found = int(usage["completion_tokens"])
            final = not obj.get("choices")
    return found, final


class _Ticket:
    __slots__ = ("conv_id", "isl", "key", "sequences", "recorded", "via_select", "strip_usage")

    def __init__(
        self, conv_id: Optional[str], isl: float, key: Optional[int], sequences: int = 1
    ) -> None:
        self.conv_id = conv_id
        self.isl = isl
        self.key = key
        # Returned sequences (``n``): the reply's usage sums their lengths.
        self.sequences = sequences
        self.recorded = False
        # The prediction comes with the generation placement (/select).
        self.via_select = False
        # The client did not ask for streamed usage: drop the final usage-only chunk.
        self.strip_usage = False


def throttled(count: int) -> bool:
    """Whether the ``count``-th occurrence of a repeated warning is logged: the first 10, then every 1000th."""
    return count <= 10 or count % 1000 == 0


class OutputLengthPriority:
    """Frontend side: set the generation request's priority, then report its reply's length.

    ``coordinator`` provides ``predict_output_length`` and ``record_output_lengths``
    (:class:`~tensorrt_llm.serve.disagg_coordinator.DisaggCoordinator`). With ``via_select``, a
    request placed after :meth:`assign` gets its prediction from the generation placement
    (:meth:`placed`) instead of a separate call. Replies are reported in batches of up to
    ``RECORD_BATCH_MAX``, at most ``record_flush_s`` after the first of a batch.
    """

    def __init__(
        self,
        coordinator: DisaggCoordinator,
        predictor: PredictorName,
        via_select: bool = False,
        record_flush_s: float = 0.0,
    ) -> None:
        self._coordinator = coordinator
        self.predictor = predictor
        self.via_select = via_select
        self.record_flush_s = record_flush_s
        self.prior_priority = priority_of(PRIOR_TOKENS)
        self._tasks: set = set()
        self._batch: List[Tuple[Optional[str], float, Optional[float], Optional[int]]] = []
        self._flush_task: Optional[asyncio.Task] = None
        self.assigned = 0
        self.missing = 0
        self.recorded = 0
        self.no_usage = 0
        self.record_errors = 0
        self._call_ms: list = []
        self._last_stats = time.monotonic()

    def startup_line(self) -> str:
        return (
            f"[output_length] generation priority from the predicted output length ({self.predictor}, "
            f"learned online in the coordinator{', predicted in /select' if self.via_select else ''}): "
            f"priority = 1 / (1 + tokens / {PRIORITY_SCALE_TOKENS:g}); "
            f"without a prediction, {PRIOR_TOKENS:g} tokens (priority {self.prior_priority:.4f}); max_tokens "
            f"is never read; /v1/responses requests are not predicted (they wait for a pooled "
            f"generation seat at the prior's priority, and get the default priority on the "
            f"generation server)"
        )

    async def assign(
        self, gen_req: GenerationRequest, placed_later: bool = False
    ) -> Optional[_Ticket]:
        """Set ``gen_req.priority`` from the prediction; the ticket (None: nothing to learn) records the reply.

        With ``placed_later`` and ``via_select`` the prediction comes with the placement
        (:meth:`placed`). A streamed request asks the generation server for its usage, which
        :meth:`track` reads (and drops from the stream if the client did not ask for it).
        Never raises: without a prediction (no prompt tokens, a coordinator error or timeout) the
        request gets the prior's priority.
        """
        if "priority" not in getattr(type(gen_req), "model_fields", {}):
            return None
        conv_id, isl = request_inputs(gen_req)
        if isl is None:
            self._miss(gen_req, "no prompt tokens")
            return None
        n = getattr(gen_req, "n", None)
        sequences = n if isinstance(n, int) and not isinstance(n, bool) and n > 1 else 1
        ticket = _Ticket(conv_id, isl, None, sequences)
        self._request_usage(gen_req, ticket)
        if self.via_select and placed_later:
            ticket.via_select = True
            return ticket
        t0 = time.perf_counter()
        try:
            tokens, key = await asyncio.wait_for(
                self._coordinator.predict_output_length(conv_id, isl), PREDICT_TIMEOUT_S
            )
        except Exception as e:  # noqa: BLE001  (an admission hint is never worth failing a request)
            self._miss(gen_req, f"{type(e).__name__}: {e}")
            return ticket
        self._call_ms.append((time.perf_counter() - t0) * 1e3)
        gen_req.priority = priority_of(tokens)
        ticket.key = key
        self.assigned += 1
        self._maybe_log_stats()
        return ticket

    def assign_prior(self, gen_req: GenerationRequest) -> None:
        """Give a request served without a context phase the prior's priority (nothing is learned)."""
        if "priority" in getattr(type(gen_req), "model_fields", {}):
            gen_req.priority = self.prior_priority

    @staticmethod
    def _request_usage(gen_req: GenerationRequest, ticket: _Ticket) -> None:
        if not getattr(gen_req, "stream", False):
            return
        options = getattr(gen_req, "stream_options", None)
        if options is not None and options.include_usage:
            return
        from tensorrt_llm.serve.openai_protocol import StreamOptions

        # Without include_usage the client also gets no per-chunk usage.
        gen_req.stream_options = StreamOptions(include_usage=True, continuous_usage_stats=False)
        ticket.strip_usage = True

    def placed(
        self, ticket: Optional[_Ticket], gen_req: GenerationRequest, route_info: Optional[dict]
    ) -> None:
        """Take the prediction from the generation placement's route info (``via_select``)."""
        if ticket is None or not ticket.via_select:
            return
        prediction = route_info.get(OUTPUT_LENGTH_KEY) if isinstance(route_info, dict) else None
        key = prediction.get("key") if isinstance(prediction, dict) else None
        if not isinstance(key, int) or isinstance(key, bool):
            self._miss(gen_req, "the generation placement returned no prediction")
            return
        ticket.key = key
        self.assigned += 1
        self._maybe_log_stats()

    def _miss(self, gen_req: GenerationRequest, reason: str) -> None:
        self.missing += 1
        gen_req.priority = self.prior_priority
        if throttled(self.missing):
            logger.warning(f"[output_length] no prediction #{self.missing}, prior used: {reason}")

    def abandon(self, ticket: Optional[_Ticket]) -> None:
        """The request failed before its reply: the coordinator drops its row unlearned."""
        if ticket is not None:
            self._record(ticket, None)

    def _record(self, ticket: _Ticket, out: Optional[int]) -> None:
        if ticket.recorded:
            return
        ticket.recorded = True
        if out is None:
            self.no_usage += 1
            if ticket.key is None:
                return
            # No length to learn: report it, so the coordinator drops its row.
            self._batch.append((ticket.conv_id, ticket.isl, None, ticket.key))
        else:
            self.recorded += 1
            # Learn the length of one sequence.
            self._batch.append(
                (ticket.conv_id, ticket.isl, float(out) / ticket.sequences, ticket.key)
            )
        if len(self._batch) >= RECORD_BATCH_MAX:
            self._send(self._take_batch())
        elif self._flush_task is None:
            self._flush_task = asyncio.get_running_loop().create_task(self._flush_later())

    def _take_batch(self) -> list:
        batch, self._batch = self._batch, []
        return batch

    async def _flush_later(self) -> None:
        try:
            await asyncio.sleep(self.record_flush_s)
        finally:
            self._flush_task = None
        if self._batch:
            self._send(self._take_batch())

    def _send(self, batch: list) -> None:
        async def _post() -> None:
            try:
                await asyncio.wait_for(
                    self._coordinator.record_output_lengths(batch), RECORD_TIMEOUT_S
                )
            except Exception as e:  # noqa: BLE001  (learning must not fail the request)
                self.record_errors += 1
                if throttled(self.record_errors):
                    logger.warning(
                        f"[output_length] recording {len(batch)} replies failed "
                        f"(#{self.record_errors}): {type(e).__name__}: {e}"
                    )

        task = asyncio.get_running_loop().create_task(_post())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def track(
        self, response: UCompletionResponseOrGenerator, ticket: Optional[_Ticket]
    ) -> UCompletionResponseOrGenerator:
        """Report the reply's ``usage.completion_tokens``: wraps a stream, reads a complete response."""
        if ticket is None:
            return response
        if hasattr(response, "__aiter__"):
            return self._track_stream(response, ticket)
        usage = getattr(response, "usage", None)
        out = getattr(usage, "completion_tokens", None) if usage is not None else None
        self._record(ticket, int(out) if out is not None else None)
        return response

    async def _track_stream(self, stream: AsyncIterator, ticket: _Ticket) -> AsyncIterator[bytes]:
        """Relay ``stream``; learn from its final usage-only chunk only (an interrupted stream is not learned).

        With ``ticket.strip_usage`` the stream is relayed in whole SSE events, without that chunk.
        """
        carry = b""
        try:
            async for chunk in stream:
                if ticket.recorded and not ticket.strip_usage:
                    yield chunk
                    continue
                data = chunk if isinstance(chunk, (bytes, bytearray)) else str(chunk).encode()
                scan = carry + bytes(data)
                if ticket.strip_usage:
                    end = scan.rfind(b"\n\n")
                    if end < 0:
                        carry = scan
                        continue
                    carry = scan[end + 2 :]
                    events = scan[: end + 2]
                    kept = self._drop_final_usage(events, ticket)
                    if kept:
                        yield kept
                    continue
                newline = scan.rfind(b"\n")
                carry = scan[newline + 1 :][-4096:] if newline >= 0 else scan[-4096:]
                if newline >= 0:
                    out, final = completion_tokens_in(scan[: newline + 1])
                    if out is not None and final:
                        self._record(ticket, out)
                yield chunk
            if ticket.strip_usage and carry:
                kept = self._drop_final_usage(carry, ticket)
                carry = b""
                if kept:
                    yield kept
        finally:
            if not ticket.recorded:
                self._record(ticket, None)

    def _drop_final_usage(self, events: bytes, ticket: _Ticket) -> bytes:
        """``events`` (whole SSE events) without the final usage-only chunk, which is recorded."""
        if _USAGE_MARK not in events:
            return events
        kept = []
        for event in events.split(b"\n\n"):
            obj = _sse_json(event) if _USAGE_MARK in event else None
            usage = obj.get("usage") if obj is not None else None
            if isinstance(usage, dict) and not obj.get("choices"):
                if usage.get("completion_tokens") is not None:
                    self._record(ticket, int(usage["completion_tokens"]))
                continue
            kept.append(event)
        return b"\n\n".join(kept)

    def _maybe_log_stats(self) -> None:
        now = time.monotonic()
        if now - self._last_stats < STATS_LOG_PERIOD_S:
            return
        self._last_stats = now
        ms = sorted(self._call_ms)
        self._call_ms = []
        p50 = ms[len(ms) // 2] if ms else NAN
        logger.info(
            f"[output_length] assigned {self.assigned}, missing {self.missing}, replies recorded {self.recorded} / "
            f"without final usage {self.no_usage} / record errors {self.record_errors}; coordinator call p50 "
            f"{p50:.2f} ms, max {ms[-1] if ms else NAN:.2f} ms"
        )
