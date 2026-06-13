"""ctypes driver for the ONE C fill engine (keel_sim.so).

This is the issue-#1 fallback binding: instead of vendoring pufferlib's
``env_binding.h`` + numpy dev headers to build a CPython extension, we compile
``sim/src/keel_sim.c`` (which ``#include``s ``trading_env.c``) into a plain shared
library and call it over a flat ctypes ABI. It is the *same* fill arithmetic the
golden fixture ``tests/test_fill_model.c`` pins — there is no second fill model.

Stdlib-only (ctypes), so ``make test`` stays toolchain-light: no numpy required.

Build it with ``make build-sim``. Typical use::

    from sim.keel_sim import MarketData, TradingEnv
    md = MarketData.load("sim/data/sample.bin")
    env = TradingEnv(md, max_steps=64, decision_lag=2, forced_offset=0)
    obs = env.reset()
    while True:
        env.step(action=0)            # action 0 = go flat
        if env.terminal:
            break
    print(env.log)                    # {'total_return': ..., 'sortino': ..., ...}
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Optional

_SO_PATH = Path(__file__).resolve().parent / "libkeelsim.so"

_lib: Optional[ctypes.CDLL] = None


def _load() -> ctypes.CDLL:
    global _lib
    if _lib is not None:
        return _lib
    if not _SO_PATH.exists():
        raise RuntimeError(
            f"{_SO_PATH} not built. Run `make build-sim` first "
            "(compiles the ctypes sim driver from the one C core)."
        )
    lib = ctypes.CDLL(str(_SO_PATH))

    c_f, c_i, c_vp, c_cp = ctypes.c_float, ctypes.c_int, ctypes.c_void_p, ctypes.c_char_p

    lib.keel_market_data_load.restype = c_vp
    lib.keel_market_data_load.argtypes = [c_cp]
    lib.keel_market_data_free.argtypes = [c_vp]
    for fn in ("keel_md_num_symbols", "keel_md_num_timesteps", "keel_md_features_per_sym"):
        getattr(lib, fn).restype = c_i
        getattr(lib, fn).argtypes = [c_vp]
    lib.keel_md_feature.restype = c_f
    lib.keel_md_feature.argtypes = [c_vp, c_i, c_i, c_i]
    lib.keel_md_price.restype = c_f
    lib.keel_md_price.argtypes = [c_vp, c_i, c_i, c_i]

    lib.keel_env_create.restype = c_vp
    lib.keel_env_create.argtypes = [c_vp]
    lib.keel_env_set_param.restype = c_i
    lib.keel_env_set_param.argtypes = [c_vp, c_cp, ctypes.c_double]
    lib.keel_env_finalize.restype = c_i
    lib.keel_env_finalize.argtypes = [c_vp]
    lib.keel_env_obs_size.restype = c_i
    lib.keel_env_obs_size.argtypes = [c_vp]
    lib.keel_env_num_actions.restype = c_i
    lib.keel_env_num_actions.argtypes = [c_vp]
    lib.keel_env_obs_ptr.restype = ctypes.POINTER(c_f)
    lib.keel_env_obs_ptr.argtypes = [c_vp]
    lib.keel_env_set_action.argtypes = [c_vp, c_i]
    lib.keel_env_reward.restype = c_f
    lib.keel_env_reward.argtypes = [c_vp]
    lib.keel_env_terminal.restype = c_i
    lib.keel_env_terminal.argtypes = [c_vp]
    lib.keel_env_reset.argtypes = [c_vp]
    lib.keel_env_step.argtypes = [c_vp]
    lib.keel_env_free.argtypes = [c_vp]
    for fn in ("keel_env_log_total_return", "keel_env_log_sortino",
               "keel_env_log_max_drawdown", "keel_env_log_num_trades",
               "keel_env_log_win_rate", "keel_env_log_n",
               "keel_env_max_drawdown_live"):
        getattr(lib, fn).restype = c_f
        getattr(lib, fn).argtypes = [c_vp]
    lib.keel_env_step_idx.restype = c_i
    lib.keel_env_step_idx.argtypes = [c_vp]
    lib.keel_seed.argtypes = [ctypes.c_ulonglong]

    lib.keel_test_roundtrip_cost.restype = c_f
    lib.keel_test_roundtrip_cost.argtypes = [
        c_f, c_f, c_f, c_f, c_f, c_f, ctypes.POINTER(c_f), ctypes.POINTER(c_f)
    ]

    _lib = lib
    return lib


def seed(value: int) -> None:
    """Seed the C-side PRNG (random episode offsets / fill-probability rolls)."""
    _load().keel_seed(ctypes.c_ulonglong(int(value)))


class MarketData:
    """Read-only market data loaded once; share across many envs."""

    def __init__(self, ptr: int):
        self._ptr = ptr
        lib = _load()
        self.num_symbols = lib.keel_md_num_symbols(ptr)
        self.num_timesteps = lib.keel_md_num_timesteps(ptr)
        self.features_per_sym = lib.keel_md_features_per_sym(ptr)

    @classmethod
    def load(cls, data_path: str | os.PathLike) -> "MarketData":
        lib = _load()
        ptr = lib.keel_market_data_load(str(data_path).encode("utf-8"))
        if not ptr:
            raise RuntimeError(f"Failed to load market data from {data_path}")
        return cls(ptr)

    def feature(self, t: int, s: int, f: int) -> float:
        return float(_load().keel_md_feature(self._ptr, t, s, f))

    def price(self, t: int, s: int, p: int) -> float:
        return float(_load().keel_md_price(self._ptr, t, s, p))

    def free(self) -> None:
        if self._ptr:
            _load().keel_market_data_free(self._ptr)
            self._ptr = 0


# Parameters accepted by TradingEnv (forwarded to keel_env_set_param). Listing them
# keeps typos loud — a name the C setter doesn't recognise raises.
_ENV_PARAMS = {
    "max_steps", "fee_rate", "max_leverage", "short_borrow_apr", "periods_per_year",
    "max_hold_hours", "action_allocation_bins", "action_level_bins",
    "action_max_offset_bps", "reward_scale", "reward_clip", "cash_penalty",
    "drawdown_penalty", "downside_penalty", "smooth_downside_penalty",
    "smooth_downside_temperature", "trade_penalty", "smoothness_penalty",
    "fill_slippage_bps", "fill_buffer_bps", "fill_probability", "decision_lag",
    "death_spiral_tolerance_bps", "death_spiral_overnight_tolerance_bps",
    "death_spiral_stale_after_bars", "forced_offset",
    "enable_drawdown_profit_early_exit", "drawdown_profit_early_exit_verbose",
    "drawdown_profit_early_exit_min_steps",
    "drawdown_profit_early_exit_progress_fraction",
}


class TradingEnv:
    """One agent stepping through the C sim. Construct, then reset() and step()."""

    def __init__(self, data: MarketData, **params):
        lib = _load()
        self._lib = lib
        self._md = data
        self._ptr = lib.keel_env_create(data._ptr)
        if not self._ptr:
            raise RuntimeError("keel_env_create failed")
        for name, value in params.items():
            if name not in _ENV_PARAMS:
                raise TypeError(f"unknown env param: {name!r}")
            if not lib.keel_env_set_param(self._ptr, name.encode("utf-8"), float(value)):
                raise RuntimeError(f"keel_env_set_param rejected {name!r}")
        if lib.keel_env_finalize(self._ptr) != 0:
            raise RuntimeError("keel_env_finalize failed")
        self.obs_size = lib.keel_env_obs_size(self._ptr)
        self.num_actions = lib.keel_env_num_actions(self._ptr)

    def set_param(self, name: str, value: float) -> None:
        if name not in _ENV_PARAMS:
            raise TypeError(f"unknown env param: {name!r}")
        if not self._lib.keel_env_set_param(self._ptr, name.encode("utf-8"), float(value)):
            raise RuntimeError(f"keel_env_set_param rejected {name!r}")

    def reset(self) -> list[float]:
        self._lib.keel_env_reset(self._ptr)
        return self.obs

    def step(self, action: int) -> tuple[list[float], float, bool]:
        self._lib.keel_env_set_action(self._ptr, int(action))
        self._lib.keel_env_step(self._ptr)
        return self.obs, self.reward, self.terminal

    @property
    def obs(self) -> list[float]:
        ptr = self._lib.keel_env_obs_ptr(self._ptr)
        return [ptr[i] for i in range(self.obs_size)]

    @property
    def reward(self) -> float:
        return float(self._lib.keel_env_reward(self._ptr))

    @property
    def terminal(self) -> bool:
        return bool(self._lib.keel_env_terminal(self._ptr))

    @property
    def max_drawdown_live(self) -> float:
        return float(self._lib.keel_env_max_drawdown_live(self._ptr))

    @property
    def step_idx(self) -> int:
        return int(self._lib.keel_env_step_idx(self._ptr))

    @property
    def log(self) -> dict:
        l = self._lib
        return {
            "total_return": float(l.keel_env_log_total_return(self._ptr)),
            "sortino": float(l.keel_env_log_sortino(self._ptr)),
            "max_drawdown": float(l.keel_env_log_max_drawdown(self._ptr)),
            "num_trades": float(l.keel_env_log_num_trades(self._ptr)),
            "win_rate": float(l.keel_env_log_win_rate(self._ptr)),
            "n": float(l.keel_env_log_n(self._ptr)),
        }

    def free(self) -> None:
        if self._ptr:
            self._lib.keel_env_free(self._ptr)
            self._ptr = 0

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass


def roundtrip_cost(o: float, h: float, l: float, c: float,
                   buffer_bps: float, slip_bps: float) -> tuple[float, float, float]:
    """Drive the real C static fill fns: returns (fill_price, entry_price, cost).

    The gate's parity guard asserts these equal tests/test_fill_model.c's golden
    values — proving the ctypes path ties to the same C arithmetic.
    """
    lib = _load()
    entry = ctypes.c_float(0.0)
    cost = ctypes.c_float(0.0)
    fill = lib.keel_test_roundtrip_cost(
        o, h, l, c, buffer_bps, slip_bps, ctypes.byref(entry), ctypes.byref(cost)
    )
    return float(fill), float(entry.value), float(cost.value)
