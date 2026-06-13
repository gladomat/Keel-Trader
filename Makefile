# keel_trader — top-level build/test entrypoints.
# The sim is pure C (the single fill engine); tests pin its behaviour.

CC      ?= clang
CFLAGS  ?= -O2 -Wall -Wextra -Isim
SAN     := -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -Isim

PYTHON ?= python3

# Shared-lib build for the ctypes sim driver (issue #1 fallback path: NO pufferlib /
# numpy headers needed; wraps the SAME C core as the golden fill fixture).
SIM_SO  := sim/libkeelsim.so
SOFLAGS := -O2 -fPIC -shared -Wall -Isim

.PHONY: test test-fill test-safety test-asan test-sim test-features test-gate test-strategy test-forecast test-backtest test-rl test-autoresearch test-kraken-paper build-sim data data-kraken gate-kraken backtest-kraken autoresearch-kraken paper-kraken clean

# Real Kraken .bin the crypto judges run against (git-ignored; build via data-kraken).
KRAKEN_BIN ?= sim/data/kraken_market.bin

test: test-fill test-safety test-sim test-features test-gate test-strategy test-forecast test-backtest test-rl test-autoresearch test-kraken-paper ## run all golden fixtures

test-fill: ## pin the single fill engine against its golden values
	$(CC) $(CFLAGS) tests/test_fill_model.c -lm -o /tmp/keel_test_fill
	/tmp/keel_test_fill

test-safety: ## pin the safety spine (singleton lock + death-spiral guard)
	PYTHONPATH=. $(PYTHON) tests/test_safety_spine.py

build-sim: $(SIM_SO) ## compile the ctypes sim driver (keel_sim.so) — the ONE C core
$(SIM_SO): sim/src/keel_sim.c sim/src/trading_env.c sim/include/trading_env.h
	$(CC) $(SOFLAGS) sim/src/keel_sim.c -lm -o $(SIM_SO)

test-sim: build-sim ## smoke + parity test the sim binding via ctypes
	PYTHONPATH=. $(PYTHON) tests/test_sim_binding.py

test-features: build-sim ## pin the ONE feature spec + validator + .bin round-trip
	PYTHONPATH=. $(PYTHON) tests/test_features.py

test-gate: build-sim ## pin the out-of-sample gate (reject flat, fill parity, fail-fast)
	PYTHONPATH=. $(PYTHON) tests/test_gate.py

test-strategy: ## pin the pure XGB strategy (conviction + inverse-vol sizing)
	PYTHONPATH=. $(PYTHON) tests/test_strategy.py

test-forecast: ## pin the forecast cache guards (leakage invariant + spec + MAE/PnL)
	PYTHONPATH=. $(PYTHON) tests/test_forecast_cache.py

test-backtest: build-sim ## pin the gate backtest verdict + paper-only decision loop
	PYTHONPATH=. $(PYTHON) tests/test_backtest_paper.py

test-rl: build-sim ## pin the RL policy -> gate seam (no second sim) + GAE
	PYTHONPATH=. $(PYTHON) tests/test_rl_policy.py

test-autoresearch: build-sim ## pin the autoresearch leaderboard (append-only, reproducible, gate-honest)
	PYTHONPATH=. $(PYTHON) tests/test_autoresearch.py

test-kraken-paper: build-sim ## pin the live-data paper loop (paper-only, fills via the ONE C engine)
	PYTHONPATH=. $(PYTHON) tests/test_kraken_paper.py

data: ## regenerate the committed-by-recipe synthetic sample .bin (git-ignored output)
	PYTHONPATH=. $(PYTHON) sim/make_sample_data.py --output sim/data/sample.bin

data-kraken: ## fetch REAL Kraken hourly OHLCV -> .bin (K1/K2, offline: needs network + ccxt)
	PYTHONPATH=. $(PYTHON) sim/kraken_data.py --output $(KRAKEN_BIN)

gate-kraken: build-sim ## run the crypto-retuned gate on the real Kraken .bin (K4, offline)
	PYTHONPATH=. $(PYTHON) research/eval.py --data $(KRAKEN_BIN) --policy long

backtest-kraken: build-sim ## backtest the XGB strategy through the gate on real Kraken data (K4)
	PYTHONPATH=. $(PYTHON) models/xgb/backtest.py --data $(KRAKEN_BIN) \
		--verdict-out artifacts/kraken_verdict.json

autoresearch-kraken: build-sim ## run the autoresearch search loop on real Kraken data (K4)
	PYTHONPATH=. $(PYTHON) research/autoresearch.py --data $(KRAKEN_BIN) \
		--leaderboard artifacts/kraken_leaderboard.csv

paper-kraken: build-sim ## live-data PAPER trading vs Kraken — no real orders (K5, offline: needs ccxt)
	PYTHONPATH=. $(PYTHON) -m core.kraken_paper --ticks 24 --sleep 3600

test-asan: ## same fixture under ASan/UBSan
	$(CC) $(SAN) tests/test_fill_model.c -lm -o /tmp/keel_test_fill_asan
	/tmp/keel_test_fill_asan

clean:
	rm -f /tmp/keel_test_fill /tmp/keel_test_fill_asan /tmp/te.o $(SIM_SO)
