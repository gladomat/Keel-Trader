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

.PHONY: test test-fill test-safety test-asan test-sim build-sim data clean

test: test-fill test-safety test-sim ## run all golden fixtures (fill model + safety spine + sim binding)

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

data: ## regenerate the committed-by-recipe synthetic sample .bin (git-ignored output)
	$(PYTHON) sim/make_sample_data.py --output sim/data/sample.bin

test-asan: ## same fixture under ASan/UBSan
	$(CC) $(SAN) tests/test_fill_model.c -lm -o /tmp/keel_test_fill_asan
	/tmp/keel_test_fill_asan

clean:
	rm -f /tmp/keel_test_fill /tmp/keel_test_fill_asan /tmp/te.o $(SIM_SO)
