# keel_trader — top-level build/test entrypoints.
# The sim is pure C (the single fill engine); tests pin its behaviour.

CC      ?= clang
CFLAGS  ?= -O2 -Wall -Wextra -Isim
SAN     := -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -Isim

PYTHON ?= python3

.PHONY: test test-fill test-safety test-asan clean

test: test-fill test-safety ## run all golden fixtures (fill model + safety spine)

test-fill: ## pin the single fill engine against its golden values
	$(CC) $(CFLAGS) tests/test_fill_model.c -lm -o /tmp/keel_test_fill
	/tmp/keel_test_fill

test-safety: ## pin the safety spine (singleton lock + death-spiral guard)
	PYTHONPATH=. $(PYTHON) tests/test_safety_spine.py

test-asan: ## same fixture under ASan/UBSan
	$(CC) $(SAN) tests/test_fill_model.c -lm -o /tmp/keel_test_fill_asan
	/tmp/keel_test_fill_asan

clean:
	rm -f /tmp/keel_test_fill /tmp/keel_test_fill_asan /tmp/te.o
