/*
 * test_fill_model.c — GOLDEN FILL FIXTURE for keel_trader.
 *
 * keel's #1 architectural rule (see docs/REBUILD_HANDOFF.md §1): there is exactly
 * ONE fill engine, and its behaviour is pinned by this test. Any future soft /
 * differentiable wrapper, Python eval, or refactor MUST reproduce these exact
 * fills and costs, or it has silently diverged from ground truth — the precise
 * bug that made the old repo's eval read ~fill_buffer bps rosier than training.
 *
 * Values below were derived empirically by driving the real C fill functions
 * over a grid (the moray fill-parity run, EVAL_SIM_PARITY_DEEPDIVE.md §5).
 *
 * Build & run:  make -C .. test        (or: clang -O1 -Isim test_fill_model.c -lm -o /tmp/t && /tmp/t)
 */
#include "../sim/src/trading_env.c"   /* include .c to reach static fill fns */
#include <assert.h>

static MarketData* make_md(float o, float h, float l, float c) {
    MarketData* md = (MarketData*)calloc(1, sizeof(MarketData));
    md->num_symbols = 1; md->num_timesteps = 1; md->features_per_sym = FEATURES_PER_SYM;
    md->prices = (float*)calloc(PRICE_FEATS, sizeof(float));
    md->features = (float*)calloc(FEATURES_PER_SYM, sizeof(float));
    md->prices[P_OPEN]=o; md->prices[P_HIGH]=h; md->prices[P_LOW]=l;
    md->prices[P_CLOSE]=c; md->prices[P_VOL]=1000.0f;
    return md;
}
static void init_env(TradingEnv* env, MarketData* md, float buf_bps, float slip_bps) {
    memset(env, 0, sizeof(*env));
    env->data = md; env->fee_rate = 0.001f; env->max_leverage = 1.0f;
    env->fill_buffer_bps = buf_bps; env->fill_slippage_bps = slip_bps;
    env->action_allocation_bins = 1; env->action_level_bins = 1;
    env->agent.cash = INITIAL_CASH; env->agent.position_sym = -1;
}
static int approx(float a, float b, float tol) { return (a-b < tol) && (b-a < tol); }

/* one grid cell: buy fill price + round-trip cost (buy then close at same open) */
static void check(float buf, float slip, float exp_fill, float exp_entry, float exp_cost) {
    MarketData* md = make_md(100.0f, 101.0f, 99.0f, 100.0f);
    TradingEnv env; init_env(&env, md, buf, slip);
    float fill = 0.0f;
    int ok = resolve_limit_fill_price(&env, md, 0, 0, /*target=*/100.0f, /*is_buy=*/1, &fill);
    assert(ok && "buy must fill: bar low (99) is below target (100)");
    assert(approx(fill, exp_fill, 1e-3f) && "fill price drifted from golden value");

    init_env(&env, md, buf, slip);
    open_long(&env, 0, 0, 1.0f, 0.0f);
    assert(env.agent.position_sym >= 0 && "open_long must take a position");
    float entry = env.agent.entry_price;   /* capture before close zeroes it */
    assert(approx(entry, exp_entry, 1e-3f) && "entry_price drifted");
    close_position(&env, 0);
    float cost = INITIAL_CASH - env.agent.cash;
    assert(approx(cost, exp_cost, 0.05f) && "round-trip cost drifted (fee+slip accounting)");

    free(md->prices); free(md->features); free(md);
    printf("  ok  buffer=%-4.0f slip=%-4.0f  fill=%.4f entry=%.4f cost=%.4f\n",
           buf, slip, fill, entry, cost);
}

int main(void) {
    printf("keel fill-model golden fixture (bar O=100 H=101 L=99, fee=10bps):\n");

    /* INVARIANT 1: buffer only GATES the fill; it never moves the fill price.
       C fills at target regardless of buffer. (The old Python intrabar eval
       instead filled at open*(1-buffer) — that divergence is what this pins out.) */
    check(0.0f,  0.0f, 100.0000f, 100.0000f, 19.98f);
    check(5.0f,  0.0f, 100.0000f, 100.0000f, 19.98f);
    check(20.0f, 0.0f, 100.0000f, 100.0000f, 19.98f);

    /* INVARIANT 2: round-trip cost ~= 2*(fee+slip). Slippage enters as an
       adverse price shift on entry AND effective_fee on exit. */
    check(0.0f,  5.0f,  100.0500f, 100.0500f, 29.96f);   /* +5bps slip  -> cost ~30 */
    check(0.0f,  20.0f, 100.2000f, 100.2000f, 59.84f);   /* +20bps slip -> cost ~60 */

    /* INVARIANT 3: entry_price is slip-shifted (SL/TP/death-spiral key off this). */
    check(5.0f,  20.0f, 100.2000f, 100.2000f, 59.84f);

    printf("ALL FILL-MODEL INVARIANTS HOLD.\n");
    return 0;
}
