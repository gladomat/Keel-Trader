/*
 * keel_sim.c — stable C ABI over the ONE fill engine, for the ctypes driver.
 *
 * keel's #1 architectural rule (docs/REBUILD_HANDOFF.md §1): there is exactly ONE
 * fill engine. This file is NOT a second sim — it #includes trading_env.c so every
 * fill/reward/accounting line is the *same* code that tests/test_fill_model.c pins
 * and that the gate/RL/backtest all run through. It only adds a flat, ctypes-callable
 * ABI (the pufferlib binding in binding.c is the alternative, extension-based path;
 * this is the no-pufferlib fallback issue #1 sanctions).
 *
 * Build:  make build-sim   ->  keel_sim.so   (gcc -shared -fPIC)
 * Driver: sim/keel_sim.py  (ctypes wrapper)
 */

#include "trading_env.c"   /* reach the static fill fns; ONE engine, no copy */

#ifdef _WIN32
#  define KEEL_API __declspec(dllexport)
#else
#  define KEEL_API __attribute__((visibility("default")))
#endif

/* ---------- market data ---------- */

KEEL_API MarketData* keel_market_data_load(const char* path) {
    return market_data_load(path);
}
KEEL_API void keel_market_data_free(MarketData* md) {
    market_data_free(md);
}
KEEL_API int keel_md_num_symbols(const MarketData* md)   { return md ? md->num_symbols : 0; }
KEEL_API int keel_md_num_timesteps(const MarketData* md) { return md ? md->num_timesteps : 0; }
KEEL_API int keel_md_features_per_sym(const MarketData* md) { return md ? md->features_per_sym : 0; }

/* ---------- env lifecycle ----------
 *
 * Flow:  env = keel_env_create(md)
 *        keel_env_set_param(env, "fee_rate", 0.001) ...   (any number of times)
 *        keel_env_finalize(env)                            (allocates obs/action buffers)
 *        keel_env_reset(env)
 *        loop:  keel_env_set_action(env, a); keel_env_step(env)
 *               read keel_env_obs_ptr / keel_env_reward / keel_env_terminal
 *        keel_env_free(env)
 */

KEEL_API TradingEnv* keel_env_create(MarketData* md) {
    if (!md) return NULL;
    TradingEnv* env = (TradingEnv*)calloc(1, sizeof(TradingEnv));
    if (!env) return NULL;
    env->data = md;

    /* production-safe defaults — mirror binding.c my_init so a naive ctypes
       caller can never accidentally run lookahead-permissive params. */
    env->max_steps                 = 720;
    env->fee_rate                  = 0.001f;
    env->max_leverage              = 1.0f;
    env->short_borrow_apr          = 0.0f;
    env->periods_per_year          = 8760.0f;
    env->max_hold_hours            = 0;
    env->action_allocation_bins    = 1;
    env->action_level_bins         = 1;
    env->action_max_offset_bps     = 0.0f;
    env->reward_scale              = 10.0f;
    env->reward_clip               = 5.0f;
    env->cash_penalty              = 0.01f;
    env->drawdown_penalty          = 0.0f;
    env->downside_penalty          = 0.0f;
    env->smooth_downside_penalty   = 0.0f;
    env->smooth_downside_temperature = 0.02f;
    env->trade_penalty             = 0.0f;
    env->smoothness_penalty        = 0.0f;
    env->fill_slippage_bps         = 0.0f;
    env->fill_buffer_bps           = 0.0f;
    env->fill_probability          = 1.0f;
    env->decision_lag              = 2;     /* production-safe default */
    env->death_spiral_tolerance_bps = 0.0f; /* off by default */
    env->death_spiral_overnight_tolerance_bps = 500.0f;
    env->death_spiral_stale_after_bars = 8;
    env->forced_offset             = -1;    /* random */
    env->enable_drawdown_profit_early_exit = 0;
    env->drawdown_profit_early_exit_verbose = 0;
    env->drawdown_profit_early_exit_min_steps = 20;
    env->drawdown_profit_early_exit_progress_fraction = 0.5f;
    return env;
}

/* Generic, ABI-stable setter keyed by name. Avoids mirroring the (large,
   order-sensitive) struct layout in Python. Returns 1 if the key matched. */
KEEL_API int keel_env_set_param(TradingEnv* env, const char* name, double v) {
    if (!env || !name) return 0;
    #define I(k) do { if (strcmp(name, #k)==0) { env->k = (int)v; return 1; } } while(0)
    #define F(k) do { if (strcmp(name, #k)==0) { env->k = (float)v; return 1; } } while(0)
    I(max_steps);            F(fee_rate);              F(max_leverage);
    F(short_borrow_apr);     F(periods_per_year);      I(max_hold_hours);
    I(action_allocation_bins); I(action_level_bins);   F(action_max_offset_bps);
    F(reward_scale);         F(reward_clip);           F(cash_penalty);
    F(drawdown_penalty);     F(downside_penalty);      F(smooth_downside_penalty);
    F(smooth_downside_temperature); F(trade_penalty);  F(smoothness_penalty);
    F(fill_slippage_bps);    F(fill_buffer_bps);       F(fill_probability);
    I(decision_lag);         F(death_spiral_tolerance_bps);
    F(death_spiral_overnight_tolerance_bps); I(death_spiral_stale_after_bars);
    I(forced_offset);        I(enable_drawdown_profit_early_exit);
    I(drawdown_profit_early_exit_verbose); I(drawdown_profit_early_exit_min_steps);
    F(drawdown_profit_early_exit_progress_fraction);
    #undef I
    #undef F
    return 0;
}

/* Compute derived sizes and allocate the pufferlib-shaped IO buffers. */
KEEL_API int keel_env_finalize(TradingEnv* env) {
    if (!env || !env->data) return -1;
    if (env->action_allocation_bins < 1) env->action_allocation_bins = 1;
    if (env->action_level_bins < 1) env->action_level_bins = 1;
    if (env->decision_lag < 1) env->decision_lag = 1;

    int S = env->data->num_symbols;
    int F = env->data->features_per_sym;
    int side_block = S * env->action_allocation_bins * env->action_level_bins;
    env->obs_size = S * F + 5 + S;
    env->num_actions = 1 + 2 * side_block;

    free(env->observations); free(env->actions);
    free(env->rewards);      free(env->terminals);
    env->observations = (float*)calloc((size_t)env->obs_size, sizeof(float));
    env->actions      = (int*)calloc(1, sizeof(int));
    env->rewards      = (float*)calloc(1, sizeof(float));
    env->terminals    = (unsigned char*)calloc(1, sizeof(unsigned char));
    if (!env->observations || !env->actions || !env->rewards || !env->terminals) return -1;
    return 0;
}

KEEL_API int   keel_env_obs_size(const TradingEnv* env)    { return env ? env->obs_size : 0; }
KEEL_API int   keel_env_num_actions(const TradingEnv* env) { return env ? env->num_actions : 0; }
KEEL_API const float* keel_env_obs_ptr(const TradingEnv* env) { return env ? env->observations : NULL; }
KEEL_API void  keel_env_set_action(TradingEnv* env, int a) { if (env && env->actions) env->actions[0] = a; }
KEEL_API float keel_env_reward(const TradingEnv* env)      { return (env && env->rewards) ? env->rewards[0] : 0.0f; }
KEEL_API int   keel_env_terminal(const TradingEnv* env)    { return (env && env->terminals) ? (int)env->terminals[0] : 0; }

KEEL_API void keel_env_reset(TradingEnv* env) { if (env) c_reset(env); }
KEEL_API void keel_env_step(TradingEnv* env)  { if (env) c_step(env);  }

KEEL_API void keel_env_free(TradingEnv* env) {
    if (!env) return;
    free(env->observations); free(env->actions);
    free(env->rewards);      free(env->terminals);
    free(env);
}

/* ---------- episode log accessors (accumulated at terminal) ---------- */
KEEL_API float keel_env_log_total_return(const TradingEnv* env) { return env ? env->log.total_return : 0.0f; }
KEEL_API float keel_env_log_sortino(const TradingEnv* env)      { return env ? env->log.sortino : 0.0f; }
KEEL_API float keel_env_log_max_drawdown(const TradingEnv* env) { return env ? env->log.max_drawdown : 0.0f; }
KEEL_API float keel_env_log_num_trades(const TradingEnv* env)   { return env ? env->log.num_trades : 0.0f; }
KEEL_API float keel_env_log_win_rate(const TradingEnv* env)     { return env ? env->log.win_rate : 0.0f; }
KEEL_API float keel_env_log_n(const TradingEnv* env)            { return env ? env->log.n : 0.0f; }

/* live (non-accumulated) per-step introspection for backtest/paper */
KEEL_API float keel_env_max_drawdown_live(const TradingEnv* env) { return env ? env->agent.max_drawdown : 0.0f; }
KEEL_API int   keel_env_step_idx(const TradingEnv* env)          { return env ? env->agent.step : 0; }

/* deterministic RNG seed for reproducible random-offset episodes / fill prob */
KEEL_API void keel_seed(unsigned long long s) {
    g_trading_rng_state = s ? s : 0x9E3779B97F4A7C15ULL;
}

/* ----------------------------------------------------------------------
 * Golden parity hook: drive the REAL static fill fns directly so the
 * Python gate test can assert it ties to tests/test_fill_model.c values.
 * Returns the buy fill price; writes entry_price and round-trip cost
 * (buy then close at the same bar OPEN) through out params.
 * -------------------------------------------------------------------- */
KEEL_API float keel_test_roundtrip_cost(float o, float h, float l, float c,
                                        float buffer_bps, float slip_bps,
                                        float* out_entry, float* out_cost) {
    MarketData md;
    memset(&md, 0, sizeof(md));
    float prices[PRICE_FEATS] = {o, h, l, c, 1000.0f};
    float feats[FEATURES_PER_SYM] = {0};
    md.num_symbols = 1; md.num_timesteps = 1; md.features_per_sym = FEATURES_PER_SYM;
    md.prices = prices; md.features = feats;

    TradingEnv env;
    memset(&env, 0, sizeof(env));
    env.data = &md; env.fee_rate = 0.001f; env.max_leverage = 1.0f;
    env.fill_buffer_bps = buffer_bps; env.fill_slippage_bps = slip_bps;
    env.action_allocation_bins = 1; env.action_level_bins = 1;
    env.agent.cash = INITIAL_CASH; env.agent.position_sym = -1;

    float fill = 0.0f;
    resolve_limit_fill_price(&env, &md, 0, 0, 100.0f, /*is_buy=*/1, &fill);

    /* fresh env for the round-trip accounting */
    memset(&env, 0, sizeof(env));
    env.data = &md; env.fee_rate = 0.001f; env.max_leverage = 1.0f;
    env.fill_buffer_bps = buffer_bps; env.fill_slippage_bps = slip_bps;
    env.action_allocation_bins = 1; env.action_level_bins = 1;
    env.agent.cash = INITIAL_CASH; env.agent.position_sym = -1;
    open_long(&env, 0, 0, 1.0f, 0.0f);
    float entry = env.agent.entry_price;
    close_position(&env, 0);
    float cost = INITIAL_CASH - env.agent.cash;

    if (out_entry) *out_entry = entry;
    if (out_cost)  *out_cost = cost;
    return fill;
}
