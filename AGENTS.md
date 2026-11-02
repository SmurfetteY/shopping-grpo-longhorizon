# Repository contract

Post-training and evaluation for a long-horizon shopping agent: LoRA SFT then veRL GRPO, on the embedded ShopSimulator environment (Chinese e-commerce, ~1.34M Taobao products).

## Workflow

```text
Baseline → SFT → GRPO → Evaluation
```

One workflow only. The runtime contract is ShopSimulator Environment v2.1, Reward v3, observation v2 and tool schema v2. Do not add compatibility launchers, historical datasets, old benchmarks, machine-specific paths or experiment journals.

Training data must never overlap `data/evaluation/tasks.jsonl`. Strict success requires a complete `gold_purchase` terminal result with `reward_valid=true`.

Do not start training, merge models or run the 200-task evaluation unless the user explicitly requests execution.

Commands, env vars, hardware requirements: `README.md`'s 快速开始/常用配置/环境要求 sections. Narrative docs, in reading order: `docs/README.md`.

## Map

| Path | What it is | Own README |
|---|---|---|
| `src/shopping_grpo/` | the installed package — everything below is here | `src/shopping_grpo/README.md` |
| `scripts/` | thin CLI entry points; call into `src/`, no separate logic | — |
| `configs/` | GRPO Hydra yaml, AgentLoop yaml, tool schema json | — |
| `data/` | checked-in task-disjoint SFT/GRPO/eval sets; generated trajectories go to gitignored `outputs/`, never here | `data/README.md` |
| `environments/ShopSimulator/` | frozen embedded environment snapshot; start via `scripts/start_environment.sh`, never by hand | `environments/ShopSimulator/README.md` |
| `docs/` | reward spec, per-stage guides, `research/` diagnosis notes, `plans/` design docs | `docs/README.md` |
| `experiments/` | compact run artifacts (config + summary.json) behind the README results table, no checkpoints | `experiments/README.md` |
| `patches/` | one pinned, hash-verified `verl==0.8.0` patch | — |
| `tests/` | 37 flat pytest files, no conftest | — |

Inside `src/shopping_grpo/`: `environment/` talks to ShopSimulator and owns the action-legality contract; `collection/` turns teacher rollouts into SFT rows; `training/sft/` builds assistant-only-masked examples; `training/grpo/` wires the AgentLoop + reward into veRL (`adapter/`, `dynamic_sampling.py`, `trace.py`); `evaluation/` is the offline Rubric + Judge pipeline; `resources/` ships packaged JSON.

## Tooling gotchas

- **`uv` is required**; `scripts/setup.sh` hard-fails without it.
- **Two separate Python environments**: main project (`.venv/`, Python 3.12, `uv sync --extra sft --extra grpo`) and an *isolated* ShopSimulator env (`.venv-shopsim/`, Python 3.10, its own `requirements.txt`). Scripts hardcode `.venv/bin/python` — don't assume one interpreter.
- **The veRL patch script is defensive on purpose**: it refuses to touch a `verl` install outside this project's own `.venv/`, requires exact version `0.8.0`, SHA-256-verifies both the original and patched `ray_trainer.py`, and auto-backs-up before patching. Never hand-edit the installed `verl` package or route around this script.
- **GPU/Linux is required for real training and rollout** (`training/grpo/adapter/agent_loop.py` is the only module in `src/` that imports torch/verl/vllm/ray at module load time). Everything else in `src/` is plain Python — meaning **most of `tests/` and the package's own logic can be exercised on a CPU-only/non-Linux box** without installing the `sft`/`grpo` extras: `python -m unittest discover -s tests` runs cleanly against a base install (verified on macOS with no GPU). Failures you'll see that are *not* real bugs there: `ModuleNotFoundError` for `verl`/`numpy`/`torch` (extras not installed) and `zip(..., strict=True)` `TypeError`s if run under Python <3.10 (the repo targets 3.10+). The one built-in CPU-safe smoke path is `shopping-grpo smoke` (`src/shopping_grpo/smoke.py::run_cpu_smoke`), which exercises tool schema, trajectory normalization, Reward v3 metrics, SFT label masking and dynamic-sampling grouping without a model or a running ShopSimulator.
- **No CI**: no `.github/`, no CONTRIBUTING.md. This contract is advisory only — nothing enforces it automatically, including the correctness risks below.

## Known correctness risks — read before touching the matching file

These were found by static review + running the CPU-safe test suite (2026-09-15); none are fixed yet. Full write-up with reproduction steps lives outside this repo, in the user's own working notes.

- **`environments/ShopSimulator/shop_env/web_agent_site/engine/reward.py`, `_brand_gate()`**: brand matching is raw substring containment, no word boundary (unlike `_model_boundary_match` in the same file). A candidate whose `shop_name` merely *contains* the required brand as a substring scores as brand-matched — can inflate `valid_alternative_purchase`/`gold_purchase`. `tests/test_reward.py`'s substring-attack test covers `comparators.py::compare_brand`, a function the scoring path never calls — it doesn't actually cover this.
- **`src/shopping_grpo/evaluation/metrics.py`**: the `repetition.environment_repeat_loop` field handed to the Trajectory Judge is derived from Reward v3's own `reward_type`/`termination_reason` — a leak past the "Judge never sees Reward's conclusions" isolation the rest of `blind_guard.py`/`prompts.py` enforces field-by-field (this one slipped through a section-level, not field-level, whitelist).
- **`src/shopping_grpo/evaluation/summary.py`**: `mean_final_reward`/`mean_terminal_utility`/`mean_weighted_score`/`average_steps` divide by *completed* task count, while `strict_success_rate` and friends in the same function correctly divide by the fixed 200-task set. An incomplete run makes the reward/step averages look better than they are relative to the success rate next to them. This is the path `scripts/evaluate.sh` actually uses.
- **`src/shopping_grpo/environment/actions.py`, `action_reject_reason()`**: `search_products` is the one tool with no argument-presence guard (`open_product`/`finish_without_purchase` both degrade safely via `.get()`); a missing `query` raises a bare `KeyError` downstream instead of a guard rejection — GRPO mislabels this as an infrastructure failure, and SFT/eval collection aborts the whole trajectory.
- **`src/shopping_grpo/environment/projection.py`, `_page_type()`**: page-type detection for the pre-V2 `[SEP]`-joined observation format depends on `product_ids()`, whose regex only matches the newer `rank|ASIN|price|title` line format — so a raw observation in the older format is misclassified as `generic` and falls through to non-ASIN-aware truncation, silently dropping product titles/prices while still listing the ASINs as clickable. Reproduced directly; whether the live ShopSimulator service ever emits non-V2 raw text (vs. always going through `render_structured_observation`) is unconfirmed. This is also why `tests/test_observation_projection.py` and 5 cases in `tests/test_rollout.py` currently fail — their `FakeEnv`/fixture observations use the older format.
- **`tests/test_shopping_reward.py`** (whole file) and two methods in `tests/test_verl_adapter.py` import a `reward_components`/`validate_reward_components` API that no longer exists in `training/grpo/adapter/runtime.py` (renamed to `validate_reward`/`reward_detail`) — they fail to collect/error out, so the leak-prevention property they were meant to guard (hidden reward fields never reaching the model-visible tool observation) currently has no live test coverage.
- **`training/grpo/trace.py`**: per-rollout `trace_target` validation checks array length, not each row's value — a `None` entry survives as the literal string `"None"` and gets scored as that turn's gold target, silently corrupting its credit signal. Unconfirmed whether the live environment can actually emit a null target while TRACE is enabled.
- **`scripts/check_grpo_runtime.py`, `validate_dynamic_sampling()`**: only checks the patch marker string is present in `ray_trainer.py`, not the SHA-256 the docs describe — weaker than advertised; a hand-edited or partially-reverted file with the marker intact would pass.

## Known, intentional design tradeoffs — don't "fix" these without reading why first

- `data/sft_curriculum/README.md` documents that the 3-stage (A→B→C) curriculum currently overexposes easy ("foundation") rows 3x vs hard ("strategy") rows 1x, calls this a plausible cause of weak curriculum gains, and explicitly warns against fixing it by duplicating hard rows (reintroduces the same reweighting problem differently).
- KL reward/loss are off, `efficiency`/`penalty_overlong` are 0 in the GRPO config — deliberate, because in this task longer trajectories often mean legitimate extra verification, not just wasted steps; see `docs/grpo.md` and `docs/research/` before reintroducing length penalties.
- Only one deterministic rollout per task in the reported Final-200 results (`docs/research/sft_grpo_diagnosis.md` runs the significance numbers) — the 60.5%→62.0% SFT→GRPO delta is not yet shown statistically significant.

## Root-level note

`面向长程购物 Agent 的 RL 后训练.html` is an archived copy of the project author's own write-up (https://yyhdbl.github.io/), saved here on request; it's a reference document, not part of the training/eval pipeline and not referenced by any script.
