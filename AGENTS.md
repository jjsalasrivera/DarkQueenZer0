# AGENTS.md

## Project
- DarkQueenZer0: an 8x8 Spanish checkers game with AlphaZero-style AI agents.
- Rust + Bevy 0.12.1. Single binary crate (no workspace); all code in `src/`, game assets in `assets/`.
- Bevy 0.12 is intentionally old (`NodeBundle`, `TextBundle`, `AppExit`, `Res`/`ResMut`). Do not "modernize" to newer Bevy APIs; changes must compile against 0.12.1.

## Commands
- `cargo run` — launch the game. Run from the repo root so Bevy finds `assets/`. A clean build compiles all of Bevy (~1 min here).
- `cargo run --release -- train [iters] [games] [sims]` — head-less AlphaZero self-play training; writes `assets/models/alpha_zero.safetensors`. Use `--release` (debug is ~20× slower).
- `cargo run --release -- arena [games] [sims] [greedy|random|self]` — head-less evaluation of the trained net vs a fast baseline (alternates colors).
- `cargo test` — rules/engine + AI unit tests (in `src/rules.rs`, `src/game_manager.rs`, `src/ia/`).
- `cargo check` / `cargo clippy` — fast feedback while iterating. Clippy is expected to be clean (`--all-targets`); the only remaining output is a warning from the external `block` crate.
- CI: `.github/workflows/ci.yml` runs `cargo check`, `cargo clippy -D warnings` and `cargo test` on push/PR. There is still no rustfmt config.

## Architecture / wiring
- `src/main.rs` — Bevy app setup, resource init, system registration. New modules must be added to the `mod` list here.
- `src/comun.rs` — shared types and board constants (`Square`, `Move`, `Turn`, `GameStatus`, `GamePlayer`, `BandPlayer`, `INITIAL_BOARD`).
- `src/rules.rs` — pure, Bevy-independent rule engine: `legal_moves_for_piece`, `queen_moves`, `apply_move`/`undo_apply`, `max_capture_depth`, `all_legal_moves`, `continuation_moves`. Operates on an owned `Board` so it can be reused headlessly by training.
- `src/game_manager.rs` — `GameManager`: board state, turn/history, `do_move`/`undo_move`, status evaluation, and the AI handle. Rules logic delegates to `src/rules.rs`.
  - **`do_move`/`undo_move` must be exact inverses.** `do_move` saves (and `undo_move` restores) `moves_with_no_capture`, `forced_piece` **and `turn`** — all three are stacked in `*_history` vectors. Forgetting to restore `turn` silently breaks every MCTS simulation (the tree's side-to-move drifts), so keep `turn_history` in sync. `undo_move` recalculates status after restoring.
- `src/board.rs` — Bevy rendering, pointer input, and UI (top bar, restart, mode buttons).
  - `bevy_mod_picking`'s default `debug` feature is **disabled** in `Cargo.toml` on purpose: it spawns a debug text label that follows the mouse cursor. Don't re-enable it (keep the explicit feature list).
- `src/train.rs` — AlphaZero self-play training loop (subcommand `train`): builds the net from a `VarMap`, runs MCTS self-play, trains with `AdamW`, saves checkpoints.
- `src/arena.rs` — head-less evaluation (subcommand `arena`): the trained net vs a fast baseline (`GreedyMover` = 1-ply material, `RandomMover`, or itself). The classic UCB brain is deliberately not used as reference (700 recursive playouts/move, prints in its inner loop).
- `src/ia/` — AI ("IA"). `brain.rs` defines the `Brain` trait; `monte_carlo_impl.rs` and `upper_confidence_bounds.rs` implement it.
- `src/ia/encoding.rs` — board → 4x8x8 input planes and move ↔ action index (`from*64 + to`, 4096 actions). The board is **canonicalized by color**: for Black it is mirrored vertically (and own/opponent swapped) so "forward" is always up; use `canonical_action_index(move, turn)` for any policy indexing, never the raw `action_index`.
- `src/ia/neural.rs` — `PolicyValueNet` (MLP, hidden 256) built with `candle`; `random()` for untrained weights, `from_varmap` to share weights for training.
- `src/ia/mcts.rs` — `Mcts`, PUCT guided by the network; implements `Brain`. `search()` returns `(move, visit-count policy)` (the training target); `load_or_random()` loads a checkpoint or random weights.
  - **Capture chains and value sign:** `Child::flips_turn` (via `move_flips_turn`) records whether a move changes player. `simulate`/`q_from_parent` negate the value only when it flips. A capture that continues leaves the turn unchanged, so the sign must NOT flip. Getting this wrong silently ruins MCTS.
  - Search is **deterministic**; variety comes from the ε-greedy random moves and the temperature sampling of the opening plies (`temperature_plies`) in `train.rs`, plus the ε-greedy moves in `arena.rs`, not from the tree.

## Rules engine facts (easy to get wrong)
- **Mandatory maximum capture:** `all_legal_moves` drops quiet moves when a capture exists, and keeps only captures whose full chain is the longest. Captures are simulated on a copy via `max_capture_depth`/`filter_max_captures`.
- **Forced continuation:** after a capture that can continue, `do_move` does NOT switch turns and sets `forced_piece`; `get_legal_moves` then returns only `continuation_moves` for that piece. `undo_move` restores it from `forced_history`.
- **Player names are inconsistent by design:** `WHITE_PAWN`/`WHITE_QUEEN` (positive) == the `Turn::Red` player == assets named `Roja`. `BLACK_*` (negative) == `Turn::Black` == `Negra`. "White" and "Red" are the same side.
- **Board orientation:** row 0 is at the bottom of the screen and rows increase upward. Red starts on rows 0-2 (bottom) and moves toward higher rows; Black on rows 5-7. By default Red is Human, Black is Computer.
- Code comments and UI/CLI strings are in English. Some asset names remain Spanish by design (`Roja`/`Negra` = the Red/Black piece textures); keep that mapping.

## AI selection
- `GameManager::with_options(band_player, brain_type)` chooses the AI via `BrainType` (`UpperConfidenceBounds` default, `MonteCarlo` and `Neural` reserved/available), wired through `create_brain`. `GameManager::set_brain_type`/`get_brain_type` swap it at runtime; the `AI: Classic`/`AI: Neural` button in `board.rs` (`MenuAction::ToggleBrain`) does this and restarts the game.
- `GameManager::new()` defaults to Human (Red) vs Computer (Black) with UCB.
- `src/ia/mcts.rs` (`BrainType::Neural`) uses `candle`. `create_brain` loads `assets/models/alpha_zero.safetensors` if present, otherwise random weights; do not expect it to play well until trained.
- Training (`src/train.rs`) builds the learner net and each self-play `Mcts` from the same `VarMap`, so self-play always uses the latest weights. `AdamW` updates the vars in place (`Var::set`), and checkpoints are safetensors saved via `VarMap::save`.
- Self-play records MCTS visit distributions as the policy target (with ε-greedy random moves for exploration) and the final result as the value target, from each position's side-to-move perspective. The network's value is also side-to-move relative, matching `encoding::encode`.
- The AI runs on `AsyncComputeTaskPool` in `board.rs` (`start_computer_move` / `finish_computer_move`, resource `AiTask`) so the window never blocks. The compute task uses a cloned `GameManager` whose `Arc<Mutex<dyn Brain>>` is shared with the live one. Cancel by setting `ai_task.task = None` (done on restart).
- `GameManager::get_hash()` hashes `board` + `turn` + `forced_piece`; it is the transposition key for the UCB maps, so changing what is hashed changes AI behavior.

## Known weak spots (candidates for Phase 6+)
- The network is only as good as its training run; the checkpoint (`assets/models/alpha_zero.safetensors`) is gitignored, so a fresh checkout starts with random weights until `train` is run. Measure with `arena`, don't assume.
- The classic UCB brain is a hand-rolled rollout search, not a real MCTS: its transposition stats persist across moves and it is very slow (700 recursive playouts/move), so it is unusable as an arena baseline.
- The network only ever self-plays against itself: no self-evaluation against previous checkpoints, no Dirichlet root noise, no tree reuse.
- No repetition/threefold draw detection; draws only come from the 16-move no-capture rule.
