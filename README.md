# DarkQueenZer0

A Spanish checkers game (damas españolas) with AI agents inspired by AlphaZero,
written in Rust with the [Bevy](https://bevyengine.org/) engine.

## Requirements

- Rust stable + Cargo
- A desktop environment able to run Bevy (macOS, Linux, Windows)

## Run

Run from the repository root so Bevy finds the `assets/` folder (textures and font):

```sh
cargo run
```

The first build compiles Bevy and takes a while.

## Controls

- Click one of your pieces to highlight its legal moves in green, then click a
  highlighted square to move.
- Capturing is mandatory and you must take the **longest** available capture.
  A capture chain continues with the same piece until no more captures exist.
- The top bar shows the turn or the result, plus five buttons:
  `New game`, `You start`, `AI starts`, `2 Players`, and `AI: Classic`. The
  button for the mode currently in effect is highlighted in green.
- `You start` plays against the computer with you as **Red** (bottom), moving
  first. This is the mode at launch.
- `AI starts` plays against the computer with the computer as **Red**, moving
  immediately. You play Black, and the board is rotated 180° so your pieces
  appear at the bottom.
- `2 Players` starts a new local game with both sides played by humans.
- `New game` restarts the current mode.
- While the computer is on move, the top bar shows `AI is thinking...`.
- `AI: Classic` toggles the computer's brain between the classic UCB search
  (`AI: Classic`) and the trained AlphaZero network (`AI: Neural`). Switching
  starts a new game.
- When a game ends, use `New game` to start over.

## Test

```sh
cargo test
```

The rules engine lives in `src/rules.rs` and is covered by unit tests
(mandatory capture, maximum-capture rule, capture chains, promotion, undo).

## Training (self-play)

The AlphaZero network is trained against itself, head-less:

```sh
cargo run --release -- train                 # default values
cargo run --release -- train 10 8 128        # iterations, games, simulations
```

Each iteration plays a batch of self-play games with PUCT MCTS, stores the
visited positions in a replay buffer, runs gradient steps on the policy/value
losses, and writes a checkpoint to `assets/models/alpha_zero.safetensors`. Use
`--release`; the debug build is roughly 20× slower. The weights start random, so
give it a real run before expecting decent play.

The input encoding and action indices are **canonicalized by color** (Black is
mirrored vertically), so both colors share one representation. Because the
input layout is part of the model, a checkpoint is only valid for the encoding
it was trained with — after changing `encoding.rs`, retrain from scratch.

## Measuring strength (arena)

Once a checkpoint exists, play the network against a quick baseline:

```sh
cargo run --release -- arena 20 64 greedy     # vs 1-ply materialist
cargo run --release -- arena 20 64 random     # vs random move
cargo run --release -- arena 20 64 self       # the network against itself
```

It alternates colors, reports wins/draws and a score. The classic UCB brain is
not used as the reference because it is far too slow for a headless series.

## Project layout

| Path | Purpose |
|------|---------|
| `src/comun.rs` | Shared types and board constants |
| `src/rules.rs` | Pure, Bevy-independent rules (move generation, captures, promotion) |
| `src/game_manager.rs` | Game state, turn management, and the AI handle |
| `src/board.rs` | Bevy rendering, pointer input, and UI |
| `src/train.rs` | Self-play training loop (subcommand `train`) |
| `src/arena.rs` | Headless evaluation against baselines (subcommand `arena`) |
| `src/ia/` | AI implementations (`Brain` trait, Monte Carlo, UCB, AlphaZero) |
| `src/ia/encoding.rs` | Board → input planes and move → action index (4096 actions) |
| `src/ia/neural.rs` | Policy/value network built with `candle` |
| `src/ia/mcts.rs` | PUCT Monte Carlo tree search guided by the network |

## Status

- [x] Rules, move generation, and game loop
- [x] Human vs. computer with selectable modes and restart
- [x] Non-blocking AI computation (runs off the main thread)
- [x] AlphaZero core: board encoding, policy/value network, and PUCT MCTS
- [x] Self-play training loop with checkpoints (`train` subcommand)
- [x] Evaluation arena (`arena` subcommand) and in-game brain switch
- [x] Self-play exploration: ε-greedy moves + temperature sampling of the opening
- [x] Train to a strong level (300 iterations: **99% vs `random`, 98% vs `greedy`**)
- [ ] AlphaZero extras still missing: Dirichlet root noise, tree reuse, self-play vs previous checkpoints

### Measured results

With ~3,600 self-play games (300 iterations × 12 games, 160 simulations), the
network scores **99.0% against `random` (197-1-2) and 97.8% against `greedy`
(192-1-7)** over 200 games each. The simulation curve against `greedy` is
monotonic and nearly saturated: 16 → 97.2%, 128 → 97.8%, 256 → 99.2%
(200 games per point).

Two bugs made the network play almost randomly; both are fixed and covered by
tests:

1. `undo_move` did not restore the turn, so every MCTS simulation left the
   side-to-move desynchronized (the worst one: without it, more search made play
   worse).
2. The value ignored chained captures (the sign was inverted).

See `AGENTS.md` for the invariants.
