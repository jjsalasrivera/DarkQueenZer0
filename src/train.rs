//! AlphaZero training: self-play + gradient descent.
//!
//! Run with `cargo run --release -- train`. The loop generates games against
//! itself with MCTS (the tree returns its visit distribution as the policy
//! target), stores positions in a replay buffer and trains the network to
//! predict that policy and the final result (value target). Each iteration
//! writes a checkpoint to `MODEL_PATH`.
//!
//! The weights start random; play improves as it trains.

use std::fs;
use std::path::Path;
use std::time::Instant;

use candle_core::{Device, Tensor};
use candle_nn::ops::log_softmax;
use candle_nn::optim::{AdamW, Optimizer, ParamsAdamW};
use candle_nn::VarMap;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::comun::{GameStatus, Move, Turn, INITIAL_BOARD};
use crate::game_manager::GameManager;
use crate::ia::encoding::{canonical_action_index, encode, INPUT_SIZE, NUM_ACTIONS};
use crate::ia::mcts::{Mcts, MODEL_PATH};
use crate::ia::neural::{self, PolicyValueNet};

/// A training position: encoded state, target policy (MCTS visits), player to
/// move and target value from that player's perspective.
struct Example {
    state: Vec<f32>,
    policy: Vec<f32>,
    value: f32,
}

pub struct TrainConfig {
    pub iterations: usize,
    pub games_per_iter: usize,
    pub simulations: u32,
    pub train_steps: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub epsilon: f32,
    pub temperature: f32,
    pub temperature_plies: usize,
    pub max_plies: usize,
    pub buffer_size: usize,
    pub output: String,
}

impl Default for TrainConfig {
    fn default() -> Self {
        TrainConfig {
            iterations: 3,
            games_per_iter: 2,
            simulations: 32,
            train_steps: 24,
            batch_size: 128,
            learning_rate: 1e-3,
            epsilon: 0.12,
            temperature: 1.0,
            temperature_plies: 12,
            max_plies: 120,
            buffer_size: 20_000,
            output: MODEL_PATH.to_string(),
        }
    }
}

impl TrainConfig {
    /// Supports `train [iterations] [games] [simulations]`.
    fn from_args() -> Self {
        let mut config = TrainConfig::default();
        let args: Vec<String> = std::env::args().skip(2).collect();

        if let Some(value) = args.first().and_then(|v| v.parse().ok()) {
            config.iterations = value;
        }
        if let Some(value) = args.get(1).and_then(|v| v.parse().ok()) {
            config.games_per_iter = value;
        }
        if let Some(value) = args.get(2).and_then(|v| v.parse().ok()) {
            config.simulations = value;
        }

        config
    }
}

pub fn run() {
    run_with_config(TrainConfig::from_args());
}

pub fn run_with_config(config: TrainConfig) {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let net = neural::from_varmap(&varmap, &device).expect("could not create the network");
    let mut optimizer = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW {
            lr: config.learning_rate,
            ..Default::default()
        },
    )
    .expect("could not create the optimizer");

    if let Some(parent) = Path::new(&config.output).parent() {
        fs::create_dir_all(parent).ok();
    }

    let mut rng = StdRng::seed_from_u64(0xD4_2C_0F_0F);
    let mut buffer: Vec<Example> = Vec::new();

    println!(
        "Training AlphaZero: {} iterations x {} games, {} simulations, {} steps/loss",
        config.iterations, config.games_per_iter, config.simulations, config.train_steps
    );

    for iteration in 1..=config.iterations {
        let start = Instant::now();
        let mut wins = [0u32; 3];

        for _ in 0..config.games_per_iter {
            let (examples, status) = self_play(&varmap, &config, &mut rng);
            match status {
                GameStatus::RedWins => wins[0] += 1,
                GameStatus::BlackWins => wins[1] += 1,
                _ => wins[2] += 1,
            }
            buffer.extend(examples);
        }

        if buffer.len() > config.buffer_size {
            let excess = buffer.len() - config.buffer_size;
            buffer.drain(0..excess);
        }

        let mut total_loss = 0.0f32;
        for _ in 0..config.train_steps {
            match train_step(&net, &mut optimizer, &buffer, &config, &device, &mut rng) {
                Ok(loss) => total_loss += loss,
                Err(err) => {
                    eprintln!("[train] gradient step error: {err}");
                    break;
                }
            }
        }

        match varmap.save(&config.output) {
            Ok(()) => {}
            Err(err) => eprintln!("[train] could not save the checkpoint: {err}"),
        }

        let avg_loss = total_loss / config.train_steps.max(1) as f32;
        println!(
            "iter {iteration}/{}: red {}-{} black, draws {}, buffer {}, avg loss {:.4}, {:.1}s",
            config.iterations,
            wins[0],
            wins[1],
            wins[2],
            buffer.len(),
            avg_loss,
            start.elapsed().as_secs_f32()
        );
    }

    println!("Done. Checkpoint at {}", config.output);
}

/// Plays a game against itself and returns the positions with their values.
fn self_play(varmap: &VarMap, config: &TrainConfig, rng: &mut StdRng) -> (Vec<Example>, GameStatus) {
    let mut mcts = Mcts::from_varmap(varmap, config.simulations)
        .expect("could not create the self-play searcher");

    let mut game_manager = GameManager::with_board(INITIAL_BOARD, Turn::Red);
    let mut history: Vec<(Vec<f32>, Vec<f32>, Turn)> = Vec::new();

    let mut ply = 0;
    while game_manager.get_game_status() == GameStatus::Playing && ply < config.max_plies {
        let legal = game_manager.get_legal_moves().clone();
        if legal.is_empty() {
            break;
        }

        let (best_move, policy) = if rng.gen::<f32>() < config.epsilon {
            // Exploration: random move with a one-hot policy.
            let chosen = legal[rng.gen_range(0..legal.len())];
            let mut one_hot = vec![0.0f32; NUM_ACTIONS];
            one_hot[canonical_action_index(&chosen, game_manager.get_turn())] = 1.0;
            (chosen, one_hot)
        } else {
            mcts.search(&game_manager)
        };

        // For the first plies the move is sampled with temperature to diversify
        // openings; afterwards the most visited move is played.
        let move_chosen = if ply < config.temperature_plies {
            sample_from_policy(
                &policy,
                &legal,
                game_manager.get_turn(),
                config.temperature,
                rng,
            )
        } else {
            best_move
        };

        history.push((
            encode(game_manager.get_board(), game_manager.get_turn()),
            policy,
            game_manager.get_turn(),
        ));

        game_manager.do_move(move_chosen);
        ply += 1;
    }

    let status = game_manager.get_game_status();
    let examples = history
        .into_iter()
        .map(|(state, policy, player)| Example {
            state,
            policy,
            value: outcome_value(&status, player),
        })
        .collect();

    (examples, status)
}

/// Samples a move from the policy distribution with temperature.
fn sample_from_policy(
    policy: &[f32],
    legal: &[Move],
    turn: Turn,
    temperature: f32,
    rng: &mut StdRng,
) -> Move {
    let inverse_temperature = 1.0 / temperature.max(1e-3);
    let weights: Vec<f32> = legal
        .iter()
        .map(|m| policy[canonical_action_index(m, turn)].max(0.0).powf(inverse_temperature))
        .collect();
    let sum: f32 = weights.iter().sum();

    if sum <= 0.0 {
        return legal[rng.gen_range(0..legal.len())];
    }

    let mut threshold = rng.gen::<f32>() * sum;
    for (i, weight) in weights.iter().enumerate() {
        threshold -= weight;
        if threshold <= 0.0 {
            return legal[i];
        }
    }

    *legal.last().unwrap()
}

fn outcome_value(status: &GameStatus, player: Turn) -> f32 {
    match status {
        GameStatus::RedWins => {
            if player == Turn::Red { 1.0 } else { -1.0 }
        }
        GameStatus::BlackWins => {
            if player == Turn::Black { 1.0 } else { -1.0 }
        }
        _ => 0.0,
    }
}

/// One gradient-descent step over a batch sampled from the buffer.
fn train_step(
    net: &PolicyValueNet,
    optimizer: &mut AdamW,
    buffer: &[Example],
    config: &TrainConfig,
    device: &Device,
    rng: &mut StdRng,
) -> candle_core::Result<f32> {
    if buffer.is_empty() {
        return Ok(0.0);
    }

    let batch = config.batch_size.min(buffer.len());
    let mut states = Vec::with_capacity(batch * INPUT_SIZE);
    let mut policies = Vec::with_capacity(batch * NUM_ACTIONS);
    let mut values = Vec::with_capacity(batch);

    for _ in 0..batch {
        let example = &buffer[rng.gen_range(0..buffer.len())];
        states.extend_from_slice(&example.state);
        policies.extend_from_slice(&example.policy);
        values.push(example.value);
    }

    let x = Tensor::from_vec(states, (batch, INPUT_SIZE), device)?;
    let target_policy = Tensor::from_vec(policies, (batch, NUM_ACTIONS), device)?;
    let target_value = Tensor::from_vec(values, (batch, 1), device)?;

    let (logits, value) = net.forward(&x)?;

    // Policy loss: cross-entropy against the visit distribution.
    let log_probs = log_softmax(&logits, 1)?;
    let policy_loss = target_policy
        .broadcast_mul(&log_probs)?
        .sum(1)?
        .mean_all()?
        .neg()?;

    // Value loss: mean squared error.
    let value_loss = value.broadcast_sub(&target_value)?.sqr()?.mean_all()?;

    let loss = policy_loss.broadcast_add(&value_loss)?;
    optimizer.backward_step(&loss)?;

    loss.to_scalar::<f32>()
}
