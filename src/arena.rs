//! Evaluation arena: pits the trained neural network against a fast opponent
//! to check whether training is worth anything.
//!
//! Usage: `cargo run --release -- arena [games] [simulations] [opponent]`
//! where `opponent` is `greedy` (default, 1-ply materialist player), `random`
//! (random legal move) or `self` (the network against itself).
//!
//! The classic AI (`UpperConfidenceBounds`) is not used here: it is too slow
//! (700 recursive playouts per move) to face in a series.

use std::time::Instant;

use rand::Rng;

use crate::comun::{
    GameStatus, Move, Square, Turn, BLACK_PAWN, BLACK_QUEEN, BOARD_SIZE, INITIAL_BOARD,
    WHITE_PAWN, WHITE_QUEEN,
};
use crate::game_manager::GameManager;
use crate::ia::brain::Brain;
use crate::ia::mcts::{Mcts, MODEL_PATH};

/// Arena opponent.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Opponent {
    Greedy,
    Random,
    SelfPlay,
}

pub fn run() {
    let args: Vec<String> = std::env::args().skip(2).collect();

    let games: usize = args.first().and_then(|v| v.parse().ok()).unwrap_or(20);
    let simulations: u32 = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(64);
    let opponent = match args.get(2).map(|s| s.as_str()) {
        Some("random") => Opponent::Random,
        Some("self") => Opponent::SelfPlay,
        _ => Opponent::Greedy,
    };

    println!(
        "Arena: neural network ({simulations} simulations) vs {} — {games} games (alternating colors)",
        match opponent {
            Opponent::Greedy => "materialist",
            Opponent::Random => "random",
            Opponent::SelfPlay => "itself",
        }
    );

    let mut neural = Mcts::load_or_random(MODEL_PATH, simulations);

    let mut neural_wins = 0u32;
    let mut opponent_wins = 0u32;
    let mut draws = 0u32;
    let mut total_plies = 0usize;
    let start = Instant::now();

    for game in 0..games {
        let neural_color = if game % 2 == 0 { Turn::Red } else { Turn::Black };
        let result = play_game(&mut neural, opponent, simulations, neural_color);
        match result.winner {
            Some(winner) => {
                if winner == result.neural_color {
                    neural_wins += 1;
                } else {
                    opponent_wins += 1;
                }
            }
            None => draws += 1,
        }
        total_plies += result.plies;

        println!(
            "game {:>2}: neural network as {} -> {} in {} plies",
            game + 1,
            match result.neural_color {
                Turn::Red => "Red",
                Turn::Black => "Black",
            },
            match result.winner {
                Some(winner) if winner == result.neural_color => "NEURAL WINS",
                Some(_) => "opponent wins",
                None => "draw",
            },
            result.plies
        );
    }

    let score = (neural_wins as f32 + 0.5 * draws as f32) / games.max(1) as f32;
    println!(
        "\nResult: neural {neural_wins} - {opponent_wins} opponent, {draws} draws  \
         (score {:.1}%)",
        score * 100.0
    );
    println!(
        "Average {:.0} plies/game, {:.1}s total",
        total_plies as f32 / games.max(1) as f32,
        start.elapsed().as_secs_f32()
    );
}

struct GameResult {
    winner: Option<Turn>,
    plies: usize,
    neural_color: Turn,
}

fn play_game(
    neural: &mut Mcts,
    opponent: Opponent,
    simulations: u32,
    neural_color: Turn,
) -> GameResult {
    let mut game_manager = GameManager::with_board(INITIAL_BOARD, Turn::Red);
    let mut opponent_brain: Box<dyn Brain> = match opponent {
        Opponent::Greedy => Box::new(GreedyMover),
        Opponent::Random => Box::new(RandomMover),
        Opponent::SelfPlay => Box::new(Mcts::load_or_random(MODEL_PATH, simulations)),
    };

    let mut plies = 0;
    while game_manager.get_game_status() == GameStatus::Playing && plies < 300 {
        let move_chosen = if game_manager.get_turn() == neural_color {
            if rand::thread_rng().gen_bool(0.1) {
                random_legal_move(&mut game_manager)
            } else {
                neural.get_best_move(game_manager.clone())
            }
        } else {
            opponent_brain.get_best_move(game_manager.clone())
        };
        game_manager.do_move(move_chosen);
        plies += 1;
    }

    let winner = match game_manager.get_game_status() {
        GameStatus::RedWins => Some(Turn::Red),
        GameStatus::BlackWins => Some(Turn::Black),
        _ => None,
    };

    GameResult {
        winner,
        plies,
        neural_color,
    }
}

/// Picks a random legal move.
struct RandomMover;

impl Brain for RandomMover {
    fn get_best_move(&mut self, mut game_manager: GameManager) -> Move {
        random_legal_move(&mut game_manager)
    }
}

fn random_legal_move(game_manager: &mut GameManager) -> Move {
    let moves = game_manager.get_legal_moves().clone();
    let index = rand::thread_rng().gen_range(0..moves.len());
    moves[index]
}

/// One-ply materialist player: picks the move that leaves the best material
/// balance (pawn = 1, queen = 3).
struct GreedyMover;

impl Brain for GreedyMover {
    fn get_best_move(&mut self, mut game_manager: GameManager) -> Move {
        let moves = game_manager.get_legal_moves().clone();
        let mover = game_manager.get_turn();

        moves
            .iter()
            .max_by_key(|m| {
                let mut after = game_manager.clone();
                after.do_move(**m);
                material_score(after.get_board(), mover)
            })
            .copied()
            .unwrap_or_else(|| fallback_move(moves))
    }
}

fn fallback_move(moves: Vec<Move>) -> Move {
    moves.into_iter().next().unwrap_or(Move {
        turn: Turn::Red,
        from: Square { row: 0, col: 0 },
        to: Square { row: 0, col: 0 },
        eat: None,
        promotion: false,
    })
}

fn material_score(board: &[[i8; BOARD_SIZE]; BOARD_SIZE], player: Turn) -> i32 {
    let (own_pawn, own_queen, opp_pawn, opp_queen) = match player {
        Turn::Red => (WHITE_PAWN, WHITE_QUEEN, BLACK_PAWN, BLACK_QUEEN),
        Turn::Black => (BLACK_PAWN, BLACK_QUEEN, WHITE_PAWN, WHITE_QUEEN),
    };

    let mut score = 0;
    for row in board.iter() {
        for &value in row.iter() {
            if value == own_pawn {
                score += 1;
            } else if value == own_queen {
                score += 3;
            } else if value == opp_pawn {
                score -= 1;
            } else if value == opp_queen {
                score -= 3;
            }
        }
    }
    score
}
