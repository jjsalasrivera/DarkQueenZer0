//! Monte Carlo tree search with PUCT (the AlphaZero "searcher").
//!
//! Each simulation walks the tree choosing the child that maximizes
//! `Q + c_puct * P * sqrt(N) / (1 + n)`, expands the leaf by evaluating the
//! neural network (policy + value) and propagates the value with per-level
//! negation.

use crate::comun::{GameStatus, Move, Turn};
use crate::game_manager::GameManager;
use crate::ia::brain::Brain;
use crate::ia::encoding::{canonical_action_index, encode, NUM_ACTIONS};
use crate::ia::neural::{self, PolicyValueNet};
use candle_core::{Device, Result as CandleResult};
use candle_nn::VarMap;

/// Simulations per move when playing.
pub const DEFAULT_SIMULATIONS: u32 = 160;
const C_PUCT: f32 = 1.5;

/// Default path of the trained checkpoint.
pub const MODEL_PATH: &str = "assets/models/alpha_zero.safetensors";

struct Node {
    visits: u32,
    value_sum: f32,
    children: Vec<Child>,
}

struct Child {
    action: Move,
    prior: f32,
    /// `true` if the move changes player. In chained captures the turn does NOT
    /// change, so the value is not negated during backpropagation.
    flips_turn: bool,
    node: Node,
}

impl Child {
    /// Average value of the child seen from the parent.
    fn q_from_parent(&self) -> f32 {
        if self.node.visits == 0 {
            0.0
        } else {
            let average = self.node.value_sum / self.node.visits as f32;
            if self.flips_turn {
                -average
            } else {
                average
            }
        }
    }
}

impl Node {
    fn new() -> Self {
        Node {
            visits: 0,
            value_sum: 0.0,
            children: Vec::new(),
        }
    }
}

pub struct Mcts {
    net: PolicyValueNet,
    simulations: u32,
    c_puct: f32,
}

impl Mcts {
    pub fn new(simulations: u32) -> Self {
        Mcts {
            net: neural::random().expect("could not initialize the neural network"),
            simulations,
            c_puct: C_PUCT,
        }
    }

    /// Builds the searcher sharing the `varmap` weights. Each call reflects the
    /// current state of the variables, so self-play always uses the latest
    /// weights during training.
    pub fn from_varmap(varmap: &VarMap, simulations: u32) -> CandleResult<Self> {
        let net = neural::from_varmap(varmap, &Device::Cpu)?;
        Ok(Mcts {
            net,
            simulations,
            c_puct: C_PUCT,
        })
    }

    /// Loads a checkpoint if it exists; otherwise uses random weights.
    pub fn load_or_random(path: &str, simulations: u32) -> Self {
        let device = Device::Cpu;
        let mut varmap = VarMap::new();

        // Register the variables by building the network once.
        if let Err(err) = neural::from_varmap(&varmap, &device) {
            eprintln!("[ia] could not initialize the network ({err}); using random weights");
            return Mcts::new(simulations);
        }

        if std::path::Path::new(path).exists() {
            match varmap.load(path) {
                Ok(()) => println!("[ia] weights loaded from {path}"),
                Err(err) => eprintln!("[ia] could not load {path} ({err}); using random weights"),
            }
        } else {
            println!("[ia] no checkpoint at {path}; using random weights (untrained)");
        }

        match neural::from_varmap(&varmap, &device) {
            Ok(net) => Mcts {
                net,
                simulations,
                c_puct: C_PUCT,
            },
            Err(_) => Mcts::new(simulations),
        }
    }

    /// Value of a terminal position from the perspective of the player to move.
    fn terminal_value(game_manager: &GameManager) -> f32 {
        match game_manager.get_game_status() {
            GameStatus::RedWins => {
                if game_manager.get_turn() == Turn::Red { 1.0 } else { -1.0 }
            }
            GameStatus::BlackWins => {
                if game_manager.get_turn() == Turn::Black { 1.0 } else { -1.0 }
            }
            _ => 0.0,
        }
    }

    fn evaluate(&self, game_manager: &GameManager) -> (Vec<f32>, f32) {
        let planes = encode(game_manager.get_board(), game_manager.get_turn());
        self.net
            .evaluate(&planes)
            .unwrap_or_else(|_| (vec![0.0; NUM_ACTIONS], 0.0))
    }

    /// Expands a leaf with the network policy and returns its value.
    fn expand(&mut self, game_manager: &mut GameManager, node: &mut Node) -> f32 {
        let (logits, value) = self.evaluate(game_manager);
        let moves = game_manager.get_legal_moves().clone();

        if moves.is_empty() {
            return -1.0;
        }

        // Softmax restricted to legal actions (canonical indices).
        let turn = game_manager.get_turn();
        let max = moves
            .iter()
            .map(|m| logits[canonical_action_index(m, turn)])
            .fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = moves
            .iter()
            .map(|m| (logits[canonical_action_index(m, turn)] - max).exp())
            .sum();

        for m in moves {
            let prior = (logits[canonical_action_index(&m, turn)] - max).exp() / exp_sum;
            let flips_turn = move_flips_turn(game_manager.get_board(), turn, &m);

            node.children.push(Child {
                action: m,
                prior,
                flips_turn,
                node: Node::new(),
            });
        }

        value
    }

    fn simulate(&mut self, game_manager: &mut GameManager, node: &mut Node) -> f32 {
        if game_manager.get_game_status() != GameStatus::Playing {
            return Self::terminal_value(game_manager);
        }

        if node.children.is_empty() {
            let value = self.expand(game_manager, node);
            node.visits += 1;
            node.value_sum += value;
            return value;
        }

        let idx = self.select(node);
        let action = node.children[idx].action;
        let flips_turn = node.children[idx].flips_turn;

        game_manager.do_move(action);
        let child_value = self.simulate(game_manager, &mut node.children[idx].node);
        game_manager.undo_move(action);

        // The value is negated only if the turn changed (negamax with chained
        // captures).
        let value = if flips_turn { -child_value } else { child_value };
        node.visits += 1;
        node.value_sum += value;
        value
    }

    fn select(&self, node: &Node) -> usize {
        let sqrt_n = (node.visits as f32).sqrt().max(1.0);
        let mut best = 0usize;
        let mut best_score = f32::NEG_INFINITY;

        for (i, child) in node.children.iter().enumerate() {
            let q = child.q_from_parent();
            let u = self.c_puct * child.prior * sqrt_n / (1.0 + child.node.visits as f32);
            let score = q + u;
            if score > best_score {
                best_score = score;
                best = i;
            }
        }

        best
    }

    /// Runs the search and returns `(best_move, policy)`, where the policy is the
    /// normalized visit distribution over the 4096 actions (training target).
    pub fn search(&mut self, game_manager: &GameManager) -> (Move, Vec<f32>) {
        let mut gm = game_manager.clone();
        let moves = gm.get_legal_moves().clone();
        let turn = gm.get_turn();

        if moves.len() <= 1 {
            let mut policy = vec![0.0f32; NUM_ACTIONS];
            if let Some(m) = moves.first() {
                policy[canonical_action_index(m, turn)] = 1.0;
            }
            return (moves[0], policy);
        }

        let mut root = Node::new();
        for _ in 0..self.simulations {
            self.simulate(&mut gm, &mut root);
        }

        let mut policy = vec![0.0f32; NUM_ACTIONS];
        let total: u32 = root.children.iter().map(|c| c.node.visits).sum();
        for child in &root.children {
            policy[canonical_action_index(&child.action, turn)] = child.node.visits as f32;
        }
        if total > 0 {
            for p in policy.iter_mut() {
                *p /= total as f32;
            }
        }

        let best = root
            .children
            .iter()
            .max_by_key(|c| c.node.visits)
            .map(|c| c.action)
            .unwrap_or(moves[0]);

        (best, policy)
    }

    pub fn best_move(&mut self, game_manager: GameManager) -> Move {
        self.search(&game_manager).0
    }
}

impl Brain for Mcts {
    fn get_best_move(&mut self, game_manager: GameManager) -> Move {
        self.best_move(game_manager)
    }
}

/// Decides whether a move changes player. In checkers, a capture that can
/// continue keeps the turn, so it does not flip the sign of the value.
fn move_flips_turn(board: &[[i8; crate::comun::BOARD_SIZE]; crate::comun::BOARD_SIZE], turn: Turn, m: &Move) -> bool {
    if m.eat.is_none() {
        return true;
    }

    let mut board_after = *board;
    crate::rules::apply_move(&mut board_after, m);

    let continues = crate::rules::continuation_moves(&board_after, turn, m.to)
        .iter()
        .any(|c| c.eat.is_some());

    !continues
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::comun::{BLACK_PAWN, BOARD_SIZE, EMPTY, WHITE_PAWN, INITIAL_BOARD};

    #[test]
    fn returns_a_legal_move() {
        let mut reference = GameManager::new();
        let legal = reference.get_legal_moves().clone();

        let mut mcts = Mcts::new(8);
        let chosen = mcts.best_move(GameManager::new());

        assert!(legal.contains(&chosen), "illegal move: {:?}", chosen);
    }

    #[test]
    fn returns_the_only_legal_move() {
        let mut board = [[EMPTY; BOARD_SIZE]; BOARD_SIZE];
        board[6][0] = WHITE_PAWN;

        let mut mcts = Mcts::new(4);
        let chosen = mcts.best_move(GameManager::with_board(board, Turn::Red));

        assert_eq!(chosen.to.row, 7);
        assert_eq!(chosen.to.col, 1);
    }

    #[test]
    fn never_panics_on_full_board() {
        let mut mcts = Mcts::new(1);
        let chosen = mcts.best_move(GameManager::with_board(INITIAL_BOARD, Turn::Red));
        assert!(chosen.turn == Turn::Red);
    }

    #[test]
    fn capture_chain_position_is_playable() {
        let mut board = [[EMPTY; BOARD_SIZE]; BOARD_SIZE];
        board[0][0] = WHITE_PAWN;
        board[1][1] = BLACK_PAWN;
        board[3][3] = BLACK_PAWN;

        let mut mcts = Mcts::new(16);
        let chosen = mcts.best_move(GameManager::with_board(board, Turn::Red));
        assert!(chosen.eat.is_some());
    }

    #[test]
    fn quiet_move_flips_turn_but_capture_chain_does_not() {
        let mut board = [[EMPTY; BOARD_SIZE]; BOARD_SIZE];
        board[0][0] = WHITE_PAWN;
        board[1][1] = BLACK_PAWN;
        board[3][3] = BLACK_PAWN;

        let mut manager = GameManager::with_board(board, Turn::Red);
        let chain_first = manager.get_legal_moves()[0];
        assert!(chain_first.eat.is_some());
        // (2,2)->(4,4) can still capture, so it does NOT change the turn.
        assert!(!move_flips_turn(manager.get_board(), Turn::Red, &chain_first));

        // A quiet move (pawn without capture) does change the turn.
        let mut quiet_board = [[EMPTY; BOARD_SIZE]; BOARD_SIZE];
        quiet_board[2][2] = WHITE_PAWN;
        let mut quiet_manager = GameManager::with_board(quiet_board, Turn::Red);
        let quiet_move = quiet_manager.get_legal_moves()[0];
        assert!(quiet_move.eat.is_none());
        assert!(move_flips_turn(quiet_manager.get_board(), Turn::Red, &quiet_move));
    }

    #[test]
    fn loads_saved_checkpoint_and_returns_legal_move() {
        let path = std::env::temp_dir().join("dq0_checkpoint_test.safetensors");
        let path_str = path.to_string_lossy().to_string();

        let varmap = VarMap::new();
        let _net = neural::from_varmap(&varmap, &Device::Cpu).unwrap();
        varmap.save(&path_str).unwrap();

        let mut mcts = Mcts::load_or_random(&path_str, 4);

        let mut reference = GameManager::new();
        let legal = reference.get_legal_moves().clone();
        let chosen = mcts.best_move(GameManager::new());
        assert!(legal.contains(&chosen));

        std::fs::remove_file(&path).ok();
    }
}
