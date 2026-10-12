use std::collections::HashMap;
use crate::comun::{GameStatus, Move, Turn};
use crate::game_manager::GameManager;
use crate::ia::brain::{BLACKVALUE, Brain, DRAWVALUE, REDVALUE};

const MAX_ITERATIONS: u16 = 700;

pub struct UpperConfidenceBounds {
    iterations_for_movement: u16,
    visits: HashMap<String, f32>,
    differential: HashMap<String, f32>
}

impl UpperConfidenceBounds {
    pub fn new() -> Self {
        UpperConfidenceBounds {
            iterations_for_movement: MAX_ITERATIONS,
            visits: HashMap::new(),
            differential: HashMap::new()
        }
    }

    fn heuristic_value(&mut self, game_manager: &mut GameManager) -> f32 {
        let hash = game_manager.get_hash();

        // Canonical UCB1: Q + c * sqrt(ln(N) / n_i).
        let n = self.visits.get("total").copied().unwrap_or(1.0);
        let ni = self.visits.get(&hash).copied().unwrap_or(1e-5);
        let v = self.differential.get(&hash).copied().unwrap_or(0.0) / ni;
        let c = 1.5;

        v + c * (n.ln() / ni).sqrt()
    }

    fn record(&mut self, game_manager: &mut GameManager, score: f32) {
        let hash = game_manager.get_hash();

        self.visits.insert("total".to_string(), self.visits.get("total").unwrap_or(&1.0) + 1.0);
        self.visits.insert(hash.clone(), self.visits.get(&hash).unwrap_or(&0.0) + 1.0);
        self.differential.insert(hash.clone(), self.differential.get(&hash).unwrap_or(&0.0) + score);
    }

    fn playout_value(&mut self, game_manager: &mut GameManager) -> i32 {
        let status = game_manager.get_game_status();

        if status != GameStatus::Playing {
            //game_manager.print_board();
            let score = match status {
                GameStatus::RedWins => REDVALUE,
                GameStatus::BlackWins => BLACKVALUE,
                _ => DRAWVALUE
            };

            self.record(game_manager, score as f32);
            return score;
        }

        let turn_value = match game_manager.get_turn() {
            Turn::Red => REDVALUE,
            Turn::Black => BLACKVALUE
        };

        let mut action_heuristic_dict: HashMap<Move, f32> = HashMap::new();

        let moves = game_manager.get_legal_moves().clone();
        for m in moves.iter() {
            game_manager.do_move(*m);
            let v = self.heuristic_value(game_manager);
            action_heuristic_dict.insert(*m, v * turn_value as f32);
            game_manager.undo_move(*m);
        }

        let movement_opt = action_heuristic_dict.iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(k, _v)| *k);

        if let Some(m) = movement_opt {
            game_manager.do_move(m);
            let v = self.playout_value(game_manager);
            game_manager.undo_move(m);
            self.record(game_manager, v as f32);
            v
        }
        else {
            match game_manager.get_turn() {
                Turn::Red => BLACKVALUE,
                Turn::Black => REDVALUE
            }
        }
    }

    fn monte_carlo_value(&mut self, game_manager: &mut GameManager, iterations_opt: Option<u16>) -> f32 {
        let mut iterations: u16 = MAX_ITERATIONS;
        if let Some(i) = iterations_opt {
            iterations = i;
        }

        let mut values: Vec<i32> = Vec::new();
        for _ in 0..iterations {
            values.push(self.playout_value(game_manager));
        }

        let suma :i32 = values.iter().sum();
        suma as f32 / (iterations as f32)
    }
}

impl Brain for UpperConfidenceBounds {
    fn get_best_move(&mut self, mut game_manager: GameManager) -> Move {
        let moves = game_manager.get_legal_moves();
        let moves_cloned = moves.clone();

        let mut action_dict: HashMap<Move, f32> = HashMap::new();

        let turn_value = match game_manager.get_turn() {
            Turn::Red => REDVALUE,
            Turn::Black => BLACKVALUE
        };

        for m in moves_cloned.iter() {
            game_manager.do_move(*m);
            action_dict.insert(*m, turn_value as f32 * self.monte_carlo_value(&mut game_manager, Some(self.iterations_for_movement)));
            game_manager.undo_move(*m);
        }

        let movement_opt = action_dict.iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(k, _v)| *k);

        match movement_opt {
            Some(m) => m,
            None => Move {
                turn: game_manager.get_turn(),
                from: Default::default(),
                to: Default::default(),
                eat: None,
                promotion: false
            }
        }
    }
}