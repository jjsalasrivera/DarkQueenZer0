use std::collections::HashMap;
use bevy_mod_picking::input::debug::print;
use rand::seq::SliceRandom;
use rand::thread_rng;
use crate::comun::{GameStatus, Move, Turn};
use crate::game_manager::GameManager;
use crate::ia::brain::{BLACKVALUE, Brain, DRAWVALUE, REDVALUE};

const MAX_ITERATIONS: u16 = 800;

pub struct MonteCarlo{
    iterations_for_movement: u16,
    position_counter: u32
}

impl MonteCarlo {
    pub fn new() -> Self{
        MonteCarlo {
            iterations_for_movement: MAX_ITERATIONS,
            position_counter: 0
        }
    }

    fn play_random_game(&mut self, game_manager: &mut GameManager) -> i32 {
        self.position_counter += 1;

        let status = game_manager.get_game_status();

        if status != GameStatus::Playing {
            //game_manager.print_board();
            return match status {
                GameStatus::RedWins => REDVALUE,
                GameStatus::BlackWins => BLACKVALUE,
                _ => DRAWVALUE
            };
        }

        let moves = game_manager.get_legal_moves();

        // get random element from moves
        let mut rng = thread_rng();

        if let Some(&random_move) = moves.choose(&mut rng) {
            game_manager.do_move(random_move);
            let v = self.play_random_game(game_manager);
            game_manager.undo_move(random_move);
            v
        }
        else {
            return match game_manager.get_turn() {
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
            values.push(self.play_random_game(game_manager));
        }

        let suma :i32 = values.iter().sum();
        let count_c = values.iter().count();

        let average = suma as f32 / (iterations as f32);
        println!("Average: {}", average);

        average
    }

}

impl Brain for MonteCarlo {
    fn get_best_move(&mut self, mut game_manager: GameManager) -> Move {
        self.position_counter = 0;
        let moves = game_manager.get_legal_moves();
        let moves_cloned = moves.clone();

        let mut action_dict: HashMap<Move, f32> = HashMap::new();

        let turn_value = match game_manager.get_turn() {
            Turn::Red => REDVALUE,
            Turn::Black => BLACKVALUE
        };

        for m in moves_cloned.iter() {
            game_manager.do_move(*m);
            action_dict.insert(*m,turn_value as f32 * self.monte_carlo_value(&mut game_manager, Some(self.iterations_for_movement)));
            game_manager.undo_move(*m);
        }

        let movement_opt = action_dict.iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(k, _v)| k.clone());

        println!("action_dict: {:?}", action_dict);

        println!("Position counter: {}", self.position_counter);

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
