use crate::comun::Move;
use crate::game_manager::GameManager;

pub const REDVALUE: i32 = 1;
pub const BLACKVALUE: i32 = -1;
pub const DRAWVALUE: i32 = 0;

pub trait Brain: Send + Sync {
    fn get_best_move(&mut self, game_manager: GameManager) -> Move;
}
