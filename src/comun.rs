use bevy::prelude::Component;

pub const BOARD_SIZE: usize = 8;
pub const WHITE_PAWN: i8 = 1;
pub const BLACK_PAWN: i8 = -1;
pub const WHITE_QUEEN: i8 = 2;
pub const BLACK_QUEEN: i8 = -2;
pub const EMPTY: i8 = 0;

#[derive(Component, Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Square {
    pub row: usize,
    pub col: usize,
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, Hash)]
pub struct Move {
    pub turn: Turn,
    pub from: Square,
    pub to: Square,
    pub eat: Option<(Square, i8)>,
    pub promotion: bool
}

#[derive(Clone, Debug, Default, Copy, PartialEq, Eq, Hash)]
pub enum Turn {
    #[default]
    Red,
    Black,
}

#[derive(Clone, Default)]
pub enum GameStatus {
    #[default]
    Playing,
    RedWins,
    BlackWins,
    Draw,
}

impl PartialEq for GameStatus {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (GameStatus::Playing, GameStatus::Playing) => true,
            (GameStatus::RedWins, GameStatus::RedWins) => true,
            (GameStatus::BlackWins, GameStatus::BlackWins) => true,
            (GameStatus::Draw, GameStatus::Draw) => true,
            _ => false
        }
    }
}

#[derive(Clone, Default)]
pub enum GamePlayer {
    #[default]
    Human,
    Computer,
}

impl PartialEq for GamePlayer {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (GamePlayer::Human, GamePlayer::Human) => true,
            (GamePlayer::Computer, GamePlayer::Computer) => true,
            _ => false
        }
    }
}

#[derive(Clone, Default)]
pub struct BandPlayer {
    pub red: GamePlayer,
    pub black: GamePlayer,
}