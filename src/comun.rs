use bevy::prelude::Component;

pub const BOARD_SIZE: usize = 8;
pub const WHITE_PAWN: i8 = 1;
pub const BLACK_PAWN: i8 = -1;
pub const WHITE_QUEEN: i8 = 2;
pub const BLACK_QUEEN: i8 = -2;
pub const EMPTY: i8 = 0;

pub const INITIAL_BOARD: [[i8; BOARD_SIZE]; BOARD_SIZE] = [
    [WHITE_PAWN, EMPTY, WHITE_PAWN, EMPTY, WHITE_PAWN, EMPTY, WHITE_PAWN, EMPTY],
    [EMPTY, WHITE_PAWN, EMPTY, WHITE_PAWN, EMPTY, WHITE_PAWN, EMPTY, WHITE_PAWN],
    [WHITE_PAWN, EMPTY, WHITE_PAWN, EMPTY, WHITE_PAWN, EMPTY, WHITE_PAWN, EMPTY],
    [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
    [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
    [EMPTY, BLACK_PAWN, EMPTY, BLACK_PAWN, EMPTY, BLACK_PAWN, EMPTY, BLACK_PAWN],
    [BLACK_PAWN, EMPTY, BLACK_PAWN, EMPTY, BLACK_PAWN, EMPTY, BLACK_PAWN, EMPTY],
    [EMPTY, BLACK_PAWN, EMPTY, BLACK_PAWN, EMPTY, BLACK_PAWN, EMPTY, BLACK_PAWN],
];

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

#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum GameStatus {
    #[default]
    Playing,
    RedWins,
    BlackWins,
    Draw,
}

#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum GamePlayer {
    #[default]
    Human,
    Computer,
}

#[derive(Clone, Copy, Default, Hash)]
pub struct BandPlayer {
    pub red: GamePlayer,
    pub black: GamePlayer,
}