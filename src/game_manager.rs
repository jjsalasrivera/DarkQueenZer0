use std::sync::{Arc, Mutex};
use crate::comun::{BLACK_PAWN, BOARD_SIZE, EMPTY, WHITE_PAWN, WHITE_QUEEN, BLACK_QUEEN,
                   Move, Turn, Square, GameStatus, BandPlayer, GamePlayer};
use crate::ia::brain::Brain;
use crate::ia::monte_carlo_impl::MonteCarlo;

const INITIAL_BOARD: [[i8; BOARD_SIZE]; BOARD_SIZE] = [
    [WHITE_PAWN, EMPTY, WHITE_PAWN, EMPTY, WHITE_PAWN, EMPTY, WHITE_PAWN, EMPTY],
    [EMPTY, WHITE_PAWN, EMPTY, WHITE_PAWN, EMPTY, WHITE_PAWN, EMPTY, WHITE_PAWN],
    [WHITE_PAWN, EMPTY, WHITE_PAWN, EMPTY, WHITE_PAWN, EMPTY, WHITE_PAWN, EMPTY],
    [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
    [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
    [EMPTY, BLACK_PAWN, EMPTY, BLACK_PAWN, EMPTY, BLACK_PAWN, EMPTY, BLACK_PAWN],
    [BLACK_PAWN, EMPTY, BLACK_PAWN, EMPTY, BLACK_PAWN, EMPTY, BLACK_PAWN, EMPTY],
    [EMPTY, BLACK_PAWN, EMPTY, BLACK_PAWN, EMPTY, BLACK_PAWN, EMPTY, BLACK_PAWN],
];

#[derive(Clone)]
pub struct GameManager{
    board: [[i8; BOARD_SIZE]; BOARD_SIZE],
    moves: Vec<Move>,
    moves_with_no_capture: i8,
    moves_with_no_capture_history: Vec<i8>,
    turn: Turn,
    band_player: BandPlayer,
    game_status: GameStatus,
    brain: Arc<Mutex<dyn Brain>>
}

impl Default for GameManager {
    fn default() -> Self {
        Self::new()
    }
}

impl GameManager {
    pub fn new() -> GameManager {
        GameManager {
            board: INITIAL_BOARD.clone(),
            moves: Vec::new(),
            moves_with_no_capture: 0,
            moves_with_no_capture_history: Vec::new(),
            turn: Turn::Red,
            band_player: BandPlayer {
                red: GamePlayer::Human,
                black: GamePlayer::Computer,
            },
            game_status: GameStatus::Playing,
            brain: Arc::new(Mutex::new(MonteCarlo::new()))
        }
    }

    #[allow(dead_code)]
    pub fn print_board(&self) {
        println!("  0 1 2 3 4 5 6 7");
        println!("  ---------------");
        for (i, row) in self.board.iter().enumerate().rev() {
            print!("{} ", 8 - i - 1);
            for &cell in row {
                let symbol = match cell {
                    WHITE_PAWN => "●",
                    BLACK_PAWN => "○",
                    WHITE_QUEEN => "◉",
                    BLACK_QUEEN => "◎",
                    EMPTY => "·",
                    _ => "?",
                };
                print!("{} ", symbol);
            }
            println!("| {}", 8 - i);
        }
        println!("  ---------------");
        println!("  0 1 2 3 4 5 6 7");
    }

    pub fn get_band_player(&self) -> &BandPlayer { &self.band_player }

    pub fn get_board(&self) -> &[[i8; BOARD_SIZE]; BOARD_SIZE] {
        &self.board
    }

    pub fn get_legal_moves(&mut self) -> &Vec<Move> {
        self.moves.clear();
        let mut raw_moves: Vec<Move> = Vec::new();

        for row in 0..BOARD_SIZE {
            for col in 0..BOARD_SIZE {
                let l_moves = match self.turn {
                    Turn::Red => self.get_legal_moves_for_piece(row, col),
                    Turn::Black => self.get_legal_moves_for_piece(row, col)
                };

                raw_moves.extend(l_moves.clone());
            }
        }

        // if there is a eat move, delete all non eat moves
        let mut has_eat = false;
        for m in raw_moves.iter() {
            if m.eat.is_some() {
                has_eat = true;
                break;
            }
        }

        if has_eat {
            for m in raw_moves.iter() {
                if m.eat.is_some() {
                    self.moves.push(m.clone());
                }
            }
        } else {
            self.moves = raw_moves;
        }

        &self.moves
    }

    pub fn calculate_game_status(&mut self) -> GameStatus {
        if self.moves_with_no_capture >= 16 {
            return GameStatus::Draw;
        }

        let moves = self.get_legal_moves();
        if moves.len() == 0 {
            if self.turn == Turn::Red {
                return GameStatus::BlackWins;
            } else {
                return GameStatus::RedWins;
            }
        }

        let mut white_pieces = 0;
        let mut black_pieces = 0;

        for row in 0..BOARD_SIZE {
            for col in 0..BOARD_SIZE {
                if self.is_white_piece(self.board[row][col]) {
                    white_pieces += 1;
                } else if self.is_black_piece(self.board[row][col]) {
                    black_pieces += 1;
                }
            }
        }

        if white_pieces == 0 {
            return GameStatus::BlackWins
        } else if black_pieces == 0 {
            return GameStatus::RedWins
        }

        GameStatus::Playing
    }

    pub fn do_move(&mut self, m: Move) {
        let from = m.from;
        let to = m.to;
        let eat = m.eat;

        self.board[to.row][to.col] = self.board[from.row][from.col];
        self.board[from.row][from.col] = EMPTY;

        self.moves_with_no_capture_history.push(self.moves_with_no_capture);

        let mut can_eat = false;
        if let Some(eat) = eat {
            self.board[eat.0.row][eat.0.col] = EMPTY;
            self.moves_with_no_capture = 0;

            // check if you can eat again
            let moves = self.get_legal_moves_for_piece(to.row, to.col);
            for m in moves.iter() {
                if m.eat.is_some() {
                    can_eat = true;
                    break;
                }
            }
        }
        else {
            self.moves_with_no_capture += 1;
        }

        if m.promotion && self.board[to.row][to.col] == WHITE_PAWN {
            self.board[to.row][to.col] = WHITE_QUEEN;
        }
        else if m.promotion && self.board[to.row][to.col] == BLACK_PAWN {
            self.board[to.row][to.col] = BLACK_QUEEN;
        }

        if !can_eat {
            self.switch_turn();
        }

        self.game_status = self.calculate_game_status();
    }

    pub fn undo_move(&mut self, m: Move) {
        let from = m.from;
        let to = m.to;
        let eat = m.eat;

        self.board[from.row][from.col] = self.board[to.row][to.col];
        self.board[to.row][to.col] = EMPTY;

        if let Some(eat) = eat {
            self.board[eat.0.row][eat.0.col] = eat.1;
        }

        if m.promotion {
            if self.board[from.row][from.col] == WHITE_QUEEN {
                self.board[from.row][from.col] = WHITE_PAWN;
            }
            else if self.board[from.row][from.col] == BLACK_QUEEN {
                self.board[from.row][from.col] = BLACK_PAWN;
            }
        }

        self.moves_with_no_capture = self.moves_with_no_capture_history.pop().unwrap();

        self.game_status = self.calculate_game_status();
    }

    pub fn get_turn(&self) -> Turn { self.turn }

    pub fn get_game_status(&self) -> GameStatus { self.game_status.clone() }

    pub fn computer_plays(&mut self) -> Move {
        let mut brain = self.brain.lock().unwrap();
        brain.get_best_move(self.clone())
    }

    fn get_legal_moves_for_piece(&self, row: usize, col: usize) -> Vec<Move> {
        let piece = self.board[row][col];
        let mut moves = Vec::new();
        let cell_from = Square { row, col };

        match piece {
            WHITE_PAWN => {
                if self.turn == Turn::Red {
                    if row < BOARD_SIZE - 1 && col > 0 && self.board[row + 1][col - 1] == EMPTY {
                        let promotion = row + 1 == BOARD_SIZE - 1;
                        moves.push(Move { from: cell_from.clone(), to: Square{ row: row + 1, col: col - 1 }, turn: self.turn, eat: None, promotion });
                    }
                    if row < BOARD_SIZE - 1 && col < BOARD_SIZE - 1 && self.board[row + 1][col + 1] == EMPTY {
                        let promotion = row + 1 == BOARD_SIZE - 1;
                        moves.push(Move { from: cell_from.clone(), to: Square{ row: row + 1, col: col + 1 }, turn: self.turn, eat: None, promotion });
                    }
                    if row < BOARD_SIZE - 2 && col > 1 && self.is_black_piece(self.board[row + 1][col - 1]) && self.board[row + 2][col - 2] == EMPTY {
                        let promotion = row + 2 == BOARD_SIZE - 1;
                        moves.push(Move { from: cell_from.clone(), to: Square{ row: row + 2, col: col - 2 }, turn: self.turn, eat: Some((Square{ row: row + 1, col: col - 1 }, self.get_piece_at(row + 1, col - 1))), promotion });
                    }
                    if row < BOARD_SIZE - 2 && col < BOARD_SIZE - 2 && self.is_black_piece(self.board[row + 1][col + 1]) && self.board[row + 2][col + 2] == EMPTY {
                        let promotion = row + 2 == BOARD_SIZE - 1;
                        moves.push(Move { from: cell_from.clone(), to: Square{ row: row + 2, col: col + 2 }, turn: self.turn, eat: Some((Square{ row: row + 1, col: col + 1 }, self.get_piece_at(row + 1, col + 1))), promotion });
                    }
                }
            },
            WHITE_QUEEN => {
                if self.turn == Turn::Red {
                    moves.extend(self.get_queen_moves(row, col));
                }
            },
            BLACK_PAWN => {
                if self.turn == Turn::Black {
                    if row > 0 && col > 0 && self.board[row - 1][col - 1] == EMPTY {
                        let promotion = row - 1 == 0;
                        moves.push(Move { from: cell_from.clone(), to: Square{ row: row - 1, col: col - 1 }, turn: self.turn, eat: None, promotion });
                    }
                    if row > 0 && col < BOARD_SIZE - 1 && self.board[row - 1][col + 1] == EMPTY {
                        let promotion = row - 1 == 0;
                        moves.push(Move { from: cell_from.clone(), to: Square{ row: row - 1, col: col + 1 }, turn: self.turn, eat: None, promotion });
                    }
                    if row > 1 && col > 1 && self.is_white_piece(self.board[row - 1][col - 1]) && self.board[row - 2][col - 2] == EMPTY {
                        let promotion = row - 2 == 0;
                        moves.push(Move { from: cell_from.clone(), to: Square{ row: row - 2, col: col - 2 }, turn: self.turn, eat: Some((Square{ row: row - 1, col: col - 1 }, self.get_piece_at(row - 1, col - 1))), promotion });
                    }
                    if row > 1 && col < BOARD_SIZE - 2 && self.is_white_piece(self.board[row - 1][col + 1]) && self.board[row - 2][col + 2] == EMPTY {
                        let promotion = row - 2 == 0;
                        moves.push(Move { from: cell_from.clone(), to: Square{ row: row - 2, col: col + 2 }, turn: self.turn, eat: Some((Square{ row: row - 1, col: col + 1 }, self.get_piece_at(row - 1, col + 1))), promotion });
                    }
                }
            },
            BLACK_QUEEN => {
                if self.turn == Turn::Black {
                    moves.extend(self.get_queen_moves(row, col));
                }
            },
            _ => {}
        };

        moves
    }

    fn get_queen_moves(&self, row: usize, col: usize) -> Vec<Move> {
        let mut moves = Vec::new();
        let cell_from = Square { row, col };
        let mut eated_square: Option<(Square, i8)> = None;

        for i in 1..BOARD_SIZE {
            if row + i < BOARD_SIZE && col + i < BOARD_SIZE {
                if self.board[row + i][col + i] == EMPTY {
                    moves.push(Move { from: cell_from.clone(), to: Square{ row: row + i, col: col + i }, turn: self.turn, eat: eated_square.clone(), promotion: false});
                }
                else if self.board[row + i][col + i] * self.board[row][col] < 0 && eated_square.is_none() {
                    eated_square = Some((Square{ row: row + i, col: col + i }, self.get_piece_at(row + i, col + i)));
                }
                else {
                    break;
                }
            }
        }

        eated_square = None;
        for i in 1..BOARD_SIZE {
            if row + i < BOARD_SIZE && col >= i {
                if self.board[row + i][col - i] == EMPTY {
                    moves.push(Move { from: cell_from.clone(), to: Square{ row: row + i, col: col - i }, turn: self.turn, eat: eated_square.clone(), promotion: false});
                }
                else if self.board[row + i][col - i] * self.board[row][col] < 0 && eated_square.is_none() {
                    eated_square = Some((Square{ row: row + i, col: col - i }, self.get_piece_at(row + i, col - i)));
                }
                else {
                    break;
                }
            }
        }

        eated_square = None;
        for i in 1..BOARD_SIZE {
            if row >= i && col + i < BOARD_SIZE {
                if self.board[row - i][col + i] == EMPTY {
                    moves.push(Move { from: cell_from.clone(), to: Square{ row: row - i, col: col + i }, turn: self.turn, eat: eated_square.clone(), promotion: false });
                }
                else if self.board[row - i][col + i] * self.board[row][col] < 0 && eated_square.is_none() {
                    eated_square = Some((Square{ row: row - i, col: col + i }, self.get_piece_at(row - i, col + i)));
                }
                else {
                    break;
                }
            }
        }

        eated_square = None;
        for i in 1..BOARD_SIZE {
            if row  >= i && col >= i {
                if self.board[row - i][col - i] == EMPTY {
                    moves.push(Move { from: cell_from.clone(), to: Square{ row: row - i, col: col - i }, turn: self.turn, eat: eated_square.clone(), promotion: false });
                }
                else if self.board[row - i][col - i] * self.board[row][col] < 0 && eated_square.is_none() {
                    eated_square = Some((Square{ row: row - i, col: col - i }, self.get_piece_at(row - i, col - i)));
                }
                else {
                    break;
                }
            }
        }

        moves
    }

    fn is_black_piece(&self, piece: i8) -> bool {
        piece < 0
    }

    fn is_white_piece(&self, piece: i8) -> bool {
        piece > 0
    }

    fn get_piece_at(&self, row: usize, col: usize) -> i8 {
        self.board[row][col]
    }

    fn switch_turn(&mut self) {
        if self.turn == Turn::Red {
            self.turn = Turn::Black;
        } else {
            self.turn = Turn::Red;
        }
    }
}
