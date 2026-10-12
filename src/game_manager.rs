use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::{Arc, Mutex};
use crate::comun::{BLACK_PAWN, BOARD_SIZE, EMPTY, WHITE_PAWN, WHITE_QUEEN, BLACK_QUEEN,
                   Move, Turn, Square, GameStatus, BandPlayer, GamePlayer, INITIAL_BOARD};
use crate::ia::brain::Brain;
use crate::ia::mcts::{Mcts, DEFAULT_SIMULATIONS, MODEL_PATH};
use crate::ia::monte_carlo_impl::MonteCarlo;
use crate::ia::upper_confidence_bounds::UpperConfidenceBounds;
use crate::rules;



#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum BrainType {
    #[default]
    UpperConfidenceBounds,
    /// Classic MCTS alternative; reserved for comparisons and training.
    #[allow(dead_code)]
    MonteCarlo,
    /// AlphaZero neural network + PUCT search.
    #[allow(dead_code)]
    Neural,
}

fn create_brain(brain_type: BrainType) -> Arc<Mutex<dyn Brain>> {
    match brain_type {
        BrainType::MonteCarlo => Arc::new(Mutex::new(MonteCarlo::new())),
        BrainType::UpperConfidenceBounds => Arc::new(Mutex::new(UpperConfidenceBounds::new())),
        BrainType::Neural => Arc::new(Mutex::new(Mcts::load_or_random(MODEL_PATH, DEFAULT_SIMULATIONS))),
    }
}

#[derive(Clone)]
pub struct GameManager{
    board: [[i8; BOARD_SIZE]; BOARD_SIZE],
    moves: Vec<Move>,
    moves_with_no_capture: i8,
    moves_with_no_capture_history: Vec<i8>,
    forced_piece: Option<Square>,
    forced_history: Vec<Option<Square>>,
    turn: Turn,
    turn_history: Vec<Turn>,
    band_player: BandPlayer,
    game_status: GameStatus,
    brain: Arc<Mutex<dyn Brain>>,
    brain_type: BrainType
}

impl Default for GameManager {
    fn default() -> Self {
        Self::new()
    }
}

impl GameManager {
    pub fn new() -> GameManager {
        GameManager::with_options(
            BandPlayer {
                red: GamePlayer::Human,
                black: GamePlayer::Computer,
            },
            BrainType::default(),
        )
    }

    pub fn with_options(band_player: BandPlayer, brain_type: BrainType) -> GameManager {
        GameManager {
            board: INITIAL_BOARD,
            moves: Vec::new(),
            moves_with_no_capture: 0,
            moves_with_no_capture_history: Vec::new(),
            forced_piece: None,
            forced_history: Vec::new(),
            turn: Turn::Red,
            turn_history: Vec::new(),
            band_player,
            game_status: GameStatus::Playing,
            brain: create_brain(brain_type),
            brain_type,
        }
    }

    /// Creates a manager from a specific board and turn. Useful for tests and
    /// headless training.
    #[allow(dead_code)]
    pub fn with_board(board: [[i8; BOARD_SIZE]; BOARD_SIZE], turn: Turn) -> GameManager {
        let mut game_manager = GameManager::with_options(
            BandPlayer {
                red: GamePlayer::Human,
                black: GamePlayer::Human,
            },
            BrainType::default(),
        );
        game_manager.board = board;
        game_manager.turn = turn;
        game_manager.game_status = game_manager.calculate_game_status();
        game_manager
    }

    #[allow(dead_code)]
    pub fn get_forced_piece(&self) -> Option<Square> {
        self.forced_piece
    }

    /// Resets the game keeping the play mode and the AI brain.
    pub fn reset(&mut self) {
        self.board = INITIAL_BOARD;
        self.moves.clear();
        self.moves_with_no_capture = 0;
        self.moves_with_no_capture_history.clear();
        self.forced_piece = None;
        self.forced_history.clear();
        self.turn = Turn::Red;
        self.turn_history.clear();
        self.game_status = GameStatus::Playing;
    }

    pub fn set_band_player(&mut self, band_player: BandPlayer) {
        self.band_player = band_player;
    }

    pub fn get_brain_type(&self) -> BrainType {
        self.brain_type
    }

    /// Switches the computer AI. Rebuilds the brain (loading the checkpoint for
    /// the neural one) so it takes effect immediately.
    pub fn set_brain_type(&mut self, brain_type: BrainType) {
        self.brain_type = brain_type;
        self.brain = create_brain(brain_type);
    }

    pub fn get_hash(&self) -> String {
        // Create a hasher
        let mut hasher = DefaultHasher::new();

        // Hash each field except for `brain`
        self.board.hash(&mut hasher);
        self.turn.hash(&mut hasher);
        self.forced_piece.hash(&mut hasher);

        // Finalize the hash and convert to hex string
        let result = hasher.finish();
        format!("{:x}", result)
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

        self.moves = match self.forced_piece {
            Some(from) => rules::continuation_moves(&self.board, self.turn, from),
            None => rules::all_legal_moves(&self.board, self.turn),
        };

        &self.moves
    }

    pub fn calculate_game_status(&mut self) -> GameStatus {
        if self.moves_with_no_capture >= 16 {
            return GameStatus::Draw;
        }

        let moves = self.get_legal_moves();
        if moves.is_empty() {
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
        self.moves_with_no_capture_history.push(self.moves_with_no_capture);
        self.forced_history.push(self.forced_piece);
        self.turn_history.push(self.turn);

        rules::apply_move(&mut self.board, &m);

        if m.eat.is_some() {
            self.moves_with_no_capture = 0;
        } else {
            self.moves_with_no_capture += 1;
        }

        // If the same piece can keep capturing, the turn stays and that piece is
        // forced to move.
        let can_eat_more = m.eat.is_some()
            && rules::continuation_moves(&self.board, self.turn, m.to)
                .iter()
                .any(|c| c.eat.is_some());

        if can_eat_more {
            self.forced_piece = Some(m.to);
        } else {
            self.forced_piece = None;
            self.switch_turn();
        }

        self.game_status = self.calculate_game_status();
    }

    pub fn undo_move(&mut self, m: Move) {
        rules::undo_apply(&mut self.board, &m);

        self.moves_with_no_capture = self.moves_with_no_capture_history.pop().unwrap();
        self.forced_piece = self.forced_history.pop().unwrap();
        self.turn = self.turn_history.pop().unwrap();

        self.game_status = self.calculate_game_status();
    }

    pub fn get_turn(&self) -> Turn { self.turn }

    pub fn get_game_status(&self) -> GameStatus { self.game_status }

    pub fn computer_plays(&mut self) -> Move {
        let mut brain = self.brain.lock().unwrap();
        brain.get_best_move(self.clone())
    }


    fn is_black_piece(&self, piece: i8) -> bool {
        rules::is_black_piece(piece)
    }

    fn is_white_piece(&self, piece: i8) -> bool {
        rules::is_white_piece(piece)
    }

    fn switch_turn(&mut self) {
        if self.turn == Turn::Red {
            self.turn = Turn::Black;
        } else {
            self.turn = Turn::Red;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty() -> [[i8; BOARD_SIZE]; BOARD_SIZE] {
        [[EMPTY; BOARD_SIZE]; BOARD_SIZE]
    }

    #[test]
    fn opening_has_seven_moves() {
        let mut game_manager = GameManager::new();
        assert_eq!(game_manager.get_legal_moves().len(), 7);
    }

    #[test]
    fn capturing_is_mandatory() {
        let mut board = empty();
        board[2][2] = WHITE_PAWN;
        board[3][3] = BLACK_PAWN;
        board[2][6] = WHITE_PAWN; // has normal moves, but must ignore them

        let mut game_manager = GameManager::with_board(board, Turn::Red);
        let moves = game_manager.get_legal_moves().clone();

        assert_eq!(moves.len(), 1);
        assert_eq!(moves[0].from, Square { row: 2, col: 2 });
        assert_eq!(moves[0].to, Square { row: 4, col: 4 });
    }

    #[test]
    fn maximum_capture_is_mandatory() {
        let mut board = empty();
        // Two-capture chain.
        board[0][0] = WHITE_PAWN;
        board[1][1] = BLACK_PAWN;
        board[3][3] = BLACK_PAWN;
        // Single-capture chain, no continuation.
        board[4][0] = WHITE_PAWN;
        board[5][1] = BLACK_PAWN;

        let mut game_manager = GameManager::with_board(board, Turn::Red);
        let moves = game_manager.get_legal_moves().clone();

        assert_eq!(moves.len(), 1);
        assert_eq!(moves[0].from, Square { row: 0, col: 0 });
        assert_eq!(moves[0].to, Square { row: 2, col: 2 });
    }

    #[test]
    fn capture_chain_keeps_turn_and_forces_piece() {
        let mut board = empty();
        board[0][0] = WHITE_PAWN;
        board[1][1] = BLACK_PAWN;
        board[3][3] = BLACK_PAWN;

        let mut game_manager = GameManager::with_board(board, Turn::Red);
        let first = game_manager.get_legal_moves()[0];
        game_manager.do_move(first);

        assert_eq!(game_manager.get_turn(), Turn::Red);
        assert_eq!(game_manager.get_forced_piece(), Some(Square { row: 2, col: 2 }));

        let second = game_manager.get_legal_moves().clone();
        assert_eq!(second.len(), 1);
        assert_eq!(second[0].to, Square { row: 4, col: 4 });

        game_manager.do_move(second[0]);
        assert_eq!(game_manager.get_turn(), Turn::Black);
        assert_eq!(game_manager.get_forced_piece(), None);
    }

    #[test]
    fn undo_restores_board_turn_and_forced_piece() {
        let mut board = empty();
        board[0][0] = WHITE_PAWN;
        board[1][1] = BLACK_PAWN;
        board[3][3] = BLACK_PAWN;

        let mut game_manager = GameManager::with_board(board, Turn::Red);
        let m = game_manager.get_legal_moves()[0];
        game_manager.do_move(m);
        game_manager.undo_move(m);

        assert_eq!(game_manager.get_board(), &board);
        assert_eq!(game_manager.get_turn(), Turn::Red);
        assert_eq!(game_manager.get_forced_piece(), None);
    }

    #[test]
    fn undo_restores_turn_after_a_normal_move() {
        // A quiet move DOES change the turn; undo must restore it.
        let mut game_manager = GameManager::new();
        let m = game_manager.get_legal_moves()[0];
        assert!(m.eat.is_none());
        let turn_before = game_manager.get_turn();

        game_manager.do_move(m);
        assert_ne!(game_manager.get_turn(), turn_before);

        game_manager.undo_move(m);
        assert_eq!(game_manager.get_turn(), turn_before);
        assert_eq!(game_manager.get_board(), &crate::comun::INITIAL_BOARD);
    }

    #[test]
    fn promotion_turns_pawn_into_queen() {
        let mut board = empty();
        board[6][2] = WHITE_PAWN;

        let mut game_manager = GameManager::with_board(board, Turn::Red);
        let moves = game_manager.get_legal_moves().clone();
        assert!(moves.iter().all(|m| m.promotion));

        game_manager.do_move(moves[0]);
        let row = moves[0].to.row;
        let col = moves[0].to.col;
        assert_eq!(game_manager.get_board()[row][col], WHITE_QUEEN);
    }

    #[test]
    fn brain_type_can_be_switched() {
        let mut game_manager = GameManager::new();
        assert_eq!(game_manager.get_brain_type(), BrainType::UpperConfidenceBounds);

        game_manager.set_brain_type(BrainType::MonteCarlo);
        assert_eq!(game_manager.get_brain_type(), BrainType::MonteCarlo);
    }
}
