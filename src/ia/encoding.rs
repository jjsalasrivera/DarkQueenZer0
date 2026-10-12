//! Board and action encoding for the neural network.
//!
//! The board is encoded as 4 planes of 8x8 from the perspective of the player to
//! move: own men, own kings, opponent men and opponent kings. Black is
//! **mirrored vertically** so that "forward" is always up, just like Red; thus
//! both colors share the same representation. Actions are encoded in that same
//! canonical view as `from * 64 + to`.

use crate::comun::{BLACK_PAWN, BLACK_QUEEN, BOARD_SIZE, Move, Square, Turn};

pub const CELLS: usize = BOARD_SIZE * BOARD_SIZE;
pub const NUM_PLANES: usize = 4;
pub const INPUT_SIZE: usize = NUM_PLANES * CELLS;
pub const NUM_ACTIONS: usize = CELLS * CELLS;

/// Flat index of a square.
pub fn square_index(square: Square) -> usize {
    square.row * BOARD_SIZE + square.col
}

/// Action index of a move (raw coordinates).
#[allow(dead_code)]
pub fn action_index(m: &Move) -> usize {
    square_index(m.from) * CELLS + square_index(m.to)
}

/// Maps a square to the moving player's canonical view: Black's board is
/// mirrored vertically so its own pieces always move "up".
pub fn canonical_square(square: Square, turn: Turn) -> Square {
    match turn {
        Turn::Red => square,
        Turn::Black => Square {
            row: BOARD_SIZE - 1 - square.row,
            col: square.col,
        },
    }
}

/// Action index of a move in the moving player's canonical view.
pub fn canonical_action_index(m: &Move, turn: Turn) -> usize {
    square_index(canonical_square(m.from, turn)) * CELLS
        + square_index(canonical_square(m.to, turn))
}

/// Rebuilds `(from, to)` from an action index.
#[allow(dead_code)]
pub fn action_to_squares(index: usize) -> (Square, Square) {
    let from = index / CELLS;
    let to = index % CELLS;
    (
        Square { row: from / BOARD_SIZE, col: from % BOARD_SIZE },
        Square { row: to / BOARD_SIZE, col: to % BOARD_SIZE },
    )
}

/// Encodes the board from the moving player's canonical perspective.
pub fn encode(board: &[[i8; BOARD_SIZE]; BOARD_SIZE], turn: Turn) -> Vec<f32> {
    let (own_men, own_kings, opp_men, opp_kings) = match turn {
        Turn::Red => (crate::comun::WHITE_PAWN, crate::comun::WHITE_QUEEN, BLACK_PAWN, BLACK_QUEEN),
        Turn::Black => (BLACK_PAWN, BLACK_QUEEN, crate::comun::WHITE_PAWN, crate::comun::WHITE_QUEEN),
    };

    let mut planes = vec![0.0f32; INPUT_SIZE];
    for row in 0..BOARD_SIZE {
        let source_row = match turn {
            Turn::Red => row,
            Turn::Black => BOARD_SIZE - 1 - row,
        };
        for (col, &value) in board[source_row].iter().enumerate() {
            let idx = row * BOARD_SIZE + col;
            if value == own_men {
                planes[idx] = 1.0;
            } else if value == own_kings {
                planes[CELLS + idx] = 1.0;
            } else if value == opp_men {
                planes[2 * CELLS + idx] = 1.0;
            } else if value == opp_kings {
                planes[3 * CELLS + idx] = 1.0;
            }
        }
    }
    planes
}

/// Binary mask of legal actions (length `NUM_ACTIONS`, canonical).
#[allow(dead_code)]
pub fn legal_action_mask(moves: &[Move], turn: Turn) -> Vec<f32> {
    let mut mask = vec![0.0f32; NUM_ACTIONS];
    for m in moves {
        mask[canonical_action_index(m, turn)] = 1.0;
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::comun::INITIAL_BOARD;

    #[test]
    fn action_index_roundtrip() {
        let m = Move {
            from: Square { row: 2, col: 3 },
            to: Square { row: 4, col: 5 },
            turn: Turn::Red,
            eat: None,
            promotion: false,
        };
        let index = action_index(&m);
        let (from, to) = action_to_squares(index);
        assert_eq!(from, m.from);
        assert_eq!(to, m.to);
    }

    #[test]
    fn encoding_has_expected_size_and_counts() {
        let planes = encode(&INITIAL_BOARD, Turn::Red);
        assert_eq!(planes.len(), INPUT_SIZE);

        let own: f32 = planes[0..CELLS].iter().sum();
        let opp: f32 = planes[2 * CELLS..3 * CELLS].iter().sum();
        assert_eq!(own, 12.0);
        assert_eq!(opp, 12.0);
    }

    #[test]
    fn encoding_is_turn_relative() {
        let red = encode(&INITIAL_BOARD, Turn::Red);
        let black = encode(&INITIAL_BOARD, Turn::Black);
        assert_ne!(red, black);
    }

    #[test]
    fn canonical_action_maps_mirrored_moves_together() {
        // A Red advance upward and the symmetric Black advance downward must
        // share the same canonical index.
        let red_move = Move {
            from: Square { row: 2, col: 1 },
            to: Square { row: 3, col: 2 },
            turn: Turn::Red,
            eat: None,
            promotion: false,
        };
        let black_move = Move {
            from: Square { row: 5, col: 1 },
            to: Square { row: 4, col: 2 },
            turn: Turn::Black,
            eat: None,
            promotion: false,
        };

        assert_eq!(
            canonical_action_index(&red_move, Turn::Red),
            canonical_action_index(&black_move, Turn::Black)
        );
    }

    #[test]
    fn mirrored_board_encodes_identically() {
        // A board and its vertical mirror with colors swapped must give the same
        // encoding from the moving player's perspective.
        let mut board = [[0i8; BOARD_SIZE]; BOARD_SIZE];
        board[1][0] = crate::comun::WHITE_PAWN;
        board[2][1] = crate::comun::BLACK_PAWN;

        let mut mirrored = [[0i8; BOARD_SIZE]; BOARD_SIZE];
        for row in 0..BOARD_SIZE {
            for col in 0..BOARD_SIZE {
                mirrored[row][col] = -board[BOARD_SIZE - 1 - row][col];
            }
        }

        assert_eq!(encode(&board, Turn::Red), encode(&mirrored, Turn::Black));
    }

    #[test]
    fn legal_mask_marks_moves() {
        let moves = crate::rules::all_legal_moves(&INITIAL_BOARD, Turn::Red);
        let mask = legal_action_mask(&moves, Turn::Red);
        assert_eq!(mask.len(), NUM_ACTIONS);
        assert_eq!(mask.iter().sum::<f32>(), moves.len() as f32);
    }
}
