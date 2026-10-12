//! Spanish checkers rules engine, independent of Bevy.
//!
//! These functions operate on a pure board so they can be reused both from the
//! UI and from headless AI training.

use crate::comun::{BLACK_PAWN, BLACK_QUEEN, BOARD_SIZE, EMPTY, Move, Square, Turn, WHITE_PAWN, WHITE_QUEEN};

pub type Board = [[i8; BOARD_SIZE]; BOARD_SIZE];

/// Generates all moves (quiet and capturing) for a given piece.
pub fn legal_moves_for_piece(board: &Board, turn: Turn, row: usize, col: usize) -> Vec<Move> {
    let piece = board[row][col];
    let mut moves = Vec::new();
    let from = Square { row, col };

    match piece {
        WHITE_PAWN => {
            if turn == Turn::Red {
                if row < BOARD_SIZE - 1 && col > 0 && board[row + 1][col - 1] == EMPTY {
                    let promotion = row + 1 == BOARD_SIZE - 1;
                    moves.push(Move { from, to: Square { row: row + 1, col: col - 1 }, turn, eat: None, promotion });
                }
                if row < BOARD_SIZE - 1 && col < BOARD_SIZE - 1 && board[row + 1][col + 1] == EMPTY {
                    let promotion = row + 1 == BOARD_SIZE - 1;
                    moves.push(Move { from, to: Square { row: row + 1, col: col + 1 }, turn, eat: None, promotion });
                }
                if row + 2 < BOARD_SIZE && col > 1 && is_black_piece(board[row + 1][col - 1]) && board[row + 2][col - 2] == EMPTY {
                    let promotion = row + 2 == BOARD_SIZE - 1;
                    moves.push(Move { from, to: Square { row: row + 2, col: col - 2 }, turn, eat: Some((Square { row: row + 1, col: col - 1 }, board[row + 1][col - 1])), promotion });
                }
                if row + 2 < BOARD_SIZE && col + 2 < BOARD_SIZE && is_black_piece(board[row + 1][col + 1]) && board[row + 2][col + 2] == EMPTY {
                    let promotion = row + 2 == BOARD_SIZE - 1;
                    moves.push(Move { from, to: Square { row: row + 2, col: col + 2 }, turn, eat: Some((Square { row: row + 1, col: col + 1 }, board[row + 1][col + 1])), promotion });
                }
            }
        }
        WHITE_QUEEN => {
            if turn == Turn::Red {
                moves.extend(queen_moves(board, turn, row, col));
            }
        }
        BLACK_PAWN => {
            if turn == Turn::Black {
                if row > 0 && col > 0 && board[row - 1][col - 1] == EMPTY {
                    let promotion = row - 1 == 0;
                    moves.push(Move { from, to: Square { row: row - 1, col: col - 1 }, turn, eat: None, promotion });
                }
                if row > 0 && col < BOARD_SIZE - 1 && board[row - 1][col + 1] == EMPTY {
                    let promotion = row - 1 == 0;
                    moves.push(Move { from, to: Square { row: row - 1, col: col + 1 }, turn, eat: None, promotion });
                }
                if row >= 2 && col > 1 && is_white_piece(board[row - 1][col - 1]) && board[row - 2][col - 2] == EMPTY {
                    let promotion = row - 2 == 0;
                    moves.push(Move { from, to: Square { row: row - 2, col: col - 2 }, turn, eat: Some((Square { row: row - 1, col: col - 1 }, board[row - 1][col - 1])), promotion });
                }
                if row >= 2 && col + 2 < BOARD_SIZE && is_white_piece(board[row - 1][col + 1]) && board[row - 2][col + 2] == EMPTY {
                    let promotion = row - 2 == 0;
                    moves.push(Move { from, to: Square { row: row - 2, col: col + 2 }, turn, eat: Some((Square { row: row - 1, col: col + 1 }, board[row - 1][col + 1])), promotion });
                }
            }
        }
        BLACK_QUEEN
            if turn == Turn::Black => {
                moves.extend(queen_moves(board, turn, row, col));
            }
        _ => {}
    }

    moves
}

/// Moves of a flying queen (slides and captures any distance along a diagonal).
pub fn queen_moves(board: &Board, turn: Turn, row: usize, col: usize) -> Vec<Move> {
    let mut moves = Vec::new();
    let from = Square { row, col };

    for (dr, dc) in [(1i32, 1i32), (1, -1), (-1, 1), (-1, -1)] {
        let mut captured: Option<(Square, i8)> = None;
        let mut i = 1i32;
        loop {
            let r = row as i32 + dr * i;
            let c = col as i32 + dc * i;
            if r < 0 || c < 0 || r >= BOARD_SIZE as i32 || c >= BOARD_SIZE as i32 {
                break;
            }
            let r = r as usize;
            let c = c as usize;
            if captured.is_none() {
                if board[r][c] == EMPTY {
                    moves.push(Move { from, to: Square { row: r, col: c }, turn, eat: None, promotion: false });
                } else if is_opponent(board[r][c], board[row][col]) {
                    captured = Some((Square { row: r, col: c }, board[r][c]));
                } else {
                    break;
                }
            } else {
                // After the captured piece you can only land on empty squares.
                if board[r][c] == EMPTY {
                    moves.push(Move { from, to: Square { row: r, col: c }, turn, eat: captured, promotion: false });
                } else {
                    break;
                }
            }
            i += 1;
        }
    }

    moves
}

/// Applies a move to a board (includes promotion and clearing the captured
/// piece).
pub fn apply_move(board: &mut Board, m: &Move) {
    board[m.to.row][m.to.col] = board[m.from.row][m.from.col];
    board[m.from.row][m.from.col] = EMPTY;

    if let Some((sq, _)) = m.eat {
        board[sq.row][sq.col] = EMPTY;
    }

    if m.promotion {
        if board[m.to.row][m.to.col] == WHITE_PAWN {
            board[m.to.row][m.to.col] = WHITE_QUEEN;
        } else if board[m.to.row][m.to.col] == BLACK_PAWN {
            board[m.to.row][m.to.col] = BLACK_QUEEN;
        }
    }
}

/// Reverts a move applied with [`apply_move`].
pub fn undo_apply(board: &mut Board, m: &Move) {
    board[m.from.row][m.from.col] = board[m.to.row][m.to.col];
    board[m.to.row][m.to.col] = EMPTY;

    if let Some((sq, piece)) = m.eat {
        board[sq.row][sq.col] = piece;
    }

    if m.promotion {
        if board[m.from.row][m.from.col] == WHITE_QUEEN {
            board[m.from.row][m.from.col] = WHITE_PAWN;
        } else if board[m.from.row][m.from.col] == BLACK_QUEEN {
            board[m.from.row][m.from.col] = BLACK_PAWN;
        }
    }
}

/// Maximum number of pieces the piece at `(row, col)` can capture in its
/// capture chain.
pub fn max_capture_depth(board: &Board, turn: Turn, row: usize, col: usize) -> u32 {
    let captures: Vec<Move> = legal_moves_for_piece(board, turn, row, col)
        .into_iter()
        .filter(|m| m.eat.is_some())
        .collect();

    let mut best = 0;
    for m in captures {
        let mut next = *board;
        apply_move(&mut next, &m);
        let depth = 1 + max_capture_depth(&next, turn, m.to.row, m.to.col);
        best = best.max(depth);
    }
    best
}

/// From a set of captures, keeps only those that are part of a maximum-length
/// chain.
pub fn filter_max_captures(board: &Board, turn: Turn, moves: &[Move]) -> Vec<Move> {
    let mut best = 0;
    let mut depths = Vec::with_capacity(moves.len());
    for m in moves {
        let mut next = *board;
        apply_move(&mut next, m);
        let depth = 1 + max_capture_depth(&next, turn, m.to.row, m.to.col);
        best = best.max(depth);
        depths.push(depth);
    }

    moves
        .iter()
        .zip(depths)
        .filter(|(_, depth)| *depth == best)
        .map(|(m, _)| *m)
        .collect()
}

/// All legal moves for a turn, enforcing mandatory capture and the maximum
/// capture rule.
pub fn all_legal_moves(board: &Board, turn: Turn) -> Vec<Move> {
    let mut raw = Vec::new();
    for row in 0..BOARD_SIZE {
        for col in 0..BOARD_SIZE {
            raw.extend(legal_moves_for_piece(board, turn, row, col));
        }
    }

    let captures: Vec<Move> = raw.iter().filter(|m| m.eat.is_some()).cloned().collect();
    if captures.is_empty() {
        return raw;
    }

    filter_max_captures(board, turn, &captures)
}

/// Mandatory continuation moves for a piece that already started a capture
/// chain.
pub fn continuation_moves(board: &Board, turn: Turn, from: Square) -> Vec<Move> {
    let captures: Vec<Move> = legal_moves_for_piece(board, turn, from.row, from.col)
        .into_iter()
        .filter(|m| m.eat.is_some())
        .collect();

    if captures.is_empty() {
        return Vec::new();
    }

    filter_max_captures(board, turn, &captures)
}

pub fn is_white_piece(piece: i8) -> bool {
    piece > 0
}

pub fn is_black_piece(piece: i8) -> bool {
    piece < 0
}

fn is_opponent(piece: i8, other: i8) -> bool {
    (is_white_piece(piece) && is_black_piece(other)) || (is_black_piece(piece) && is_white_piece(other))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::comun::INITIAL_BOARD;

    fn empty() -> Board {
        [[EMPTY; BOARD_SIZE]; BOARD_SIZE]
    }

    #[test]
    fn red_and_black_open_with_seven_moves() {
        assert_eq!(all_legal_moves(&INITIAL_BOARD, Turn::Red).len(), 7);
        assert_eq!(all_legal_moves(&INITIAL_BOARD, Turn::Black).len(), 7);
    }

    #[test]
    fn flying_queen_moves_along_empty_diagonal() {
        let mut board = empty();
        board[0][0] = WHITE_QUEEN;
        assert_eq!(queen_moves(&board, Turn::Red, 0, 0).len(), 7);
    }

    #[test]
    fn flying_queen_captures_at_distance() {
        let mut board = empty();
        board[0][0] = WHITE_QUEEN;
        board[3][3] = BLACK_PAWN;

        let moves = queen_moves(&board, Turn::Red, 0, 0);
        let captures: Vec<Move> = moves.into_iter().filter(|m| m.eat.is_some()).collect();

        assert_eq!(captures.len(), 4); // landings at (4,4)..(7,7)
        assert!(captures.iter().all(|m| m.eat == Some((Square { row: 3, col: 3 }, BLACK_PAWN))));
    }

    #[test]
    fn apply_and_undo_roundtrip() {
        let mut board = empty();
        board[5][0] = WHITE_PAWN;
        board[4][1] = BLACK_PAWN;

        let move_capture = Move {
            from: Square { row: 5, col: 0 },
            to: Square { row: 3, col: 2 },
            turn: Turn::Red,
            eat: Some((Square { row: 4, col: 1 }, BLACK_PAWN)),
            promotion: true,
        };

        let original = board;
        apply_move(&mut board, &move_capture);
        assert_eq!(board[3][2], WHITE_QUEEN);
        assert_eq!(board[4][1], EMPTY);
        assert_eq!(board[5][0], EMPTY);

        undo_apply(&mut board, &move_capture);
        assert_eq!(board, original);
    }

    #[test]
    fn continuation_only_returns_captures() {
        let mut board = empty();
        board[2][2] = WHITE_QUEEN;
        board[3][3] = BLACK_PAWN;

        let continuations = continuation_moves(&board, Turn::Red, Square { row: 2, col: 2 });
        assert_eq!(continuations.len(), 4); // landings at (4,4)..(7,7)
        assert!(continuations.iter().all(|m| m.eat.is_some()));
    }
}
