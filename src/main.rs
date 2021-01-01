#![deny(missing_debug_implementations)]
#![deny(missing_docs)]

//! Rust chess engine, not intend for serious use (yet?)

fn main() {
    println!("Hello, world!");
}

/// Types of chess pieces and all necessary state
#[derive(Copy, Clone)]
enum PieceType {
    /// Pawn
    Pawn,

    /// Rook
    Rook,

    /// Knight
    Knight,

    /// Bishop
    Bishop,

    /// Queen
    Queen,

    /// King
    King,
}

/// Chess color
#[derive(Copy, Clone)]
enum PieceColor {
    /// White, goes first
    White,

    /// Black
    Black,
}

type Rank = u64;

#[derive(Copy, Clone)]
enum File {
    A = 0,
    B = 1,
    C = 2,
    D = 3,
    E = 4,
    F = 5,
    G = 6,
    H = 7,
}

type Position = (File, Rank);

fn pos_to_coord(pos: Position) -> (u64, u64) {
    let (rank, file) = pos;
    (rank as u64, file - 1)
}

type Move = (Piece, Position);

/// An individual chess piece
struct Piece {
    kind: PieceType,
    color: PieceColor,
    position: Position,
    history: Vec<Position>,
}

impl Piece {
    fn possible_moves(self: &Self) -> Vec<Position> {
        let (r, f) = pos_to_coord(self.position);
        let mut possibles = Vec::new();

        match self.kind {
            PieceType::Pawn => match self.color {
                PieceColor::White => possibles.push((r + 1, f)),
                PieceColor::Black => possibles.push((r - 1, f)),
            },
            PieceType::Bishop => {}
            PieceType::Knight => {}
            PieceType::Rook => {}
            PieceType::Queen => {}
            PieceType::King => {}
        }
        Vec::new()
    }
}

/// Board object, represents the current board layout
struct Board {
    active_pieces: Vec<Piece>,
    taken_pieces: Vec<Piece>,
}

impl Board {
    fn valid_moves(self: &Self) -> Vec<Move> {
        Vec::new()
    }

    fn print_board(self: &Self) {}
}
