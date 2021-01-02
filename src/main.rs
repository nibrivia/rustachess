#![deny(missing_debug_implementations)]
#![deny(missing_docs)]

//! Rust chess engine, not intend for serious use (yet?)

use std::convert::TryFrom;
use std::convert::TryInto;

fn main() {
    let board = Board::new();
    println!("Hello, world!");
}

/// Types of chess pieces and all necessary state
#[derive(Copy, Debug, Clone, PartialEq, Eq)]
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
#[derive(Copy, Debug, Clone, Eq, PartialEq)]
enum PieceColor {
    /// White, goes first
    White,

    /// Black
    Black,
}

type Rank = u64;

#[derive(Copy, Debug, Clone)]
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

impl TryFrom<u64> for File {
    type Error = ();

    fn try_from(v: u64) -> Result<Self, Self::Error> {
        match v {
            x if x == File::A as u64 => Ok(File::A),
            x if x == File::B as u64 => Ok(File::B),
            x if x == File::C as u64 => Ok(File::C),
            x if x == File::D as u64 => Ok(File::D),
            x if x == File::E as u64 => Ok(File::E),
            x if x == File::F as u64 => Ok(File::F),
            x if x == File::G as u64 => Ok(File::G),
            x if x == File::H as u64 => Ok(File::H),
            _ => Err(()),
        }
    }
}

type Position = (File, Rank);

fn pos_to_coord(pos: Position) -> (u64, u64) {
    let (file, rank) = pos;
    (file as u64, rank - 1)
}

fn coord_to_pos(coord: (u64, u64)) -> Position {
    let (x, y) = coord;
    (x.try_into().unwrap(), y + 1)
}

fn parse_pos(pos_str: &str) -> Result<Position, ()> {
    Err(())
}

/// An individual chess piece
type Piece = (PieceColor, PieceType);

fn to_fen(p: Piece) -> String {
    let (color, kind) = p;
    let s = match kind {
        PieceType::Pawn => "p",
        PieceType::Rook => "r",
        PieceType::Knight => "n",
        PieceType::Bishop => "b",
        PieceType::Queen => "q",
        PieceType::King => "k",
    };

    if PieceColor::White == color {
        s.to_uppercase()
    } else {
        s.to_string()
    }
}

/// Board object, represents the current board layout
#[derive(Debug)]
struct Board {
    board: Vec<Vec<Option<Piece>>>,

    en_passant: Option<Position>,
    player: PieceColor,
    cur_turn: u64,
}

impl Board {
    fn new() -> Board {
        Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap()
    }

    fn from_fen(fen_string: &str) -> Result<Board, ()> {
        let parts: Vec<&str> = fen_string.split(" ").collect();
        assert!(parts.len() == 6);

        let pieces = parts[0];
        let player = parts[1];
        let _castle_state = parts[2];
        let en_passant = parts[3];
        let turn = parts[4];
        let _ = parts[5]; // turns since last capture or pawn move, ignored here

        // First the pieces
        let mut board: Vec<Vec<Option<Piece>>> = Vec::new();
        for _ in 0..=7 {
            let mut col: Vec<Option<Piece>> = Vec::new();
            for _ in 0..=7 {
                col.push(None);
            }
            board.push(col);
        }

        let mut r = 7;
        let mut f = 0;
        for c in pieces.chars() {
            if f == 8 {
                if r == 0 {
                    assert!(c == ' ');
                    break;
                }
                assert!(c == '/');
                r -= 1;
                f = 0;
                continue;
            }

            let p: Option<Piece> = match c {
                'p' => Some((PieceColor::Black, PieceType::Pawn)),
                'P' => Some((PieceColor::White, PieceType::Pawn)),
                'r' => Some((PieceColor::Black, PieceType::Rook)),
                'R' => Some((PieceColor::White, PieceType::Rook)),
                'n' => Some((PieceColor::Black, PieceType::Knight)),
                'N' => Some((PieceColor::White, PieceType::Knight)),
                'b' => Some((PieceColor::Black, PieceType::Bishop)),
                'B' => Some((PieceColor::White, PieceType::Bishop)),
                'q' => Some((PieceColor::Black, PieceType::Queen)),
                'Q' => Some((PieceColor::White, PieceType::Queen)),
                'k' => Some((PieceColor::Black, PieceType::King)),
                'K' => Some((PieceColor::White, PieceType::King)),
                '1'..='8' => {
                    f += c.to_digit(10).unwrap() as usize - 1;
                    None
                }
                _ => {
                    eprintln!("Something else...");
                    panic!("Can't parse this");
                }
            };
            board[f][r] = p;

            f += 1;
        }

        // Active player
        let player = if player == "w" {
            PieceColor::White
        } else if player == "b" {
            PieceColor::Black
        } else {
            panic!("Player color {} isn't parsed", player);
        };

        // TODO Castling availability

        // En passant square
        let en_passant = if en_passant == "-" {
            None
        } else {
            Some(parse_pos(en_passant)?)
        };

        // TODO halfmove clock

        // TODO Full turn number
        let cur_turn = turn.parse::<u64>().unwrap();

        Ok(Board {
            board,
            en_passant,
            player,
            cur_turn,
        })
    }

    fn valid_moves(self: &Self) -> Vec<Position> {
        //self.active_pieces.iter().map(|p| p.possible_moves())
        //let active_pieces = Vec::new();
        let mut possibles = Vec::new();
        /*
        for p in self.active_pieces.iter() {
            let (r, f) = pos_to_coord(p.position);
            match p.kind {
                PieceType::Pawn => match p.color {
                    PieceColor::White => possibles.push((r + 1, f)),
                    PieceColor::Black => possibles.push((r - 1, f)),
                },
                PieceType::Bishop => {
                    for i in 0..8 {
                        possibles.push((r + i, f + i));
                        possibles.push((r + i, f - i));
                        possibles.push((r - i, f + i));
                        possibles.push((r - i, f - i));
                    }
                }
                PieceType::Knight => {
                    possibles.push((r + 2, f + 1));
                    possibles.push((r + 2, f - 1));
                    possibles.push((r - 2, f + 1));
                    possibles.push((r - 2, f - 1));
                    possibles.push((r + 1, f + 2));
                    possibles.push((r + 1, f - 2));
                    possibles.push((r - 1, f + 2));
                    possibles.push((r - 1, f - 2));
                }
                PieceType::Rook => {
                    for i in 0..8 {
                        possibles.push((r + i, f));
                        possibles.push((r - i, f));
                        possibles.push((r, f + i));
                        possibles.push((r, f - i));
                    }
                }
                PieceType::Queen => {
                    for i in 0..8 {
                        // diagonals
                        possibles.push((r + i, f + i));
                        possibles.push((r + i, f - i));
                        possibles.push((r - i, f + i));
                        possibles.push((r - i, f - i));

                        // files and ranks
                        possibles.push((r + i, f));
                        possibles.push((r - i, f));
                        possibles.push((r, f + i));
                        possibles.push((r, f - i));
                    }
                }
                PieceType::King => {
                    // diagonals
                    possibles.push((r + 1, f + 1));
                    possibles.push((r + 1, f - 1));
                    possibles.push((r - 1, f + 1));
                    possibles.push((r - 1, f - 1));

                    // files and ranks
                    possibles.push((r + 1, f));
                    possibles.push((r - 1, f));
                    possibles.push((r, f + 1));
                    possibles.push((r, f - 1));
                }
            }
        }
        */
        // Remove off the board moves, make them valid positions
        possibles
            .iter()
            .filter(|c| {
                let (f, r) = c;
                *f <= 8 && *r <= 8
            })
            .map(|c| coord_to_pos(*c))
            .collect()
    }

    fn print_board(self: &Self) {
        for r in 0..=7 {
            for f in 0..=7 {
                if let Some(p) = self.board[f][7 - r] {
                    print!("{} ", to_fen(p));
                } else {
                    print!("  ");
                }
            }
            println!();
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn pos_coord() {
        for f in 0..8 {
            for r in 0..8 {
                let c = (f, r);
                assert_eq!(c, pos_to_coord(coord_to_pos(c)));
            }
        }
    }

    #[test]
    fn fen_test() {
        //let start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        let start = "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2";
        let board = Board::from_fen(start).unwrap();
        println!("{:#?}", board);
        board.print_board();
        assert!(board.player == PieceColor::Black);
        assert!(board.cur_turn == 2);
        assert!(false);
    }
}
