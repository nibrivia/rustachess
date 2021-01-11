#![deny(missing_debug_implementations)]
#![deny(missing_docs)]

//! Rust chess engine, not intend for serious use (yet?)

use std::collections::HashMap;
use std::convert::{TryFrom, TryInto};

/// Types of chess pieces and all necessary state
#[derive(Copy, Debug, Clone, PartialEq, Eq, Hash)]
pub enum PieceType {
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
#[derive(Copy, Debug, Clone, Eq, PartialEq, Hash)]
pub enum Color {
    /// White, goes first
    White,

    /// Black
    Black,
}

type Rank = u64;

#[derive(Copy, Debug, Clone, Eq, PartialEq, Hash)]
/// File enum...
pub enum File {
    /// A file
    A = 0,
    /// B file
    B = 1,
    /// C file
    C = 2,
    /// D file
    D = 3,
    /// E file
    E = 4,
    /// F file
    F = 5,
    /// G file
    G = 6,
    /// H file
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

fn pos_to_coord(pos: Position) -> (usize, usize) {
    let (file, rank) = pos;
    ((rank - 1) as usize, file as usize)
}

fn coord_to_pos(row: usize, col: usize) -> Position {
    ((col as u64).try_into().unwrap(), row as u64 + 1)
}

fn parse_pos(pos_str: &str) -> Result<Position, ()> {
    let chars: Vec<char> = pos_str.chars().collect();
    if chars.len() != 2 {
        return Err(());
    }

    let f = match chars[0] {
        'a' | 'A' => File::A,
        'b' | 'B' => File::B,
        'c' | 'C' => File::C,
        'd' | 'D' => File::D,
        'e' | 'E' => File::E,
        'f' | 'F' => File::F,
        'g' | 'G' => File::G,
        'h' | 'H' => File::H,
        _ => return Err(()),
    };
    println!("{:?}", f);

    let r = chars[1].to_digit(10).unwrap() as u64;

    Ok((f, r))
}

/// An individual chess piece
type Piece = (Color, PieceType);

/// Translates a piece into its unicode emoji
pub fn to_unicode(p: Piece) -> String {
    match p {
        (Color::White, PieceType::Pawn) => "♙",
        (Color::White, PieceType::Rook) => "♖",
        (Color::White, PieceType::Bishop) => "♗",
        (Color::White, PieceType::Knight) => "♘",
        (Color::White, PieceType::Queen) => "♕",
        (Color::White, PieceType::King) => "♔",
        (Color::Black, PieceType::Pawn) => "♟︎",
        (Color::Black, PieceType::Rook) => "♜",
        (Color::Black, PieceType::Bishop) => "♝",
        (Color::Black, PieceType::Knight) => "♞",
        (Color::Black, PieceType::Queen) => "♛",
        (Color::Black, PieceType::King) => "♚",
    }
    .to_string()
}

/// Translates a piece into its FEN character
pub fn to_fen(p: Piece) -> String {
    let (color, kind) = p;
    let s = match kind {
        PieceType::Pawn => "p",
        PieceType::Rook => "r",
        PieceType::Knight => "n",
        PieceType::Bishop => "b",
        PieceType::Queen => "q",
        PieceType::King => "k",
    };

    if Color::White == color {
        s.to_uppercase()
    } else {
        s.to_string()
    }
}

/// Board object, represents the current board layout
#[derive(Debug)]
pub struct Board {
    board: Vec<Vec<Option<Piece>>>,

    en_passant: Option<Position>,
    castling_options: (bool, bool, bool, bool), // White then Black, Queen then King
    player: Color,
    cur_turn: u64,
    halfmove_clock: u64,
}

impl Default for Board {
    fn default() -> Self {
        Self::new()
    }
}

impl Board {
    /// Creates a new board in the default starting position
    pub fn new() -> Board {
        Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap()
    }

    fn from_fen(fen_string: &str) -> Result<Board, ()> {
        let parts: Vec<&str> = fen_string.split(' ').collect();
        assert!(parts.len() == 6);

        let pieces = parts[0];
        let player = parts[1];
        let _castle_state = parts[2];
        let en_passant = parts[3];
        let halfmove = parts[4]; // turns since last capture or pawn move, ignored here
        let turn = parts[5];

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
                'p' => Some((Color::Black, PieceType::Pawn)),
                'P' => Some((Color::White, PieceType::Pawn)),
                'r' => Some((Color::Black, PieceType::Rook)),
                'R' => Some((Color::White, PieceType::Rook)),
                'n' => Some((Color::Black, PieceType::Knight)),
                'N' => Some((Color::White, PieceType::Knight)),
                'b' => Some((Color::Black, PieceType::Bishop)),
                'B' => Some((Color::White, PieceType::Bishop)),
                'q' => Some((Color::Black, PieceType::Queen)),
                'Q' => Some((Color::White, PieceType::Queen)),
                'k' => Some((Color::Black, PieceType::King)),
                'K' => Some((Color::White, PieceType::King)),
                '1'..='8' => {
                    f += c.to_digit(10).unwrap() as usize - 1;
                    None
                }
                _ => {
                    eprintln!("Something else...");
                    panic!("Can't parse this");
                }
            };
            board[r][f] = p;

            f += 1;
        }

        // Active player
        let player = if player == "w" {
            Color::White
        } else if player == "b" {
            Color::Black
        } else {
            panic!("Player color {} isn't parsed", player);
        };

        // TODO Castling availability
        let castling_options = (true, true, true, true);

        // En passant square
        let en_passant = if en_passant == "-" {
            None
        } else {
            Some(parse_pos(en_passant)?)
        };

        // TODO halfmove clock
        let halfmove_clock = halfmove.parse::<u64>().unwrap();

        // TODO Full turn number
        let cur_turn = turn.parse::<u64>().unwrap();

        Ok(Board {
            board,
            en_passant,
            castling_options,
            player,
            cur_turn,
            halfmove_clock,
        })
    }

    fn go_direction(
        self: &Self,
        coord: (usize, usize),
        dirs: Vec<(i64, i64)>,
        max: i64,
        stop_at_piece: bool,
        color: Color,
    ) -> Vec<(usize, usize)> {
        let (r, c) = coord;

        let mut pos = Vec::new();
        for (mut dr, mut dc) in dirs {
            loop {
                for i in 0..=max {
                    if i == 0 {
                        continue;
                    }
                    let cur_r = r as i64 + i as i64 * dr;
                    let cur_c = c as i64 + i as i64 * dc;

                    // stop if out of bounds
                    if cur_r > 7 || cur_r < 0 || cur_c > 7 || cur_c < 0 {
                        break;
                    }
                    let cur_c = cur_c as usize;
                    let cur_r = cur_r as usize;

                    // stop if there's a piece and we're not going through
                    if stop_at_piece && self.board[cur_r][cur_c].is_some() {
                        if let Some((c, _)) = self.board[cur_r][cur_c] {
                            if c != color {
                                pos.push((cur_r, cur_c));
                            }
                        }
                        break;
                    }

                    pos.push((cur_r, cur_c));
                }
                if dc > 0 {
                    dc = -dc;
                } else if dr > 0 {
                    dr = -dr;
                    dc = -dc;
                } else {
                    break;
                }
            }
        }
        pos
    }

    /// Does the specified move
    pub fn do_move(self: &mut Self, piece: Piece, from: Position, to: Position) -> Result<(), ()> {
        let (from_row, from_col) = pos_to_coord(from);
        let (to_row, to_col) = pos_to_coord(to);

        let valids = self.valid_moves();
        let pv = valids.get(&(piece, from)).unwrap();
        assert!(pv.iter().any(|x| *x == to));

        if let Some(p) = self.board[from_row][from_col] {
            if p != piece {
                return Err(());
            }
        } else {
            return Err(());
        }

        self.board[from_row][from_col] = None;
        self.board[to_row][to_col] = Some(piece);

        self.player = match self.player {
            Color::White => Color::Black,
            Color::Black => Color::White,
        };

        // TODO en passant
        // TODO promotion
        // TODO halfclock
        // TODO turn count
        // TODO history?

        Ok(())
    }

    /// Lists all currently valid-ish moves
    pub fn valid_moves(self: &Self) -> HashMap<(Piece, Position), Vec<Position>> {
        //self.active_pieces.iter().map(|p| p.possible_moves())
        //let active_pieces = Vec::new();
        let mut all_moves = HashMap::new();

        for (piece, pos) in self.cell_pieces() {
            let coord = pos_to_coord(pos);
            let (row, col) = coord;
            let (color, kind) = piece;

            if color != self.player {
                continue;
            }

            let possibles: Vec<(usize, usize)> = match kind {
                PieceType::Pawn => {
                    let mut p: Vec<(usize, usize)> = Vec::new();
                    match color {
                        // TODO En passant
                        // TODO Promotion
                        Color::White => {
                            p.push((row + 1, col));
                            if row == 1 {
                                p.push((row + 2, col))
                            }
                            if row < 7 {
                                if col > 0 && self.board[row + 1][col - 1].is_some() {
                                    p.push((row + 1, col - 1));
                                }
                                if col < 7 && self.board[row + 1][col + 1].is_some() {
                                    p.push((row + 1, col + 1));
                                }
                            }
                        }
                        Color::Black => {
                            p.push((row - 1, col));
                            if row == 6 {
                                p.push((row - 2, col));
                            }
                            if row > 0 {
                                if col > 0 && self.board[row - 1][col - 1].is_some() {
                                    p.push((row - 1, col - 1));
                                }
                                if col < 7 && self.board[row - 1][col + 1].is_some() {
                                    p.push((row - 1, col + 1));
                                }
                            }
                        }
                    }
                    p
                }
                PieceType::Bishop => self.go_direction(coord, vec![(1, 1)], 8, true, color),
                PieceType::Knight => self.go_direction(coord, vec![(2, 1), (1, 2)], 1, true, color),
                PieceType::Rook => self.go_direction(coord, vec![(1, 0), (0, 1)], 8, true, color),
                PieceType::Queen => {
                    self.go_direction(coord, vec![(1, 0), (0, 1), (1, 1)], 8, true, color)
                }
                PieceType::King => {
                    self.go_direction(coord, vec![(1, 0), (0, 1), (1, 1)], 1, true, color)
                }
            };
            // Remove off the board moves, make them valid positions
            let possibles = possibles
                .iter()
                .map(|c| {
                    let (r, c) = c;
                    coord_to_pos(*r as usize, *c as usize)
                })
                .collect();

            all_moves.insert(
                ((color, kind), coord_to_pos(row as usize, col as usize)),
                possibles,
            );
        }

        all_moves
    }

    /// Prints the board to stdout, should be phased out
    pub fn print_board(self: &Self) {
        for r in 0..=7 {
            for f in 0..=7 {
                if let Some(p) = self.board[7 - r][f] {
                    //print!("{} ", to_fen(p));
                    print!("{} ", to_unicode(p));
                } else {
                    print!("  ");
                }
            }
            println!();
        }
    }

    /// Returns an array of all pieces on the board with their position
    fn cell_pieces(self: &Self) -> Vec<(Piece, Position)> {
        let mut pieces = Vec::new();
        for (row_i, row) in self.board.iter().enumerate() {
            for (col_i, cell) in row.iter().enumerate() {
                if let Some(p) = cell {
                    pieces.push((*p, coord_to_pos(row_i, col_i)))
                }
            }
        }
        pieces
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn pos_coord() {
        for row in 0..8 {
            for col in 0..8 {
                let c = (row, col);
                assert_eq!(c, pos_to_coord(coord_to_pos(row, col)));
            }
        }
        assert_eq!((File::A, 1), coord_to_pos(0, 0));
        assert_eq!((File::H, 1), coord_to_pos(0, 7));
        assert_eq!((File::H, 8), coord_to_pos(7, 7));
        assert_eq!((File::E, 5), coord_to_pos(4, 4));
    }

    #[test]
    fn fen_init() {
        // starting position
        let start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        let board = Board::from_fen(start).unwrap();
        println!("{:#?}", board);
        board.print_board();

        assert!(board.player == Color::White);
        assert!(board.cur_turn == 1);
        assert!(board.halfmove_clock == 0);
        assert!(board.en_passant.is_none());
        assert!(board.castling_options == (true, true, true, true));
    }

    #[test]
    fn fen_enpassant() {
        // a few moves in
        let start = "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2";
        let board = Board::from_fen(start).unwrap();
        println!("{:#?}", board);
        board.print_board();

        assert!(board.player == Color::White);
        assert!(board.cur_turn == 2);
        assert!(board.halfmove_clock == 0);
        assert!(board.en_passant.is_some());
        let (f, r) = board.en_passant.unwrap();
        assert!(f == File::C);
        assert!(r == 6);
        assert!(board.castling_options == (true, true, true, true));
    }

    #[test]
    fn fen_black() {
        // one more
        let start = "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2";
        let board = Board::from_fen(start).unwrap();
        println!("{:#?}", board);
        board.print_board();
        assert!(board.player == Color::Black);
        assert!(board.cur_turn == 2);
        assert!(board.halfmove_clock == 1);
        assert!(board.en_passant.is_none());
        assert!(board.castling_options == (true, true, true, true));
    }

    #[test]
    fn fen_nocastle() {
        // one more
        let start = "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2";
        let board = Board::from_fen(start).unwrap();
        println!("{:#?}", board);
        board.print_board();
        assert!(board.player == Color::Black);
        assert!(board.cur_turn == 2);
        assert!(board.halfmove_clock == 1);
        assert!(board.en_passant.is_none());
        assert!(board.castling_options == (true, true, true, true));
    }
}
