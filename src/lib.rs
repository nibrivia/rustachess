#![deny(missing_debug_implementations)]
#![deny(missing_docs)]

//! Rust chess engine, not intend for serious use (yet?)

use std::collections::HashMap;
use std::convert::{TryFrom, TryInto};
use std::ops::Not;

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

impl Not for Color {
    type Output = Self;

    fn not(self) -> Self::Output {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }
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

/// Describes check, checkmate, or not
#[derive(Debug, Eq, PartialEq, Copy, Clone)]
pub enum Checkstate {
    /// No check
    Normal,

    /// Check
    Check,

    /// Checkmate
    Checkmate,
}

#[derive(Debug, Copy, Clone)]
/// Describes all of the possible pawn moves
pub enum Pawnmoves {
    /// First pawn move, results in an en passant square
    Zoomies(Option<Position>),

    /// Takes en passant on given square
    TakesEnpassant(Option<Position>),

    /// Promotes to new piece
    Promotes(Option<Piece>),

    /// Usual forward 1 or diagonal takes
    Normal,
}

/// Encodes everything needed for a single move
#[derive(Debug, Copy, Clone)]
pub struct Move {
    /// What piece is moving
    piece: Piece,

    /// Initial piece position
    from: Position,

    /// Final piece position
    to: Position,

    /// Is a piece taken?
    takes: bool,

    /// If pawn, what kind of move is it
    pawnmove: Option<Pawnmoves>,

    /// Final state
    checkstate: Checkstate,
}

/// All of the reasons a move may not be valid
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum MoveError {
    /// The wrong player is moving
    WrongPlayer,

    /// The piece at this position is not the specified one
    WrongPiece,

    /// There is no such piece
    PieceNotFound,

    /// The piece exists, but cannot move that way
    WrongMovement,

    /// Fails to stop a current check
    StaysInCheck,

    /// Puts self in check (wasn't previously)
    PutsInCheck,

    /// Move runs through some other piece
    Bulldozer,

    /// Move takes piece of own color
    Cannibalism,

    /// Generic error
    Generic,
}

/// Board object, represents the current board layout
#[derive(Debug, Clone)]
pub struct Board {
    board: Vec<Vec<Option<Piece>>>,

    history: Vec<Move>,

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
            history: Vec::new(),
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
            // iterates through all +/- directions
            loop {
                for i in 1..=max {
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

                // slightly hacky, but iterates through all +/- combos
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

    /// Returns all the positions where such piece can be found
    pub fn whereare(self: &Self, piece: Piece) -> Vec<Position> {
        let mut positions = Vec::new();
        for (p, pos) in self.cell_pieces() {
            if p == piece {
                positions.push(pos);
            }
        }
        positions
    }

    /// Returns the piece on the given positon
    pub fn whatsat(self: &Self, position: Position) -> Option<Piece> {
        let (row, col) = pos_to_coord(position);
        self.board[row][col]
    }

    //fn will_be_check(self: &Self, mv: Move) -> Result<bool, MoveError> {
    fn will_be_check(self: &Self, piece: Piece, from: Position, to: Position) -> bool {
        let mut new_board = self.clone();
        //new_board.do_move(mv.piece, mv.from, mv.to)?;
        let mv = Move {
            piece: piece,
            from: from,
            to: to,
            takes: false,
            pawnmove: None,
            checkstate: Checkstate::Normal,
        };

        new_board._do_move_unchecked(mv);
        return new_board.is_atleast_check(self.player);
    }

    fn is_atleast_check(self: &Self, color: Color) -> bool {
        let &king_pos = self.whereare((color, PieceType::King)).get(0).unwrap();
        // don't do pin/check checks, but do block checks
        for (_p, moves) in self.valid_moves(!color, true, false) {
            for dest in moves {
                if dest == king_pos {
                    return true;
                }
            }
        }
        false
    }

    /// Returns the current check(mate)/not state
    pub fn checkstate(self: &Self) -> Checkstate {
        if self.is_atleast_check(self.player) {
            self.print_board();
            for (_p, moves) in self.cur_moves() {
                // if there is *any* valid move
                if moves.len() > 0 {
                    return Checkstate::Check;
                }
            }
            self.print_board();
            Checkstate::Checkmate
        } else {
            Checkstate::Normal
        }
    }

    fn check_move(self: &Self, mv: Move) -> Option<MoveError> {
        let (color, _) = mv.piece;
        if color != self.player {
            return Some(MoveError::WrongPlayer);
        }

        // Source checks
        let (from_row, from_col) = pos_to_coord(mv.from);
        if let Some(p) = self.board[from_row][from_col] {
            if p != mv.piece {
                return Some(MoveError::WrongPiece);
            }
        } else {
            return Some(MoveError::PieceNotFound);
        }

        // Destination checks
        let (to_row, to_col) = pos_to_coord(mv.to);
        if let Some((c, _)) = self.board[to_row][to_col] {
            if c == self.player {
                return Some(MoveError::Cannibalism);
            }
        }

        if self.will_be_check(mv.piece, mv.from, mv.to) {
            if self.is_atleast_check(self.player) {
                return Some(MoveError::StaysInCheck);
            } else {
                return Some(MoveError::PutsInCheck);
            }
        }

        let moves = self.valid_moves(self.player, false, false);
        let movement_check = moves.get(&(mv.piece, mv.from)).unwrap();
        if !movement_check.iter().any(|x| *x == mv.to) {
            return Some(MoveError::WrongMovement);
        }

        let blocks = self.valid_moves(self.player, true, false);
        let block_check = blocks.get(&(mv.piece, mv.from)).unwrap();
        if block_check.is_empty() {
            return Some(MoveError::Bulldozer);
        }

        // Catch everything else
        let valids = self.cur_moves();
        let pv = valids.get(&(mv.piece, mv.from)).unwrap();
        if !pv.iter().any(|x| *x == mv.to) {
            return Some(MoveError::Generic);
        }

        None
    }

    fn _do_move_unchecked(self: &mut Self, mv: Move) {
        let (from_row, from_col) = pos_to_coord(mv.from);
        let (to_row, to_col) = pos_to_coord(mv.to);

        self.board[from_row][from_col] = None;
        self.board[to_row][to_col] = Some(mv.piece);

        self.player = !self.player;
        if self.player == Color::White {
            self.cur_turn += 1;
        }

        // TODO en passant square
        // TODO promotion
        // TODO halfclock
        // TODO castling
        // TODO history?
    }

    /// Does the specified move
    pub fn do_move(
        self: &mut Self,
        piece: Piece,
        from: Position,
        to: Position,
    ) -> Result<(), MoveError> {
        let mv = Move {
            piece: piece,
            from: from,
            to: to,
            takes: false,
            pawnmove: None,
            checkstate: Checkstate::Normal,
        };
        // verify validity of move
        if let Some(err) = self.check_move(mv) {
            return Err(err);
        }

        self._do_move_unchecked(mv);
        Ok(())
    }

    /// List the moves available to the current player
    pub fn cur_moves(self: &Self) -> HashMap<(Piece, Position), Vec<Position>> {
        self.valid_moves(self.player, true, true)
    }

    /// Lists all currently valid moves for each player.
    /// If the player is not the current player, should do less checks. Useful for premoves?
    /// do_block_checks chooses whether to go through pieces or no
    /// do_other_checks chooses whether to verify all other rules of chess (pins, etc)
    pub fn valid_moves(
        self: &Self,
        player: Color,
        do_block_checks: bool,
        do_other_checks: bool,
    ) -> HashMap<(Piece, Position), Vec<Position>> {
        //self.active_pieces.iter().map(|p| p.possible_moves())
        //let active_pieces = Vec::new();
        let mut all_moves = HashMap::new();

        for (piece, pos) in self.cell_pieces() {
            let coord = pos_to_coord(pos);
            let (row, col) = coord;
            let (color, kind) = piece;

            if color != player {
                continue;
            }

            let possibles: Vec<(usize, usize)> = match kind {
                PieceType::Pawn => {
                    let mut p: Vec<(usize, usize)> = Vec::new();
                    let (first_row, promote_row, dir): (i64, i64, i64) = match color {
                        Color::White => (1, 7, 1),
                        Color::Black => (6, 0, -1),
                    };
                    let irow = row as i64;
                    let nrow = (irow + dir) as usize;
                    let nnrow = (irow + 2 * dir) as usize;

                    if irow == promote_row {
                        panic!("Pawns should never exist on the last row?");
                    }

                    if let Some(ep_pos) = self.en_passant {
                        let (ep_row, ep_col) = pos_to_coord(ep_pos);
                        if ep_row == nrow
                            && (col > 0 && ep_col == col - 1 || col < 7 && ep_col == col + 1)
                        {
                            p.push((ep_row, ep_col));
                        }
                    }

                    // Move forward 1 if empty
                    if self.board[nrow][col].is_none() {
                        p.push((nrow, col));
                        // if on first row, try to move twice
                        if irow == first_row && (self.board[nnrow][col].is_none()) {
                            p.push((nnrow, col));
                        }
                    }

                    if col > 0 && self.board[nrow][col - 1].is_some() {
                        p.push((nrow, col - 1));
                    }
                    if col < 7 && self.board[nrow][col + 1].is_some() {
                        p.push((nrow, col + 1));
                    }

                    // TODO promotion
                    p
                }
                PieceType::Bishop => {
                    self.go_direction(coord, vec![(1, 1)], 8, do_block_checks, color)
                }
                PieceType::Knight => {
                    self.go_direction(coord, vec![(2, 1), (1, 2)], 1, do_block_checks, color)
                }
                PieceType::Rook => {
                    self.go_direction(coord, vec![(1, 0), (0, 1)], 8, do_block_checks, color)
                }
                PieceType::Queen => self.go_direction(
                    coord,
                    vec![(1, 0), (0, 1), (1, 1)],
                    8,
                    do_block_checks,
                    color,
                ),
                PieceType::King => {
                    // TODO castling
                    self.go_direction(
                        coord,
                        vec![(1, 0), (0, 1), (1, 1)],
                        1,
                        do_block_checks,
                        color,
                    )
                }
            };
            //println!("{:?}: {:?}", piece, possibles);
            // coord -> positions
            let possibles = possibles
                .iter()
                .map(|c| {
                    let (r, c) = c;
                    coord_to_pos(*r as usize, *c as usize)
                })
                .filter(|&to_pos| {
                    if do_other_checks {
                        !self.will_be_check((color, kind), pos, to_pos)
                    } else {
                        true
                    }
                })
                .collect();

            all_moves.insert(((color, kind), pos), possibles);
        }

        all_moves
    }

    /// Prints the board to stdout, should be phased out
    pub fn print_board(self: &Self) {
        for r in 0..=7 {
            for f in 0..=7 {
                if let Some(p) = self.board[7 - r][f] {
                    print!("{} ", to_fen(p));
                //print!("{} ", to_unicode(p));
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

    #[test]
    fn scholars_mate() {
        let mut board = Board::new();
        print!("1.  e4 ");
        board
            .do_move((Color::White, PieceType::Pawn), (File::E, 2), (File::E, 4))
            .unwrap();

        println!(" e5");
        board
            .do_move((Color::Black, PieceType::Pawn), (File::E, 7), (File::E, 5))
            .unwrap();

        print!("2. Bc3 ");
        board
            .do_move(
                (Color::White, PieceType::Bishop),
                (File::F, 1),
                (File::C, 4),
            )
            .unwrap();

        println!("Nc6");
        board
            .do_move(
                (Color::Black, PieceType::Knight),
                (File::B, 8),
                (File::C, 6),
            )
            .unwrap();

        print!("3. Qh5 ");
        board
            .do_move((Color::White, PieceType::Queen), (File::D, 1), (File::H, 5))
            .unwrap();

        println!("Nf6");
        board
            .do_move(
                (Color::Black, PieceType::Knight),
                (File::G, 8),
                (File::F, 6),
            )
            .unwrap();

        print!("4. Qf7#");
        board
            .do_move((Color::White, PieceType::Queen), (File::H, 5), (File::F, 7))
            .unwrap();

        assert!(board.checkstate() == Checkstate::Checkmate);
    }

    #[test]
    fn en_passant() {
        let mut board = Board::new();

        print!("1. e4 ");
        board
            .do_move((Color::White, PieceType::Pawn), (File::E, 2), (File::E, 4))
            .unwrap();

        println!("d5");
        board
            .do_move((Color::Black, PieceType::Pawn), (File::D, 7), (File::D, 5))
            .unwrap();

        print!("2. e5 ");
        board
            .do_move((Color::White, PieceType::Pawn), (File::E, 4), (File::E, 5))
            .unwrap();

        println!("f5");
        board
            .do_move((Color::Black, PieceType::Pawn), (File::F, 7), (File::F, 5))
            .unwrap();

        print!("3. exf6 ");
        board
            .do_move((Color::White, PieceType::Pawn), (File::E, 5), (File::F, 6))
            .unwrap();
    }

    #[test]
    fn takes_and_checks() {
        // A silly game that has takes and pawn takes from both sides
        let mut board = Board::new();

        assert!(board.cur_turn == 1);
        print!("1.  e4 ");
        board
            .do_move((Color::White, PieceType::Pawn), (File::E, 2), (File::E, 4))
            .unwrap();

        assert!(board.cur_turn == 1);
        println!("Nf6");
        board
            .do_move(
                (Color::Black, PieceType::Knight),
                (File::G, 8),
                (File::F, 6),
            )
            .unwrap();

        assert!(board.cur_turn == 2);
        print!("2.  d3 ");
        board
            .do_move((Color::White, PieceType::Pawn), (File::D, 2), (File::D, 3))
            .unwrap();

        assert!(board.cur_turn == 2);
        println!("Nxe4");
        board
            .do_move(
                (Color::Black, PieceType::Knight),
                (File::F, 6),
                (File::E, 4),
            )
            .unwrap();

        print!("3.dxe4 ");
        board
            .do_move((Color::White, PieceType::Pawn), (File::D, 3), (File::E, 4))
            .unwrap();

        println!(" f5");
        board
            .do_move((Color::Black, PieceType::Pawn), (File::F, 7), (File::F, 5))
            .unwrap();

        print!("4. Qd5 ");
        board
            .do_move((Color::White, PieceType::Queen), (File::D, 1), (File::D, 5))
            .unwrap();

        println!("fxe4");
        board
            .do_move((Color::Black, PieceType::Pawn), (File::F, 5), (File::E, 4))
            .unwrap();

        print!("5. Qh5+");
        board
            .do_move((Color::White, PieceType::Queen), (File::D, 5), (File::H, 5))
            .unwrap();
        assert_eq!(board.checkstate(), Checkstate::Check);

        assert_eq!(board.cur_turn, 5);
        println!("  g6");
        board
            .do_move((Color::Black, PieceType::Pawn), (File::G, 7), (File::G, 6))
            .unwrap();
        assert_eq!(board.checkstate(), Checkstate::Normal);

        assert_eq!(board.cur_turn, 6);
        print!("6. Qe5 ");
        board
            .do_move((Color::White, PieceType::Queen), (File::H, 5), (File::E, 5))
            .unwrap();
        assert_eq!(board.checkstate(), Checkstate::Normal);

        println!("  e6");
        board
            .do_move((Color::Black, PieceType::Pawn), (File::E, 7), (File::E, 6))
            .unwrap();
        assert_eq!(board.checkstate(), Checkstate::Normal);

        print!("7. Bh6 ");
        board
            .do_move(
                (Color::White, PieceType::Bishop),
                (File::C, 1),
                (File::H, 6),
            )
            .unwrap();
        assert!(board.checkstate() == Checkstate::Normal);

        println!(" Bb4+");
        board
            .do_move(
                (Color::Black, PieceType::Bishop),
                (File::F, 8),
                (File::B, 4),
            )
            .unwrap();
        assert!(board.checkstate() == Checkstate::Check);

        print!("8.  c3 ");
        board
            .do_move((Color::White, PieceType::Pawn), (File::C, 2), (File::C, 3))
            .unwrap();
        assert!(board.checkstate() == Checkstate::Normal);

        println!(" Bxc3+");
        board
            .do_move(
                (Color::Black, PieceType::Bishop),
                (File::B, 4),
                (File::C, 3),
            )
            .unwrap();
        assert!(board.checkstate() == Checkstate::Check);

        print!("9.bxc3 ");
        board
            .do_move((Color::White, PieceType::Pawn), (File::B, 2), (File::C, 3))
            .unwrap();
        assert!(board.checkstate() == Checkstate::Normal);
    }

    #[test]
    fn wrong_movements() {
        // These are all incorrect moves
        let mut b = Board::new();

        println!("1.  e2f4 ");
        assert_eq!(
            b.do_move((Color::White, PieceType::Pawn), (File::E, 2), (File::F, 4))
                .unwrap_err(),
            MoveError::WrongMovement
        );
        assert!(b.cur_turn == 1);
        assert!(b.player == Color::White);
    }

    #[test]
    fn bulldozers() {
        // These are all incorrect moves
        let mut b = Board::new();

        println!("1. Ra4 ");
        assert_eq!(
            b.do_move((Color::White, PieceType::Rook), (File::A, 1), (File::A, 4))
                .unwrap_err(),
            MoveError::Bulldozer
        );
        assert!(b.cur_turn == 1);
    }

    #[test]
    fn stuck_pawns() {
        // These are all incorrect moves
        let mut b = Board::from_fen("rnbqkbnr/ppp2ppp/8/4pP2/3pP3/8/PPPP2PP/RNBQKBNR w KQkq - 0 4")
            .unwrap();

        //assert_eq!(
        b.do_move((Color::White, PieceType::Pawn), (File::E, 4), (File::E, 5))
            .unwrap_err();
        //MoveError::WrongMovement
        //);
        //assert_eq!(
        b.do_move((Color::White, PieceType::Pawn), (File::D, 2), (File::D, 4))
            .unwrap_err();
        //MoveError::WrongMovement
        //);

        b.do_move((Color::White, PieceType::Pawn), (File::D, 2), (File::D, 3))
            .unwrap();

        //assert_eq!(
        b.do_move((Color::Black, PieceType::Pawn), (File::E, 5), (File::E, 4))
            .unwrap_err();
        //MoveError::WrongMovement
        //);
        //assert_eq!(
        b.do_move((Color::Black, PieceType::Pawn), (File::F, 7), (File::F, 5))
            .unwrap_err();
        //MoveError::WrongMovement
        //);

        b.do_move((Color::Black, PieceType::Pawn), (File::F, 7), (File::F, 6))
            .unwrap();
    }

    #[test]
    fn cannibalism() {
        // These are all incorrect moves
        let mut b = Board::new();

        println!("1. Ra4 ");
        assert_eq!(
            b.do_move((Color::White, PieceType::King), (File::E, 1), (File::F, 2))
                .unwrap_err(),
            MoveError::Cannibalism
        );
        assert!(b.cur_turn == 1);
    }

    #[test]
    fn wrong_pieces() {
        // These are all incorrect moves
        let mut b = Board::new();

        assert!(b.cur_turn == 1);
        println!("1.  e3e4 ");
        assert_eq!(
            b.do_move((Color::White, PieceType::Pawn), (File::E, 3), (File::E, 4))
                .unwrap_err(),
            MoveError::PieceNotFound
        );

        assert!(b.cur_turn == 1);
        println!("1.  Qe2e5 ");
        assert_eq!(
            b.do_move((Color::White, PieceType::Queen), (File::E, 2), (File::E, 4))
                .unwrap_err(),
            MoveError::WrongPiece
        );

        assert!(b.cur_turn == 1);
        println!("1.  e7e5 ");
        assert_eq!(
            b.do_move((Color::Black, PieceType::Pawn), (File::E, 7), (File::E, 5))
                .unwrap_err(),
            MoveError::WrongPlayer
        );
        assert!(b.cur_turn == 1);
    }

    // TODO pins, staying in check, ...
    // TODO castling
    // TODO promotion
    // TODO whereare/whatsat
}
