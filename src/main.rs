use rustachess::*;

fn main() {
    let mut board = Board::new();
    board.print_board();
    println!();

    board
        .do_move((Color::White, PieceType::Pawn), (File::E, 2), (File::E, 4))
        .unwrap();
    board.print_board();
    println!();

    board
        .do_move(
            (Color::Black, PieceType::Knight),
            (File::G, 8),
            (File::F, 6),
        )
        .unwrap();
    board.print_board();
    println!();

    board
        .do_move((Color::White, PieceType::Pawn), (File::D, 2), (File::D, 3))
        .unwrap();
    board.print_board();
    println!();

    board
        .do_move(
            (Color::Black, PieceType::Knight),
            (File::F, 6),
            (File::E, 4),
        )
        .unwrap();
    board.print_board();
    println!();

    //println!("{:?}", board.valid_moves());
    println!("done");
}
