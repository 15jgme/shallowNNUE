use chess::{ChessMove, Color, Square, Piece, Board};

type ReorientedSq = i16;

pub(crate) fn orient(sq: Square, colour: Color) -> ReorientedSq {
    match colour{
        Color::White => sq.to_int() as ReorientedSq,
        Color::Black => 63 - (sq.to_int() as ReorientedSq),
    }
}

pub(crate) fn piece_index(piece: Piece, own_piece: bool) -> u16 {
    // Gets a piece index
    if own_piece{
        piece as u16
    } else {
        piece as u16 + 6
    }
}

pub(crate) fn get_index(piece: Piece, own_piece: bool, sq_reoriented: ReorientedSq) -> u16 {
    // index = piece_map_own[piece.piece_type] * 64 + sq_reoriented python code
    piece_index(piece, own_piece) * 64 + (sq_reoriented as u16)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PieceValueChange{
    Place = 1,
    Remove = -1,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct PieceMove{
    // A struct to capture the change of a single piece (add or subtract)
    pub index: u16,
    pub value: PieceValueChange, // Should be only -1 or 1
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum MoveType{
    NonCapture([PieceMove; 2]), // 2-bit change
    Promote([PieceMove; 2]),    // 2-bit change
    Capture([PieceMove; 3]),    // 3-bit change
    Castle([PieceMove; 4]),     // 4-bit change
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct BitMove{
    pub mve: MoveType,
}

impl BitMove{
    pub(crate) fn new(chess_move: ChessMove, turn: Color, pre_move_board: Board) -> Result<BitMove, ()>{
        // figure out what type of move this is (MoveType enum)
        
        // Castle check
        if chess_move.to_string() == "O-O" {
            // Kingside castle

            // For both white and black, the own perspecctive of casteling looks like the white perspective
            let king_place = PieceMove{index: get_index(Piece::King, true, orient(Square::G1, Color::White)), value: PieceValueChange::Place};
            let king_remove = PieceMove{index: get_index(Piece::King, true, orient(Square::E1, Color::White)), value: PieceValueChange::Remove};
            let rook_place = PieceMove{index: get_index(Piece::Rook, true, orient(Square::F1, Color::White)), value: PieceValueChange::Place};
            let rook_remove = PieceMove{index: get_index(Piece::Rook, true, orient(Square::H1, Color::White)), value: PieceValueChange::Remove};
            
            let mve: MoveType = MoveType::Castle([king_place, king_remove, rook_place, rook_remove]);

            return Ok(BitMove{mve})
        } else if chess_move.to_string() == "O-O-O" {
            // Queenside castle

            // For both white and black, the own perspecctive of casteling looks like the white perspective
            let king_place = PieceMove{index: get_index(Piece::King, true, orient(Square::C1, Color::White)), value: PieceValueChange::Place};
            let king_remove = PieceMove{index: get_index(Piece::King, true, orient(Square::E1, Color::White)), value: PieceValueChange::Remove};
            let rook_place = PieceMove{index: get_index(Piece::Rook, true, orient(Square::D1, Color::White)), value: PieceValueChange::Place};
            let rook_remove = PieceMove{index: get_index(Piece::Rook, true, orient(Square::A1, Color::White)), value: PieceValueChange::Remove};
            
            let mve: MoveType = MoveType::Castle([king_place, king_remove, rook_place, rook_remove]);
            return Ok(BitMove{mve})
        }

        // Promotion check
        match chess_move.get_promotion(){
            Some(promotion_piece) => {
                let piece_remove: PieceMove = PieceMove { index: get_index(pre_move_board.piece_on(chess_move.get_source()).expect("Source sq should have a piece during promote"), true, orient(chess_move.get_source(), turn)), value: PieceValueChange::Remove };
                let piece_add: PieceMove = PieceMove { index: get_index(promotion_piece, true, orient(chess_move.get_dest(), turn)), value: PieceValueChange::Place };

                let mve: MoveType = MoveType::Promote([piece_add, piece_remove]);
                return Ok(BitMove{mve})
            },
            None => {},
        }

        
        match pre_move_board.color_on(chess_move.get_dest()){
            Some(color) => {
                if color != turn {
                    // Capture move
                    let captured_piece = PieceMove {index: get_index(pre_move_board.piece_on(chess_move.get_dest()).expect("Dest sq should have a piece in a capture"), false, orient(chess_move.get_dest(), turn)), value: PieceValueChange::Remove};
                    let destination_piece = PieceMove {index: get_index(pre_move_board.piece_on(chess_move.get_source()).expect("Source sq should have a piece"), true, orient(chess_move.get_dest(), turn)), value: PieceValueChange::Place};
                    let source_piece = PieceMove {index: get_index(pre_move_board.piece_on(chess_move.get_source()).expect("Source sq should have a piece"), true, orient(chess_move.get_source(), turn)), value: PieceValueChange::Remove};

                    let mve: MoveType = MoveType::Capture([captured_piece, destination_piece, source_piece]);
                    return Ok(BitMove{mve})
                } else {
                    // Can't capture own piece
                    return Err(())
                }
            },
            None => {
                /* No piece on target square */
                // Non-capture
                let destination_piece = PieceMove {index: get_index(pre_move_board.piece_on(chess_move.get_source()).expect("Source sq should have a piece"), true, orient(chess_move.get_dest(), turn)), value: PieceValueChange::Place};
                let source_piece = PieceMove {index: get_index(pre_move_board.piece_on(chess_move.get_source()).expect("Source sq should have a piece"), true, orient(chess_move.get_source(), turn)), value: PieceValueChange::Remove};

                let mve: MoveType = MoveType::NonCapture([destination_piece, source_piece]);
                return Ok(BitMove{mve})
            },
        }
        Err(())
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orient() {
        assert_eq!(Square::A1.to_int() as ReorientedSq, orient(Square::H8, Color::Black));
        assert_eq!(Square::A1.to_int() as ReorientedSq, orient(Square::A1, Color::White));
    }

    #[test]
    fn test_piece_index() {
       assert_eq!(piece_index(Piece::King, false), 11);
       assert_eq!(piece_index(Piece::King, true), 5);
       assert_eq!(piece_index(Piece::Bishop, false), 8);
       assert_eq!(piece_index(Piece::Bishop, true), 2);
    }

    #[test]
    fn test_default_bitmove() {
        let board: Board = Board::default();
        let mve: ChessMove = ChessMove::new(Square::E2, Square::E4, None);
        let bitmove: BitMove = BitMove::new(mve, board.side_to_move(), board).unwrap();

        assert!(matches!(bitmove.mve, MoveType::NonCapture(..)));
        if let MoveType::NonCapture(piece_indicies) = bitmove.mve {
            assert_eq!(piece_indicies[1], PieceMove{index: 12, value: PieceValueChange::Remove});
            assert_eq!(piece_indicies[0], PieceMove{index: 28, value: PieceValueChange::Place});
        }
     }
}