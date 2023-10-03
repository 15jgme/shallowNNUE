use chess::{self, Board, ChessMove, Color, ALL_SQUARES};
use tch::{
    nn::{Module, VarStore},
    vision::{imagenet, resnet::resnet18},
    CModule, Device, IndexOp, Kind, Tensor,
};

use crate::bit_move::{BitMove, MoveType, PieceValueChange, piece_index, get_index, orient};

pub trait NNUE {
    fn forward(&mut self, chess_move: ChessMove) -> Result<i16, ()>; // Runs the model given the supplied move, and unmakes the move afterwards
    fn set_board_hard(&mut self, board: Board) -> Result<(), ()>; // Slow reset of the board (cleans and adds pieces)
}

#[derive(Debug)]
pub struct ShallowNNUE {
    board: Board,
    encoding_tensor: Tensor, // Represents self
    // encoding_tensor_black: Tensor,
    model: CModule,
}

impl ShallowNNUE {
    fn make_move(&self, bitmove: BitMove) {
        match bitmove.mve {
            MoveType::NonCapture(indicies) => {
                for index in indicies {
                    let change_value = match index.value {
                        PieceValueChange::Place => 1.0,
                        PieceValueChange::Remove => 0.0,
                    };
                    let _ = self
                        .encoding_tensor
                        .i(index.index as i64)
                        .fill_(change_value);
                }
            }
            MoveType::Promote(indicies) => {
                for index in indicies {
                    let change_value = match index.value {
                        PieceValueChange::Place => 1.0,
                        PieceValueChange::Remove => 0.0,
                    };
                    let _ = self
                        .encoding_tensor
                        .i(index.index as i64)
                        .fill_(change_value);
                }
            }
            MoveType::Capture(indicies) => {
                for index in indicies {
                    let change_value = match index.value {
                        PieceValueChange::Place => 1.0,
                        PieceValueChange::Remove => 0.0,
                    };
                    let _ = self
                        .encoding_tensor
                        .i(index.index as i64)
                        .fill_(change_value);
                }
            }
            MoveType::Castle(indicies) => {
                for index in indicies {
                    let change_value = match index.value {
                        PieceValueChange::Place => 1.0,
                        PieceValueChange::Remove => 0.0,
                    };
                    let _ = self
                        .encoding_tensor
                        .i(index.index as i64)
                        .fill_(change_value);
                }
            }
        };
    }

    fn unmake_move(&self, bitmove: BitMove) {
        match bitmove.mve {
            MoveType::NonCapture(indicies) => {
                for index in indicies {
                    let change_value = match index.value {
                        PieceValueChange::Place => 0.0,
                        PieceValueChange::Remove => 1.0,
                    };
                    let _ = self
                        .encoding_tensor
                        .i(index.index as i64)
                        .fill_(change_value);
                }
            }
            MoveType::Promote(indicies) => {
                for index in indicies {
                    let change_value = match index.value {
                        PieceValueChange::Place => 0.0,
                        PieceValueChange::Remove => 1.0,
                    };
                    let _ = self
                        .encoding_tensor
                        .i(index.index as i64)
                        .fill_(change_value);
                }
            }
            MoveType::Capture(indicies) => {
                for index in indicies {
                    let change_value = match index.value {
                        PieceValueChange::Place => 0.0,
                        PieceValueChange::Remove => 1.0,
                    };
                    let _ = self
                        .encoding_tensor
                        .i(index.index as i64)
                        .fill_(change_value);
                }
            }
            MoveType::Castle(indicies) => {
                for index in indicies {
                    let change_value = match index.value {
                        PieceValueChange::Place => 0.0,
                        PieceValueChange::Remove => 1.0,
                    };
                    let _ = self
                        .encoding_tensor
                        .i(index.index as i64)
                        .fill_(change_value);
                }
            }
        };
    }

    pub fn new(global_path_to_model: String) -> Result<ShallowNNUE, ()> {
        let mut model = match tch::CModule::load(global_path_to_model) {
            Ok(model) => model,
            Err(_) => return Err(()),
        };

        model.to(Device::cuda_if_available(), Kind::Float, false); // Send the model to the CPU or GPU if available
        model.set_eval();

        let encoding_tensor = tch::Tensor::zeros(768, (Kind::Float, Device::cuda_if_available()));
        let board = Board::default();

        Ok(ShallowNNUE {
            board,
            encoding_tensor,
            model,
        })
    }
}

impl NNUE for ShallowNNUE {
    fn forward(&mut self, chess_move: ChessMove) -> Result<i16, ()> {
        let turn = self.board.side_to_move();
        let bitmove = BitMove::new(chess_move, turn, self.board)?;

        // Apply the move to the tensors
        self.make_move(bitmove);

        let result = self
            .model
            .forward(&self.encoding_tensor)
            .f_int64_value(&[0])
            .expect("Model forward should not fail") as i16;

        // Reset the tensors unmaking the move
        self.unmake_move(bitmove);

        Ok(result)
    }

    fn set_board_hard(&mut self, board: Board) -> Result<(), ()> {
        self.board = board;

        // Clear encodings
        let _ = self.encoding_tensor.i(..).fill_(0.0);

        // Encode all pieces
        for sq in ALL_SQUARES{
            // Iterate through all squares, if there is a piece, add it to the tensor
            match self.board.piece_on(sq){
                Some(piece) => {
                    let colour = self.board.side_to_move();
                    let own_piece: bool = self.board.color_on(sq).expect("Square with piece should not be empty") == self.board.side_to_move();
                    let index = get_index(piece, own_piece, orient(sq, colour));
                    let _ = self.encoding_tensor.i(index as i64).fill_(1.0);
                },
                None => {/* Skip */},
            }

        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use chess::Square;

    use super::*;
    #[test]
    fn test() {
        let test_tensor = tch::Tensor::zeros(768, (Kind::Float, Device::cuda_if_available()));
        println!("{}", test_tensor);
        test_tensor.i(2).fill_(1.0);
        println!("{}", test_tensor)
    }

    #[test]
    fn test_nnue_struct() {
        let nnue = ShallowNNUE::new(
            "/home/jgme/Documents/software-projects/shallowNNUE/shallow-learn-tscript.pt"
                .to_string(),
        )
        .unwrap();
    }

    #[test]
    fn test_default_board() {
        let mut nnue  = ShallowNNUE::new(
            "/home/jgme/Documents/software-projects/shallowNNUE/shallow-learn-tscript.pt"
                .to_string(),
        )
        .unwrap();
        
        let board = Board::default();
        nnue.set_board_hard(board).unwrap();
        let mve: ChessMove = ChessMove::new(Square::E2, Square::E4, None);
        println!("{}", nnue.forward(mve).unwrap());

        let mve: ChessMove = ChessMove::new(Square::G1, Square::F3, None);
        println!("{}", nnue.forward(mve).unwrap());

        assert!(matches!(nnue.forward(mve), Ok(_))); // Ensure forward doesnt result in error
        assert!(nnue.forward(mve) == nnue.forward(mve)); // Ensure a repeated test yeilds the same result
        assert!(nnue.encoding_tensor.i(28) == Tensor::from(0.0)); // Check that E4 is once again unoccupied (unmake move works)
    }
}
