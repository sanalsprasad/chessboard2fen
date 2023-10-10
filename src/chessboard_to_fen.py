"""File to identify the FEN string using the image tensor of a chessboard."""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress Tensorflow output
import argparse  # For command line arguments
import sys
import numpy as np
import tensorflow as tf
from piece_classifier import PieceClassifier
from utils import labels_to_fen, labels, flip_fen


class ChessboardToFEN:
    """Generate FEN string of a chessboard image tensor."""
    def __init__(self, piece_clf = PieceClassifier()):
        """Initialize ChessboardToFEN object

        Args:
            piece_clf (PieceClassifier, optional): object for classifying chess pieces. Defaults to PieceClassifier().
        """
        self.piece_clf = piece_clf

    def predict_confidence(self, board: tf.Tensor) -> tuple[str, np.ndarray]:
        """Predict the FEN string of a board and return with confidence.

        Args:
            board (tensorflow.tensor): pixel data of a (grayscale) chessboard of shape (400,400,1)

        Returns:
            fen_string (str): Predicted FEN string of the board
            conf (np.ndarray): Confidence of prediction of each shape, shape = (8,8)
        """
        board = tf.image.resize(board, size=[400,400])
        board = board/255.0
        board_matrix = []
        conf = np.zeros((8,8))
        for i in range(8):
            row = []
            for j in range(8):
                square = board[50*i:50*(i+1), 50*j:50*(j+1),:]
                pred_prob = self.piece_clf.predict_proba(square)
                piece_pred = np.argmax(pred_prob)
                conf[i,j] = pred_prob[0,piece_pred]
                row.append(labels[piece_pred])
            board_matrix.append(row)
        return labels_to_fen(board_matrix), conf
    

def parse_arguments():
    parser = argparse.ArgumentParser(description="Predict FEN string of a chessboard image.")
    parser.add_argument("board", help="Filename of the chessboard image.")
    parser.add_argument("--flip", "-f", action='store_true',
                        help="Flip the image. Use this if the chessboard is upside down.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    board_file, flip = args.board, args.flip
    board = tf.io.read_file(board_file)
    board = tf.io.decode_image(board, channels=1)
    
    # Print predictions
    board_to_fen = ChessboardToFEN()
    fen, conf = board_to_fen.predict_confidence(board)
    if flip:
        fen = flip_fen(fen)
    print("The predicted FEN string of the board is : ", fen)
    print("The confidence matrix of prediction is: ")
    print(conf)
