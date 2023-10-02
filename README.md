# chessboard2fen
The goal of this project is to extract the [Forsyth-Edwards Notation](https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation) from a chessboard image. This is accomplished via training a convolutional neural network to recognize the chess pieces.

## Usage 
Ensure that tensorflow package is installed. Save the chessboard as jpeg file and run 
`python chessboard_to_fen.py file_name.jpg`. 

## List of files
The project contains the following files: 
* ulits.py - contains some commonly used functions (such as to convert a matrix of piece labels to FEN notation and vice-versa).
* train_cnn_model.py -  creates and trains the convolutional neural network used to classify pieces.
* model_weights.index - stores the weights of the trained convolutional neural network model.
* piece_classifier.py - to classify a single chess piece.
* chessboard_to_fen.py - to predict FEN of a chessboard.

## Acknowledgements
The neural network is trained using a subset of [this kaggle dataset](https://www.kaggle.com/datasets/koryakinp/chess-positions) created Pavel Koryakin.
This project is inspired by [chessvision-ai](https://chessvision.ai/) and their reddit bot [/u/chessvision-ai-bot](https://reddit.com/u/chessvision-ai-bot).
