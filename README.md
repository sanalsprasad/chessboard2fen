# chessboard2fen
The goal of this project is to extract the [Forsyth-Edwards Notation](https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation) from a chessboard image. This is accomplished via training a convolutional neural network to recognize the chess pieces.

## List of files
The project contains the following files: 
* ulits.py - contains some commonly used functions (such as to convert a matrix of piece labels to FEN notation and vice-versa).
* train_cnn_model.py -  creates and trains the convolutional neural network used to classify pieces.
* model_weights.index - stores the weights of the trained convolutional neural network model. 

## Acknowledegments
The neural network is trained using a subset of [this kaggle dataset](https://www.kaggle.com/datasets/koryakinp/chess-positions) created Pavel Koryakin. 
This project is inspired by [chessvision-ai](https://chessvision.ai/) and their reddit bot [/u/chessvision-ai-bot](https://reddit.com/u/chessvision-ai-bot).
