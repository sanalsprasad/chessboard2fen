import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress Tensorflow output

import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from utils import labels, label_keys
from train_cnn_model import load_model

class PieceClassifier:
    """Object to classify a chees piece using a CNN."""
    def __init__(self,weight_file: str ="model_weights"):
        """Initialized PieceClassifier object using CNN model weights.

        Args:
            weight_file (str, optional): File with model weights. Defaults to
                    "model_weights".
        """ 
        self.model = load_model(weight_file=weight_file)
    
    def predict(self, square: tf.Tensor):
        """Predicts the piece type based piece image tensor.

        Args:
            square (tf.tensor): shape (height,width,1) 

        Returns:
            str: alphabetic label of shape
        """
        square = tf.image.resize(square, size=[50,50])
        square = square[tf.newaxis,:,:,:]
        pred = np.argmax(self.model.predict(square,verbose=0))
        return labels[pred]
    
    def predict_proba(self, square):
        square = tf.image.resize(square,size=[50,50])
        square = square[tf.newaxis,:,:,:]
        return tf.nn.softmax(self.model.predict(square,verbose=0))

if __name__ == "__main__":
    # Read square
    square_path = sys.argv[1]
    square = tf.io.read_file(square_path)
    square = tf.io.decode_jpeg(square, channels=1)
    
    # Print predictions
    piece_clf = PieceClassifier()
    preds = piece_clf.predict_proba(square)
    print("The following are the predictions probabilities:")
    for piece in range(preds.shape[1]):
        print(label_keys[labels[piece]] + " : " + str(np.round(preds[0,piece]*100,3)) + "%")
    piece = np.argmax(preds)
    print("Based on the above probabilites, the piece is predicted to be a " +label_keys[labels[piece]] + ".")