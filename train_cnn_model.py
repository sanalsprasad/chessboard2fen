## Train a convolutions neural network model to label pices 

import os 
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

from utils import fen_to_labels, labels, num_labels    

def create_model():
    """Function to create neural network with specified layers, loss and optimizer functions

    Returns:
        model (keras.Model.Sequential): neural network
    """
    ## Define model layers
    model = keras.models.Sequential()
    model.add(layers.Conv2D(50,(5,5),strides=(1,1), activation="relu", input_shape=(50,50,1)))
    model.add(layers.MaxPool2D((5,5)))
    model.add(layers.Conv2D(50,(5,5) ,strides=(1,1), activation="relu"))
    model.add(layers.Flatten())
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(len(labels)))

    ## Loss and Optimizer
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optim = keras.optimizers.Adam(learning_rate= 0.001)
    metrics = ["accuracy"]
    model.compile(optimizer=optim, loss=loss, metrics=metrics)
    return model


def load_train_data(train_board_count=500):
    """Loads the training data of squares for the neural network

    Args:
        train_board_count (int, optional): Number of chessboards to be used for training. Defaults to 500.

    Returns:
        x_train, y_train :tensorflow.tensor's of shape (64*train_board_count,50,50,3) and (64*train_board_count,) respectively. 
                                    x_train is normailzed to [0,1] and y_trian contains numerical labels associated to the pieces
    """
    ## Train data is located in /dataset/train/filename.jpeg where filename is FEN string
    train_path = "dataset/train/" 
    file_names = os.listdir(train_path)
    np.random.shuffle(file_names)
    file_names = file_names[:train_board_count]

    square_size = 50

    x_train = []
    y_train = []
    for file in file_names:
        ## Read file and convert to a tensor
        board = tf.io.read_file(train_path+file)
        board = tf.io.decode_jpeg(board, channels=1) # Read in gray-scale
        
        ## Labels of the board
        board_labels = fen_to_labels(file[:-5])

        for i in range(8):
            for j in range(8):
                ## Append the (i,j)th square of the board to the training set
                square = board[square_size*i: square_size*(i+1), square_size*j: square_size*(j+1),:]
                x_train.append(square)
                y_train.append(num_labels[board_labels[i][j]])

    x_train = tf.convert_to_tensor(x_train,dtype=float)
    x_train = x_train / 255.0
    y_train = tf.convert_to_tensor(y_train)
    print("Loaded Training data with ", x_train.shape[0], " many squares.")
    return x_train, y_train


def train_model(x_train, y_train):
    """Function to train neural network using training data x_train, y_train

    Args:
        x_train (tensorflow.tensor): train data of shape (N,50,50,1) containing (normalized) pixel data of a piece
        y_train (tensorflow.tensor): _description_

    Returns:
        model (keras.Model.Sequential): trained neural network model
    """
    print("Training model.")
    model = create_model()
    batch_size = 64
    epochs = 5
    model.fit(x_train, y_train, epochs=epochs)
    print("Done training model.")
    model.save_weights('model_weights')     # Save the weights of the model 
    print("Model weights saved to file model_weights")
    return model


def validate_model(model, test_board_count=100):
    """Validates the trained neural network

    Args:
        model (keras.model.Sequential): trained neural network model
        test_board_count (int, optional): Number of chessboards to be used for training. Defaults to 100.
    """     
    ## Load Test dataset
    test_board_count = 100 # Maximum number of boards to be used from training directory 
    test_path = "dataset/test/" 
    file_names = os.listdir(test_path)
    np.random.shuffle(file_names)
    file_names = file_names[:test_board_count]

    print("Loading Testing Data")
    square_size = 50

    x_test = []
    y_test = []
    for file in file_names:
        ## Read file and convert to a tensor
        board = tf.io.read_file(test_path+file)
        board = tf.io.decode_jpeg(board, channels=1) # Read in gray scale
        
        ## Labels of the board
        board_labels = fen_to_labels(file[:-5])

        for i in range(8):
            for j in range(8):
                ## Append the (i,j)th square of the board to the training set
                square = board[square_size*i: square_size*(i+1), square_size*j: square_size*(j+1),:]
                x_test.append(square)
                y_test.append(num_labels[board_labels[i][j]])

    x_test = tf.convert_to_tensor(x_test,dtype=float)
    x_test = x_test / 255.0
    y_test = tf.convert_to_tensor(y_test)

    print("Loaded Testing data with ", x_test.shape[0], " many squares.")

    ## Evaluate model 
    print("Test Evaluation:")
    model.evaluate(x_test,y_test)
    

def load_model(weight_file="model_weights"):
    """Loads the weights of the neural network from the weights file

    Args:
        weight_file (str, optional): File location for the weight file. Defaults to "model_weights".

    Returns:
        model (keras.Model.Sequential): Neural network with weights loaded from weight file
    """
    model = create_model()
    model.load_weights(weight_file)
    return model 


if __name__ == "__main__":
    x_train, y_train = load_train_data()
    model = create_model()
    model.summary()
    model = train_model(x_train, y_train)
    validate_model(model)

   
    
"""Code to Visualize the images
im = tf.io.read_file(train_boards[0])
im = tf.io.decode_jpeg(im, channels=3)
im_label = fen_to_labels(fen_labels[0])
print(im.shape)
fig , ax = plt.subplots(8,8)
for i in range(8):
    for j in range(8):
        plt.subplot(8, 8, 8*i + j + 1)
        plt.imshow(im[50*i: 50*(i+1), 50*j: 50*(j+1),:])
        plt.title(im_label[i][j])
#plt.imshow(im[0:50,0:50,:])
fig.tight_layout()
plt.subplots_adjust(top=2, right=1)
plt.show()
"""