"""Module to detect chessboard from a given image."""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress Tensorflow output

import numpy as np
import cv2
import tensorflow as tf
import sys
from chessboard_to_fen import ChessboardToFEN


class LineDetection:
    """Class to detect horizontal and vertical lines in an image."""
    def __init__(self, image : np.ndarray | None = None):
        """Constructor for LineDetection"""
        self.image = image
    
    def get_lines(self, max_len=30):
        """Method to get horizontal and vertical lines in the image. Return no 
        more than max_len horizontal and vertical lines each."""
        if self.image is None:
            raise(ValueError("LineDetection.image is None."))
        
        # Detect edges using Canny method
        image_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(image_gray, 50, 150)

        # We now detect edges using a type of 1-dimensional Hough transformation.
        # Note that we are exploiting that the edges of the chessboard are horizontal
        # and vertical here. This method will not work if the chessboard is at a 
        # different angle. This solution is inspired from 
        # https://github.com/Elucidation/tensorflow_chessbot
        
        # Find horizontal lines
        hor_edge_sum = edges.sum(axis=1)
        hor_edge_cond = (hor_edge_sum >= max(hor_edge_sum)/2)
        # Include the the two ends of the image in case it is not detected.
        hor_edge_cond[0] = True
        hor_edge_cond[-1] = True
        if np.sum(hor_edge_cond) <= max_len: 
            lines_hor = np.arange(self.image.shape[0])[hor_edge_cond]
        else:
            lines_hor = np.argsort(hor_edge_sum)[-max_len:]
            if 0 not in lines_hor:
                lines_hor = np.append(lines_hor, 0)
            if self.image.shape[0] - 1 not in lines_hor:
                lines_hor = np.append(lines_hor, self.image.shape[0] - 1)
            lines_hor.sort()

        # Find vertical lines
        ver_edge_sum = edges.sum(axis=0)
        ver_edge_cond = (ver_edge_sum >= max(ver_edge_sum)/2)
        # Include the the two ends of the image in case it is not detected.
        ver_edge_cond[0] = True
        ver_edge_cond[-1] = True
        if np.sum(ver_edge_cond) <= max_len: 
            lines_ver = np.arange(self.image.shape[1])[ver_edge_cond]
        else:
            lines_ver = np.argsort(ver_edge_sum)[-max_len:]
            if 0 not in lines_ver:
                lines_ver = np.append(lines_ver, 0)
            if self.image.shape[1] - 1 not in lines_ver:
                lines_ver = np.append(lines_ver, self.image.shape[1] - 1)
            lines_ver.sort()

        return lines_hor, lines_ver


class ChessboardFilter:
    """Class to implement a convolution chessboard filter object."""
    def __init__(self, num_cells: int = 5, fill : str = "edges"):
        """Constructor for ChessboardFilter. If fill is "edges", only edges of 
        each square are colored. If fill is "all", all of the square is colored."""
        # The filter will be of the form of 8x8 squares of alternating colors, 
        # where 1 corresponds to white and (-1) to black. Each square will be fit
        # into a num_cells x num_cells block and we only apply colors to the edges
        # of each square. This is to prevent the pieces from contributing to the
        # convolution.
        self.filter = np.zeros((num_cells * 8, num_cells * 8))
        for x_sq in range(8):
            for y_sq in range(8):
                square_color = (-1)**((x_sq + y_sq ) % 2)
                if fill == "all":
                    self.filter[x_beg: x_end, y_beg: y_end] = square_color
                    continue
                # else fill the edges
                x_beg = x_sq * num_cells
                x_end = x_beg + num_cells
                y_beg = y_sq * num_cells
                y_end = y_beg + num_cells
                self.filter[x_beg, y_beg:y_end] = square_color
                self.filter[x_end - 1, y_beg:y_end] = square_color
                self.filter[x_beg:x_end, y_beg] = square_color
                self.filter[x_beg:x_end, y_end - 1] = square_color
    
    def convolve(self, image: np.ndarray) -> float: 
        """Method to convolve an image with the chessboard filter."""
        # Normalize and resize the image
        image = image.copy() - np.mean(image)
        image = cv2.resize(image, self.filter.shape)
        return np.sum(image * self.filter)


class ChessboardDetect:
    """Class for detecting a chessboard"""
    def __init__(self,
                 image: np.ndarray | None = None,
                 max_res=(400,400),
                 filter=ChessboardFilter(),
                 line_detection=LineDetection(),
                 min_board_width_factor=0.5
                 ):
        """Constructor for ChessboardDetect."""
        self.image = image
        self.image_resized = cv2.resize(image, max_res)
        self.image_gray = cv2.cvtColor(self.image_resized, cv2.COLOR_BGR2GRAY)
        self.filter = filter
        self.line_detection = line_detection
        self.line_detection.image = self.image_resized
        self.min_board_width_factor = min_board_width_factor

    def get_optimal_edges(self):
        """Function to find the optimal edges of the chessboard using the data of horizontal 
        and vertical lines in an image. The parameter min_width_factor is the minimum distance 
        allowed between two horizontal and vertical lines."""
        min_board_width = self.min_board_width_factor * min(self.image_resized.shape)
        # Board needs to be at least 40 x 40 pixels for a good detection
        min_board_width = max(min_board_width, 40)  
        
        # Get horizontal and vertical lines in the image
        lines_hor, lines_ver = self.line_detection.get_lines()
        
        # For each possible pair of horizontal and pair of vertical lines, 
        # compute the convolution and keep track of the maximal one.
        max_conv = 0
        optimal_edges = [[None, None], [None, None]]
        for idx1_hor in range(len(lines_hor)):
            for idx2_hor in range(idx1_hor, len(lines_hor)):
                for idx1_ver in range(len(lines_ver)):
                    for idx2_ver in range(idx1_ver, len(lines_ver)):
                        hline1 = lines_hor[idx1_hor]
                        hline2 = lines_hor[idx2_hor]
                        vline1 = lines_ver[idx1_ver]
                        vline2 = lines_ver[idx2_ver]
                        if (
                            (np.abs(hline1 - hline2) < min_board_width)
                            or (np.abs(vline1 - vline2) < min_board_width)
                            ):
                            # Lines are too close
                            continue
                        edges = [[hline1, hline2], [vline1, vline2]]
                        cropped = self.image_gray[hline1:hline2, vline1:vline2]
                        conv = np.abs(self.filter.convolve(cropped))
                        if conv > max_conv:
                            max_conv = conv
                            optimal_edges = edges
        # Undo the resize factor 
        hor_resize_factor = self.image.shape[0]/self.image_resized.shape[0]
        ver_resize_factor = self.image.shape[1]/self.image_resized.shape[1]
        resize_factor = np.array([[hor_resize_factor]*2, [ver_resize_factor]*2])
        return (optimal_edges * resize_factor).astype(int)
    
    def get_unoriented_chessboard(self):
        """Method to return the unoriented chessboard detected from the image."""
        optimal_edges = self.get_optimal_edges()
        hline1, hline2 = optimal_edges[0]
        vline1, vline2 = optimal_edges[1]
        return self.image[hline1:hline2, vline1:vline2]

    def get_chessboard(self): 
        """Method to return the chessboard detected from the image such that the
        a1 square is in the bottom left corner i.e. player is facing white."""
        unoriented_chessboard = self.get_unoriented_chessboard()
        gray = cv2.cvtColor(unoriented_chessboard, cv2.COLOR_BGR2GRAY)
        if self.filter.convolve(gray) > 0:
            return unoriented_chessboard
        return cv2.rotate(unoriented_chessboard, cv2.ROTATE_180)
    
    
if __name__ == "__main__":
    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    chessboard_detect = ChessboardDetect(image)
    chessboard = chessboard_detect.get_chessboard()
    cv2.imwrite('detected_chessboard.jpg', chessboard)
    cv2.imshow('Detected Chessboard', chessboard)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    chessboard_to_fen = ChessboardToFEN()
    # Convert to gray scale and reshape to (h,w,1) for prediction
    gray = cv2.cvtColor(chessboard, cv2.COLOR_BGR2GRAY)
    reshaped = gray[:,:,np.newaxis]
    fen, _ = chessboard_to_fen.predict_confidence(tf.constant(reshaped))
    print(f"The predicted FEN string is {fen}.")
    cv2.imshow(f"Predicted FEN String: {fen}", chessboard)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    