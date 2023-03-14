## Labels and the corresponding chess pieces
label_keys = {
    "": "Empty square",
    "P": "White Pawn",
    "N": "White Knight",
    "B": "White Bishop",
    "R": "White Rook",
    "Q": "White Queen",
    "K": "White King",
    "p": "Black Pawn",
    "n": "Black Knight",
    "b": "Black Bishop",
    "r": "Black Rook",
    "q": "Black Queen",
    "k": "Black King",
}

## Character labels
labels = list(label_keys.keys())

## Numerical labels
num_labels = {labels[idx]: idx for idx in range(len(labels))}

def fen_to_labels(fen: str) -> list[list[str]]:
    """Returns an 8 x 8 label matrix from a fen string

    Args:
        fen (str):  valid FEN string of a chess board
    Returns: 
        label_matrix : label matrix of the given FEN string
    """
    fen_rows = fen.split("-")
    label_matrix = []
    for row in fen_rows:
        ## Convert each row to an array of labels of size 8
        label_row = []
        for c in row: 
            if c in "12345678":
                label_row = label_row + [""]*int(c)
            else:
                label_row.append(c)
            row = row[1:]
        label_matrix.append(label_row)
    return label_matrix

def labels_to_fen(label_matrix: list[list[int]]) -> str:
    """Returns FEN string from label matrix

    Args:
        label_matrix (list[list[int]]): 8 x 8 matrix with piece labels

    Returns:
        fen (str) : FEN String
    """
    fen = ""
    for row in label_matrix:
        for c in row:
            if c == "":
                if len(fen) > 0 and fen[-1] in "1234567":
                    fen = fen[:-1] + str(int(fen[-1])+1)
                else:
                    fen = fen + "1"
            else:
                fen = fen + c
        fen = fen + "/"
    return fen[:-1] # Ignore the last "/"