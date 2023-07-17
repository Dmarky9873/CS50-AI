"""
Tic Tac Toe Player
"""

import math
import copy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    x_count = 0
    o_count = 0
    for row in board:
        for col in row:
            if col == X:
                x_count += 1
            elif col == O:
                o_count += 1
    if x_count == o_count:
        return "X"
    elif x_count > o_count:
        return "O"
    else:
        return None


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """

    moves = set()

    for row in range(3):
        for col in range(3):
            if board[row][col] == None:
                moves.add((row, col))

    return moves


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    # Checking if action is valid
    if (action[0] > 2) or (action[0] < 0) or (action[1] > 2) or (action[1] < 0):
        raise Exception("Invalid action in result function")

    copiedBoard = copy.deepcopy(board)
    row, col = action
    copiedBoard[row][col] = player(board)
    return copiedBoard


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """

    x_horizontal_victory = [X, X, X]
    o_horizontal_victory = [O, O, O]
    # Checking for horizontal victories
    for row in board:
        if row == x_horizontal_victory:
            return X
        elif row == o_horizontal_victory:
            return O

    # Checking for vertical victories
    for i in range(3):
        if board[0][i] == X and board[1][i] == X and board[2][i] == X:
            return X
        elif board[0][i] == O and board[1][i] == O and board[2][i] == O:
            return O

    # Checking for diagonal victories
    if board[0][0] == X and board[1][1] == X and board[2][2] == X:
        return X
    elif board[0][0] == O and board[1][1] == O and board[2][2] == O:
        return O
    elif board[2][0] == X and board[1][1] == X and board[0][2] == X:
        return X
    elif board[2][0] == O and board[1][1] == O and board[0][2] == O:
        return O

    # Continue game
    return 0


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    # Checks if someone won the game
    if winner(board) != 0:
        return True

    # Checks if all blocks are full
    for row in board:
        if EMPTY in row:
            return False

    # Returns true because if no winner but no empty spaces means that everything is full
    return True


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if winner(board) == X:
        return 1
    elif winner(board) == O:
        return -1
    else:
        return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    # Gets the set of all possible actions
    possible_actions = actions(board)
    best_move = None
    best_val = None

    # If we are trying to maximize the value we take all the moves and go
    # through them using the value function and find the one that will
    # return the highest value
    if player(board) == X:
        # Sets best_val to negative infinity so that any move is better
        # than no move
        best_val = float('-inf')
        # Loops through every action
        for a in possible_actions:
            # If the value of that action is better than the current best
            # action then change current best action to current action we
            # are looking at
            if value(result(board, a)) > best_val:
                best_val = value(result(board, a))
                best_move = a

        return best_move
    else:
        # Does the same for minimizing the value
        best_val = float('inf')
        for a in possible_actions:
            if value(result(board, a)) < best_val:
                best_val = value(result(board, a))
                best_move = a

        return best_move


def value(board):
    """
    Takes a board state and returns the value of that board
    """

    # Checks if board is in a terminal state, and if so returns the value
    if terminal(board):
        return utility(board)

    # If we are trying to maximize the value, we take v, set it to be -infinity
    # and recusively checks the eventual value of each action
    elif player(board) == X:
        # Setting the value to negative infinity so that any move is better than none
        v = float('-inf')
        # Loops through all actions
        for a in actions(board):
            # Checks to see whether the path of that action results in a value higher than the current best value
            v = max(v, value(result(board, a)))
            # Breaks out of the loop because it found the best solution and no point in keep trying to find others
            if v == 1:
                break
        return v

    # Does the same but for minimizing the value
    else:
        v = float('inf')
        for a in actions(board):
            v = min(v, value(result(board, a)))
            if v == -1:
                break
        return v
