import copy

def myHeuristic(board):
    """
    Determines the best move for 'X' on a 3x3 Tic-Tac-Toe board 
    using the Minimax algorithm.
    
    Parameters:
    board (list of lists): A 3x3 2D array representing the current board state.
                           Cells can be 'X', 'Y', or empty (e.g., '', ' ', None).
                           
    Returns:
    tuple: (row, col) representing the optimal move.
    """
    
    def check_winner(b):
        # Check rows
        for row in b:
            if row[0] == row[1] == row[2] and row[0] in ('X', 'Y'):
                return row[0]
        # Check columns
        for col in range(3):
            if b[0][col] == b[1][col] == b[2][col] and b[0][col] in ('X', 'Y'):
                return b[0][col]
        # Check diagonals
        if b[0][0] == b[1][1] == b[2][2] and b[0][0] in ('X', 'Y'):
            return b[0][0]
        if b[0][2] == b[1][1] == b[2][0] and b[0][2] in ('X', 'Y'):
            return b[0][2]
        
        # Check for draw (board full)
        full = all(cell in ('X', 'Y') for row in b for cell in row)
        if full:
            return 'Draw'
            
        return None

    def minimax(b, depth, is_maximizing):
        winner = check_winner(b)
        
        # Terminal states evaluation
        if winner == 'X':
            return 10 - depth  # Prefer faster wins
        if winner == 'Y':
            return depth - 10  # Prefer slower losses (though avoided)
        if winner == 'Draw':
            return 0
            
        if is_maximizing:
            max_eval = -float('inf')
            for r in range(3):
                for c in range(3):
                    if b[r][c] not in ('X', 'Y'):
                        original = b[r][c]
                        b[r][c] = 'X'
                        evaluation = minimax(b, depth + 1, False)
                        b[r][c] = original
                        max_eval = max(max_eval, evaluation)
            return max_eval
        else:
            min_eval = float('inf')
            for r in range(3):
                for c in range(3):
                    if b[r][c] not in ('X', 'Y'):
                        original = b[r][c]
                        b[r][c] = 'Y'
                        evaluation = minimax(b, depth + 1, True)
                        b[r][c] = original
                        min_eval = min(min_eval, evaluation)
            return min_eval

    best_eval = -float('inf')
    best_move = None
    
    # Work on a deep copy to keep the original board unmodified
    b_copy = copy.deepcopy(board)
    
    for r in range(3):
        for c in range(3):
            if b_copy[r][c] not in ('X', 'Y'):
                original = b_copy[r][c]
                b_copy[r][c] = 'X'
                move_eval = minimax(b_copy, 0, False)
                b_copy[r][c] = original
                
                if move_eval > best_eval:
                    best_eval = move_eval
                    best_move = (r, c)
                    
    return best_move