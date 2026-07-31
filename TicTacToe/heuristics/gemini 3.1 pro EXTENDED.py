def myHeuristic(board):
    """
    Evaluates the best move for 'X' on a 3x3 board.
    Empty cells are assumed to be any value not equal to 'X' or 'Y' (e.g., ' ', '', None).
    Returns a tuple (row, col) representing the optimal move.
    """
    
    # Helper to check if the current board state has a winner
    def check_winner(b):
        # Check rows and columns
        for i in range(3):
            if b[i][0] == b[i][1] == b[i][2] and b[i][0] in ['X', 'Y']:
                return b[i][0]
            if b[0][i] == b[1][i] == b[2][i] and b[0][i] in ['X', 'Y']:
                return b[0][i]
                
        # Check diagonals
        if b[0][0] == b[1][1] == b[2][2] and b[0][0] in ['X', 'Y']:
            return b[0][0]
        if b[0][2] == b[1][1] == b[2][0] and b[0][2] in ['X', 'Y']:
            return b[0][2]
            
        return None

    # Helper to get all currently available moves
    def get_empty_cells(b):
        return [(r, c) for r in range(3) for c in range(3) if b[r][c] not in ['X', 'Y']]

    # The Minimax algorithm evaluates the board tree recursively
    def minimax(b, depth, is_maximizing):
        winner = check_winner(b)
        
        # Base cases: evaluate terminal states
        if winner == 'X':
            # Subtracting depth rewards faster wins
            return 10 - depth  
        elif winner == 'Y':
            # Adding depth rewards delaying a loss as long as possible
            return -10 + depth 
            
        empty_cells = get_empty_cells(b)
        if not empty_cells:
            # Draw
            return 0  
            
        if is_maximizing:
            max_eval = -float('inf')
            for r, c in empty_cells:
                orig = b[r][c]
                b[r][c] = 'X'
                eval = minimax(b, depth + 1, False)
                b[r][c] = orig # Backtrack
                max_eval = max(max_eval, eval)
            return max_eval
        else:
            min_eval = float('inf')
            for r, c in empty_cells:
                orig = b[r][c]
                b[r][c] = 'Y'
                eval = minimax(b, depth + 1, True)
                b[r][c] = orig # Backtrack
                min_eval = min(min_eval, eval)
            return min_eval

    # Main logic: test every possible initial move and pick the one with the highest Minimax score
    best_score = -float('inf')
    best_move = None
    
    for r, c in get_empty_cells(board):
        orig = board[r][c]
        board[r][c] = 'X'
        score = minimax(board, 0, False)
        board[r][c] = orig # Backtrack so we don't mutate the original array
        
        if score > best_score:
            best_score = score
            best_move = (r, c)
            
    return best_move