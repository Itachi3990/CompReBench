def check_winner(board):
    """Checks if there is a winner or a draw on the 4x4 board."""
    # Check rows
    for r in range(4):
        if board[r][0] != '' and board[r][0] == board[r][1] == board[r][2] == board[r][3]:
            return board[r][0]
    
    # Check columns
    for c in range(4):
        if board[0][c] != '' and board[0][c] == board[1][c] == board[2][c] == board[3][c]:
            return board[0][c]
    
    # Check diagonals
    if board[0][0] != '' and board[0][0] == board[1][1] == board[2][2] == board[3][3]:
        return board[0][0]
    if board[0][3] != '' and board[0][3] == board[1][2] == board[2][1] == board[3][0]:
        return board[0][3]
    
    # Check for draw (board full)
    if all(board[r][c] != '' for r in range(4) for c in range(4)):
        return 'Draw'
        
    return None

def evaluate_board(board):
    """Helper function to evaluate non-terminal board states."""
    winner = check_winner(board)
    if winner == 'X':
        return 100000
    elif winner == 'Y':
        return -100000
    elif winner == 'Draw':
        return 0
    
    score = 0
    lines = []
    
    # Collect all rows, columns, and diagonals
    for r in range(4):
        lines.append([board[r][c] for c in range(4)])
    for c in range(4):
        lines.append([board[r][c] for r in range(4)])
    lines.append([board[i][i] for i in range(4)])
    lines.append([board[i][3-i] for i in range(4)])
    
    # Score lines based on potential for 4-in-a-row
    for line in lines:
        x_count = line.count('X')
        y_count = line.count('Y')
        
        if x_count > 0 and y_count == 0:
            score += 10 ** x_count
        elif y_count > 0 and x_count == 0:
            score -= 10 ** y_count
            
    return score

def minimax(board, depth, alpha, beta, is_maximizing):
    """Minimax algorithm with Alpha-Beta pruning."""
    winner = check_winner(board)
    if winner == 'X':
        return 100000 + depth
    if winner == 'Y':
        return -100000 - depth
    if winner == 'Draw':
        return 0
    if depth == 0:
        return evaluate_board(board)
        
    if is_maximizing:
        max_eval = -float('inf')
        for r in range(4):
            for c in range(4):
                if board[r][c] == '':
                    board[r][c] = 'X'
                    eval = minimax(board, depth - 1, alpha, beta, False)
                    board[r][c] = ''
                    max_eval = max(max_eval, eval)
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for r in range(4):
            for c in range(4):
                if board[r][c] == '':
                    board[r][c] = 'Y'
                    eval = minimax(board, depth - 1, alpha, beta, True)
                    board[r][c] = ''
                    min_eval = min(min_eval, eval)
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break
            if beta <= alpha:
                break
        return min_eval

def myHeuristic(board):
    """Main heuristic function called by the external script."""
    empty_cells = sum(1 for r in range(4) for c in range(4) if board[r][c] == '')
    if empty_cells == 0:
        return None
        
    # Determine lookahead depth based on empty cells constraint
    depth_limit = 4 if empty_cells > 8 else empty_cells
    
    best_eval = -float('inf')
    best_move = None
    alpha = -float('inf')
    beta = float('inf')
    
    # Evaluate all possible valid moves for 'X'
    for r in range(4):
        for c in range(4):
            if board[r][c] == '':
                board[r][c] = 'X'
                eval = minimax(board, depth_limit - 1, alpha, beta, False)
                board[r][c] = ''
                
                if eval > best_eval:
                    best_eval = eval
                    best_move = (r, c)
                    
    return best_move