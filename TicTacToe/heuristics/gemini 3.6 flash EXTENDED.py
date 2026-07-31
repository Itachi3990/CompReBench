def myHeuristic(board):
    """
    Evaluates a 3x3 board state and returns the optimal move (row, col) for player 'X'.
    Uses Minimax with Alpha-Beta Pruning to guarantee optimal play:
    Priority: Win > Draw > Lose.
    """
    PLAYER = 'X'
    OPPONENT = 'Y'

    def check_winner(b):
        # Check rows and columns
        for i in range(3):
            if b[i][0] == b[i][1] == b[i][2] and b[i][0] in (PLAYER, OPPONENT):
                return b[i][0]
            if b[0][i] == b[1][i] == b[2][i] and b[0][i] in (PLAYER, OPPONENT):
                return b[0][i]
        
        # Check diagonals
        if b[0][0] == b[1][1] == b[2][2] and b[0][0] in (PLAYER, OPPONENT):
            return b[0][0]
        if b[0][2] == b[1][1] == b[2][0] and b[0][2] in (PLAYER, OPPONENT):
            return b[0][2]
        
        return None

    def get_empty_cells(b):
        empty = []
        for r in range(3):
            for c in range(3):
                if b[r][c] not in (PLAYER, OPPONENT):
                    empty.append((r, c))
        return empty

    def minimax(b, depth, alpha, beta, is_maximizing):
        winner = check_winner(b)
        if winner == PLAYER:
            return 10 - depth  # Prefer faster wins
        if winner == OPPONENT:
            return depth - 10  # Prefer delayed losses
        
        empty_cells = get_empty_cells(b)
        if not empty_cells:
            return 0  # Draw

        if is_maximizing:
            max_eval = -float('inf')
            for r, c in empty_cells:
                orig_val = b[r][c]
                b[r][c] = PLAYER
                eval_val = minimax(b, depth + 1, alpha, beta, False)
                b[r][c] = orig_val
                
                max_eval = max(max_eval, eval_val)
                alpha = max(alpha, eval_val)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for r, c in empty_cells:
                orig_val = b[r][c]
                b[r][c] = OPPONENT
                eval_val = minimax(b, depth + 1, alpha, beta, True)
                b[r][c] = orig_val
                
                min_eval = min(min_eval, eval_val)
                beta = min(beta, eval_val)
                if beta <= alpha:
                    break
            return min_eval

    best_val = -float('inf')
    best_move = None
    available_moves = get_empty_cells(board)

    for r, c in available_moves:
        orig_val = board[r][c]
        board[r][c] = PLAYER
        # Next turn is opponent's ('Y'), so minimizing
        move_val = minimax(board, 0, -float('inf'), float('inf'), False)
        board[r][c] = orig_val

        if move_val > best_val:
            best_val = move_val
            best_move = (r, c)

    return best_move