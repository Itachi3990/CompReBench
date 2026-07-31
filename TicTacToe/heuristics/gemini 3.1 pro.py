def myHeuristic(board):
    """
    Returns the best possible move (row, col) for 'X' on a 3x3 board.
    Assumes any cell that does not contain 'X' or 'Y' is empty.
    """
    
    # Helper to evaluate the board for a win, loss, or draw
    def evaluate(b):
        # Check rows for 'X' or 'Y' victory
        for row in range(3):
            if b[row][0] == b[row][1] == b[row][2]:
                if b[row][0] == 'X':
                    return 10
                elif b[row][0] == 'Y':
                    return -10

        # Check columns for 'X' or 'Y' victory
        for col in range(3):
            if b[0][col] == b[1][col] == b[2][col]:
                if b[0][col] == 'X':
                    return 10
                elif b[0][col] == 'Y':
                    return -10

        # Check diagonals for 'X' or 'Y' victory
        if b[0][0] == b[1][1] == b[2][2]:
            if b[0][0] == 'X':
                return 10
            elif b[0][0] == 'Y':
                return -10

        if b[0][2] == b[1][1] == b[2][0]:
            if b[0][2] == 'X':
                return 10
            elif b[0][2] == 'Y':
                return -10

        # No winner yet
        return 0

    # Helper to check if any moves are left
    def is_moves_left(b):
        for i in range(3):
            for j in range(3):
                if b[i][j] not in ('X', 'Y'):
                    return True
        return False

    # Minimax function with alpha-beta pruning
    def minimax(b, depth, is_max, alpha, beta):
        score = evaluate(b)

        # If Maximizer (X) has won
        if score == 10:
            return score - depth # Prefer faster wins
            
        # If Minimizer (Y) has won
        if score == -10:
            return score + depth # Prefer slower losses
            
        # If there are no more moves, it's a tie
        if not is_moves_left(b):
            return 0

        # If it's the maximizer's turn (X)
        if is_max:
            best = -1000
            for i in range(3):
                for j in range(3):
                    if b[i][j] not in ('X', 'Y'):
                        original = b[i][j]
                        b[i][j] = 'X'
                        best = max(best, minimax(b, depth + 1, not is_max, alpha, beta))
                        b[i][j] = original # Undo move
                        
                        alpha = max(alpha, best)
                        if beta <= alpha:
                            return best
            return best
            
        # If it's the minimizer's turn (Y)
        else:
            best = 1000
            for i in range(3):
                for j in range(3):
                    if b[i][j] not in ('X', 'Y'):
                        original = b[i][j]
                        b[i][j] = 'Y'
                        best = min(best, minimax(b, depth + 1, not is_max, alpha, beta))
                        b[i][j] = original # Undo move
                        
                        beta = min(beta, best)
                        if beta <= alpha:
                            return best
            return best

    # Evaluate all possible moves to find the best one
    best_val = -1000
    best_move = (-1, -1)

    for i in range(3):
        for j in range(3):
            # Check if cell is empty
            if board[i][j] not in ('X', 'Y'):
                original = board[i][j]
                board[i][j] = 'X'
                
                # Compute evaluation function for this move
                move_val = minimax(board, 0, False, -1000, 1000)
                
                # Undo the move
                board[i][j] = original

                # If the value of the current move is better than the best value, update
                if move_val > best_val:
                    best_move = (i, j)
                    best_val = move_val

    return best_move