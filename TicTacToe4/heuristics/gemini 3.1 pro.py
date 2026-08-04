import math

def myHeuristic(board):
    """
    Evaluates the best move for 'X' on a 4x4 board.
    Takes a 2D list `board` and returns a tuple (row, col) representing the best move.
    """
    def check_winner(b):
        # Check rows for a win
        for r in range(4):
            if b[r][0] != '' and b[r][0] == b[r][1] == b[r][2] == b[r][3]:
                return b[r][0]
                
        # Check columns for a win
        for c in range(4):
            if b[0][c] != '' and b[0][c] == b[1][c] == b[2][c] == b[3][c]:
                return b[0][c]
                
        # Check diagonals for a win
        if b[0][0] != '' and b[0][0] == b[1][1] == b[2][2] == b[3][3]:
            return b[0][0]
        if b[0][3] != '' and b[0][3] == b[1][2] == b[2][1] == b[3][0]:
            return b[0][3]
            
        return None

    def evaluate(b, current_depth):
        """
        Static evaluation helper function to grade non-terminal board states 
        or terminal states if reached before depth limit.
        """
        winner = check_winner(b)
        
        # Base cases: game over
        # We factor in depth to prefer faster wins and delayed losses
        if winner == 'X':
            return 10000 + current_depth 
        elif winner == 'Y':
            return -10000 - current_depth
            
        score = 0
        lines = []
        
        # Collect all possible winning lines
        for r in range(4):
            lines.append([b[r][c] for c in range(4)])
        for c in range(4):
            lines.append([b[r][c] for r in range(4)])
        lines.append([b[i][i] for i in range(4)])
        lines.append([b[i][3-i] for i in range(4)])

        # Heuristic scoring based on how close either player is to a win
        for line in lines:
            count_x = line.count('X')
            count_y = line.count('Y')
            
            # Line only contains 'X' and empty spaces (potential win for X)
            if count_x > 0 and count_y == 0:
                if count_x == 3: score += 100
                elif count_x == 2: score += 10
                elif count_x == 1: score += 1
            # Line only contains 'Y' and empty spaces (potential win for Y)
            elif count_y > 0 and count_x == 0:
                if count_y == 3: score -= 100
                elif count_y == 2: score -= 10
                elif count_y == 1: score -= 1
                
        return score

    def get_empty_cells(b):
        empty = []
        for r in range(4):
            for c in range(4):
                if b[r][c] == '':
                    empty.append((r, c))
        return empty

    def minimax(b, depth, alpha, beta, is_maximizing):
        """
        Minimax algorithm with Alpha-Beta pruning to explore the game tree.
        """
        winner = check_winner(b)
        empty_cells = get_empty_cells(b)

        # Stop conditions: Terminal state (win/lose/draw) or depth limit reached
        if winner or len(empty_cells) == 0 or depth == 0:
            return evaluate(b, depth)

        if is_maximizing:
            max_eval = -math.inf
            for r, c in empty_cells:
                b[r][c] = 'X'
                eval_score = minimax(b, depth - 1, alpha, beta, False)
                b[r][c] = ''  # Backtrack
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Beta cut-off
            return max_eval
        else:
            min_eval = math.inf
            for r, c in empty_cells:
                b[r][c] = 'Y'
                eval_score = minimax(b, depth - 1, alpha, beta, True)
                b[r][c] = ''  # Backtrack
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha cut-off
            return min_eval

    # Main logic of myHeuristic
    empty_cells = get_empty_cells(board)
    
    # Safety fallback if called on a full board
    if len(empty_cells) == 0:
        return None 

    # Dynamic depth limit according to instructions
    if len(empty_cells) > 8:
        max_depth = 4
    else:
        max_depth = len(empty_cells)

    best_score = -math.inf
    best_move = empty_cells[0]
    alpha = -math.inf
    beta = math.inf

    # Evaluate all valid opening moves from the current state
    for r, c in empty_cells:
        board[r][c] = 'X'
        score = minimax(board, max_depth - 1, alpha, beta, False)
        board[r][c] = '' # Backtrack
        
        # If this path leads to a better outcome, log the move
        if score > best_score:
            best_score = score
            best_move = (r, c)
            
        alpha = max(alpha, best_score)

    return best_move