def myHeuristic(board):
    """
    Evaluates the current 4x4 board state and returns the optimal move (row, col) for 'X'.
    Prioritizes: Win > Draw > Loss.
    
    Depth Limit Rules:
      - Empty cells > 8: Lookahead up to 4 moves (depth 4).
      - Empty cells <= 8: Lookahead until the end of the game (full depth).
    """
    empty_cells = sum(row.count('') for row in board)
    
    # Determine depth limit based on remaining empty cells
    max_depth = 4 if empty_cells > 8 else empty_cells
    
    best_score = float('-inf')
    best_move = None
    alpha = float('-inf')
    beta = float('inf')

    # Order moves to optimize Alpha-Beta pruning (center cells evaluated first)
    possible_moves = get_possible_moves(board)

    for r, c in possible_moves:
        board[r][c] = 'X'
        
        # Check immediate win
        if check_winner(board) == 'X':
            board[r][c] = ''
            return (r, c)
            
        score = minimax(board, max_depth - 1, False, alpha, beta)
        board[r][c] = ''

        if score > best_score:
            best_score = score
            best_move = (r, c)
            
        alpha = max(alpha, best_score)

    return best_move if best_move is not None else (possible_moves[0] if possible_moves else (0, 0))


def minimax(board, depth, is_maximizing, alpha, beta):
    winner = check_winner(board)
    
    # Terminal states
    if winner == 'X':
        return 100000 + depth  # Prefer faster wins
    if winner == 'Y':
        return -100000 - depth  # Delay losses
    if is_board_full(board):
        return 0  # Draw
    if depth == 0:
        return evaluate_board(board)

    if is_maximizing:
        max_eval = float('-inf')
        for r, c in get_possible_moves(board):
            board[r][c] = 'X'
            eval_score = minimax(board, depth - 1, False, alpha, beta)
            board[r][c] = ''
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for r, c in get_possible_moves(board):
            board[r][c] = 'Y'
            eval_score = minimax(board, depth - 1, True, alpha, beta)
            board[r][c] = ''
            min_eval = min(min_eval, eval_score)
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return min_eval


def evaluate_board(board):
    """
    Static evaluation function for non-terminal board states at the maximum depth limit.
    Evaluates lines (rows, columns, diagonals) based on potential open sequences.
    """
    lines = get_all_lines(board)
    score = 0

    for line in lines:
        x_count = line.count('X')
        y_count = line.count('Y')

        # If a line contains both 'X' and 'Y', neither can complete it (blockage)
        if x_count > 0 and y_count > 0:
            continue
        
        # Favorable states for X
        if x_count == 3:
            score += 100
        elif x_count == 2:
            score += 10
        elif x_count == 1:
            score += 1

        # Defensive penalties against Y
        if y_count == 3:
            score -= 120  # Strongly penalize opponent's near-complete lines
        elif y_count == 2:
            score -= 12
        elif y_count == 1:
            score -= 1

    return score


def check_winner(board):
    for line in get_all_lines(board):
        if line.count('X') == 4:
            return 'X'
        if line.count('Y') == 4:
            return 'Y'
    return None


def is_board_full(board):
    return all(cell != '' for row in board for cell in row)


def get_all_lines(board):
    """Returns all 10 lines (4 rows, 4 columns, 2 main diagonals)."""
    lines = []
    
    # Rows
    for row in board:
        lines.append(row)
        
    # Columns
    for c in range(4):
        lines.append([board[r][c] for r in range(4)])
        
    # Diagonals
    lines.append([board[i][i] for i in range(4)])
    lines.append([board[i][3 - i] for i in range(4)])
    
    return lines


def get_possible_moves(board):
    """
    Returns available (row, col) coordinates.
    Moves closer to the center are prioritized for better alpha-beta pruning efficiency.
    """
    moves = []
    for r in range(4):
        for c in range(4):
            if board[r][c] == '':
                moves.append((r, c))
                
    # Center-first ordering heuristic for faster cutoff
    center_priority = [(1, 1), (1, 2), (2, 1), (2, 2), 
                       (0, 1), (0, 2), (1, 0), (2, 0), (1, 3), (2, 3), (3, 1), (3, 2),
                       (0, 0), (0, 3), (3, 0), (3, 3)]
                       
    return [pos for pos in center_priority if pos in moves]