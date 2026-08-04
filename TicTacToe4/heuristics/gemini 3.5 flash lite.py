def myHeuristic(board):
    """
    Finds the best move for the current player ('X') on a 4x4 board.
    Priorities: 1. Win, 2. Draw, 3. Avoid Loss.
    Looks ahead up to 4 levels if empty cells > 8, otherwise searches to terminal states.
    """
    def get_empty_cells(b):
        return [(r, c) for r in range(4) for c in range(4) if b[r][c] == '']

    def check_win(b, player):
        # Rows and columns
        for i in range(4):
            if all(b[i][c] == player for c in range(4)):
                return True
            if all(b[r][i] == player for r in range(4)):
                return True
        # Diagonals
        if all(b[i][i] == player for i in range(4)):
            return True
        if all(b[i][3 - i] == player for i in range(4)):
            return True
        return False

    def is_board_full(b):
        return all(b[r][c] != '' for r in range(4) for c in range(4))

    def evaluate_board(b):
        if check_win(b, 'X'):
            return 1000
        if check_win(b, 'Y'):
            return -1000
        return 0

    def minimax(b, depth, max_depth, is_maximizing, alpha, beta):
        if check_win(b, 'X'):
            return 1000 + depth # Prefer faster wins
        if check_win(b, 'Y'):
            return -1000 - depth # Prefer slower losses (though we want to avoid)
        if is_board_full(b) or depth == max_depth:
            return evaluate_board(b)

        empty_cells = get_empty_cells(b)

        if is_maximizing:
            max_eval = -float('inf')
            for r, c in empty_cells:
                b[r][c] = 'X'
                eval = minimax(b, depth + 1, max_depth, False, alpha, beta)
                b[r][c] = ''
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for r, c in empty_cells:
                b[r][c] = 'Y'
                eval = minimax(b, depth + 1, max_depth, True, alpha, beta)
                b[r][c] = ''
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

    empty = get_empty_cells(board)
    if not empty:
        return None

    # Determine max depth based on empty cells count
    max_depth = 4 if len(empty) > 8 else len(empty)

    best_val = -float('inf')
    best_move = empty[0]
    alpha = -float('inf')
    beta = float('inf')

    for r, c in empty:
        board[r][c] = 'X'
        move_val = minimax(board, 1, max_depth, False, alpha, beta)
        board[r][c] = ''

        if move_val > best_val:
            best_val = move_val
            best_move = (r, c)

    return best_move