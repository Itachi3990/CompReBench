def myHeuristic(board):
    """
    Evaluates the 3x3 grid and returns the best move (row, col) for 'X'.
    Prioritizes winning first, drawing second, and avoiding loss at all costs.
    """
    # Create a mutable copy of the board state
    b = [list(row) for row in board]

    def check_winner(state):
        # Check rows, columns, and diagonals
        for i in range(3):
            if state[i][0] == state[i][1] == state[i][2] and state[i][0] in ('X', 'Y'):
                return state[i][0]
            if state[0][i] == state[1][i] == state[2][i] and state[0][i] in ('X', 'Y'):
                return state[0][i]

        if state[0][0] == state[1][1] == state[2][2] and state[0][0] in ('X', 'Y'):
            return state[0][0]
        if state[0][2] == state[1][1] == state[2][0] and state[0][2] in ('X', 'Y'):
            return state[0][2]

        return None

    def is_full(state):
        return all(cell in ('X', 'Y') for row in state for cell in row)

    def minimax(state, is_maximizing):
        winner = check_winner(state)
        if winner == 'X':
            return 1   # Win score
        if winner == 'Y':
            return -1  # Loss score
        if is_full(state):
            return 0   # Draw score

        if is_maximizing:
            best_score = -float('inf')
            for r in range(3):
                for c in range(3):
                    if state[r][c] not in ('X', 'Y'):
                        temp = state[r][c]
                        state[r][c] = 'X'
                        score = minimax(state, False)
                        state[r][c] = temp
                        best_score = max(best_score, score)
            return best_score
        else:
            best_score = float('inf')
            for r in range(3):
                for c in range(3):
                    if state[r][c] not in ('X', 'Y'):
                        temp = state[r][c]
                        state[r][c] = 'Y'
                        score = minimax(state, True)
                        state[r][c] = temp
                        best_score = min(best_score, score)
            return best_score

    best_val = -float('inf')
    best_move = None

    # Evaluate all valid available moves for player 'X'
    for r in range(3):
        for c in range(3):
            if b[r][c] not in ('X', 'Y'):
                temp = b[r][c]
                b[r][c] = 'X'
                move_val = minimax(b, False)
                b[r][c] = temp

                if move_val > best_val:
                    best_val = move_val
                    best_move = (r, c)

    return best_move