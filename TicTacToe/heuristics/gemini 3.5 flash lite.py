def myHeuristic(board):
    """
    Heuristic function for Tic-Tac-Toe (X vs Y).
    Returns the best move as a (row, col) tuple.
    Priority:
    1. Win immediately if possible.
    2. Block opponent's immediate win.
    3. Take the center if available.
    4. Take a corner if available.
    5. Take any remaining available edge.
    """
    
    def check_winner(b, player):
        # Check rows, columns, diagonals for a win
        for i in range(3):
            if all(b[i][j] == player for j in range(3)): return True
            if all(b[j][i] == player for j in range(3)): return True
        if all(b[i][i] == player for i in range(3)): return True
        if all(b[i][2 - i] == player for i in range(3)): return True
        return False

    def get_empty_cells(b):
        return [(r, c) for r in range(3) for c in range(3) if b[r][c] == '' or b[r][c] is None]

    empty_cells = get_empty_cells(board)
    if not empty_cells:
        return None

    # 1. Win immediately if possible ('X')
    for r, c in empty_cells:
        board[r][c] = 'X'
        if check_winner(board, 'X'):
            board[r][c] = '' # Reset board state
            return (r, c)
        board[r][c] = ''

    # 2. Block opponent's immediate win ('Y')
    for r, c in empty_cells:
        board[r][c] = 'Y'
        if check_winner(board, 'Y'):
            board[r][c] = '' # Reset board state
            return (r, c)
        board[r][c] = ''

    # 3. Take the center if available
    if (1, 1) in empty_cells:
        return (1, 1)

    # 4. Take a corner if available
    corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
    available_corners = [corner for corner in corners if corner in empty_cells]
    if available_corners:
        return available_corners[0]

    # 5. Take any remaining edge
    edges = [(0, 1), (1, 0), (1, 2), (2, 1)]
    available_edges = [edge for edge in edges if edge in empty_cells]
    if available_edges:
        return available_edges[0]

    return empty_cells[0]