DO NOT USE THE INTERNET.

You are playing a board game. The board is a 3x3 grid. On each turn, you place 'X' and the opponent places 'Y' on an empty cell. The game ends when either player gets three of their marks in a row, column, or diagonal, or when the board is full. If the board is full and no player has three of their marks in a row, the game is a draw.

Write a heuristic function for this game in python that avoids losing, even if it means drawing. Winning should be the first priority, then drawing, then losing.

The heuristic function for this game will take as argument the current state of the board as a 2D array and return a tuple of integers (row, col) representing the best possible move for the board state.

The name of the heuristic function will be `myHeuristic`.
