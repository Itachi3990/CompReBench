DO NOT USE THE INTERNET.

You are playing a board game. The board is a 4x4 grid. On each turn, you place 'X' and the opponent places 'Y' on an empty cell ''. The game ends when either player gets four of their marks in a row, column, or diagonal, or when the board is full. If the board is full and no player has four of their marks in a row, the game is a draw.

Write a heuristic function for this game in python that avoids losing, even if it means drawing. Winning should be the first priority, then drawing, then losing.

The heuristic function for this game will take as argument the current state of the board as a 2D array and return a tuple of integers (row, col) representing the best possible move for the board state.
The search space may get exponentially large so your heuristic should look ahead upto 4 levels of depth if the number of empty cells is > 8, and evaluate the board state using a helper of the heuristic function. Otherwise it should look ahead until the end of the game.

Make sure that the name of the heuristic function is `myHeuristic`. It will be called by an external script.
