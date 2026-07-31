# CompReBench


## Pipeline for Computationally Verifiable Turn Based Games

For each to-be-benchmarked model:

1. In a single prompt, explain the all rules of a game in detail (win, lose, draw, moves, states, etc). Ask it to write a heuristic function for the game in a specific format that will be used by a game engine. The prompt must explicitly state that the model will not be allowed to use the internet.

2. Use the generated heuristic in a deterministic program against another bot/heuristic that is expected to win 50% of matches. Such a heuristic is a modified version of the best heuristic available, only the final result is modified to be equal to the original result 50% of the time.

3. Run this for 500 times. Save the win and draw counts. Final Comprehensibility score (out of 100) = (win_cnt + draw_cnt) * 0.2. 