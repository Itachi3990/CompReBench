# CompReBench


## Pipeline for Computationally Verifiable Turn Based Games

For each to-be-benchmarked model:

1. In a single prompt, explain the all rules of a game in detail (win, lose, draw, moves, states, etc). Ask it to write a prompt that will generate a heuristic for the game in a specific format that will be used by a game engine.  That generated prompt should be such that it will only contain the instruction to write the function, and will not provide any context whatsoever of the game. The output prompt must explicitly state that the model will not be allowed to use the internet.

2. Feed the output prompt to a model that is good at coding. This is to remove the bias that arise from each model's differing abilities to code.

3. use the generated heuristic in a deterministic program against another bot/heuristic, the weakest available heuristic capable of winning against a random bot.

4. Run this for 500 times. Save the win and draw counts. Final Comprehensibility score (out of 100) = (win_cnt + draw_cnt) * 0.2. 