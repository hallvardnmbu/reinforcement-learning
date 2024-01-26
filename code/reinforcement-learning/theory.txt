Motivation
----------

Reinforcement learning is, in simple terms, a method where an artificial neural network (hereafter
Agent) is placed in an environment with instant or delayed feedback in regard to the current state.
This feedback is what makes the Agent able to learn. An example of this would be a chess-playing
Agent, whose input, for instance, is an image of the board, and output its next move in response to
the state of the game. The feedback would then be piece captures, position evaluations, etc.

An Agent usually starts its learning process from scratch, and has to figure out what moves lead to
positive feedback. Therefore, many iterations are needed to learn weights similar to, for instance,
human levels. This process may in some cases be challenging, especially if the feedback is
significantly delayed – which makes it hard for the Agent to determine which of the outputs led to
the positive feedback.

OpenAIs work on mastering Minecraft had a goal for their Agent to obtain diamond tools, which
"usually takes proficient humans over 20 minutes (24,000 actions)" (i.e. extremely delayed
feedback), and achieved this through sequential rewards [1]. Likewise, if the feedback for a
chess-Agent is the end-result of a game, it is hard for the Agent to distinguish between bad and
good moves throughout the game. Therefore, like OpenAIs solution, step-wise feedback (e.g. position
evaluations in chess) is required to speed up the learning process.

Action-value function
---------------------




Sources
[1] https://openai.com/research/vpt
[2] Human-level control through deep reinforcement learning