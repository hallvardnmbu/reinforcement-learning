"Modern applied deep learning with reinforcement methodology"

Special syllabus Spring 2024
Norwegian University of Life Sciences (NMBU)

---

This repository contains theory, implementation and  examples for various reinforcement learning
algorithms. Said algorithms are implemented in Python (using `PyTorch` and to some extent
`ml-explore`), and are taught to play various games from the `gymnasium` library, ranging from
simple to complex in the approximate order:

frozen-lake
  Tabular Q-learning
  * input space     [16,]
  * action space    [4,]

cart-pole
  REINFORCE and deep Q-learning
  * input space     [4,]
  * action space    [2,]

tetris
  Deep Q-learning
  * input space     [128,]
  * action space    [5,]

breakout
  Deep Q-learning
  * input space     [210, 160, 1]
  * action space    [4,]

enduro
  Deep Q-learning
  * input space     [210, 160, 1]
  * action space    [9,]

The implementation, examples and results are pr