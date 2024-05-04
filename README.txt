"Modern applied deep learning with reinforcement methodology"

Special syllabus Spring 2024
Norwegian University of Life Sciences (NMBU)

---

This repository contains theory, implementation and  examples for various reinforcement learning
algorithms. Said algorithms are implemented in Python (using `PyTorch` and to some extent
`ml-explore`), and are taught to play various games from the `gymnasium` library, ranging from
simple to complex in approximate order:

frozen-lake
  Tabular Q-learning
  * input space     [16,]
  * action space    [4,]

cart-pole
  REINFORCE and deep Q-learning
  * input space     [4,]
  * action space    [2,]

enduro
  Deep Q-learning
  * input space     [210, 160, 1]
  * action space    [9,]

breakout (suboptimal results)
  Deep Q-learning
  * input space     [210, 160, 1]
  * action space    [4,]

tetris (suboptimal results)
  Deep Q-learning
  * input space     [210, 160, 1]
  * action space    [5,]

---

The theory is presented in `report.pdf`, along with results and simplified implementation examples.

The implementation, examples and results are presented in their corresponding directories. During
training of the latter four games, Orion HPC (https://orion.nmbu.no) at the Norwegian University of
Life Sciences (NMBU) provided computational resources.

N.b., in order for the examples to access atari games from `gymnasium`, Python<=3.10 must be used.

---

Relevant papers:

- "Human-level control through deep reinforcement learning"
                                    doi:10.1038/nature14236
- "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm"
                                                                        arXiv:1712.01815v1

---

Learning goals:

- Understand and know how to build, use and deploy reinforcement learning algorithms
  * Experiment with reinforcement agent(s) (for instance playing video-games)

Learning outcomes:

- Be competent in modern deep learning situations
  * Understand (and to some extent be able to reproduce) cutting-edge "artificial intelligence"
    models
