"""
Christmax tree for FedeRez
             /\
            <  >
             \/
             /\
            /  \
           /++++\
          /  ()  \
          /      \
         /~`~`~`~`\
        /  ()  ()  \
        /          \
       /*&*&*&*&*&*&\
      /  ()  ()  ()  \
      /              \
     /++++++++++++++++\
    /  ()  ()  ()  ()  \
    /                  \
   /~`~`~`~`~`~`~`~`~`~`\
  /  ()  ()  ()  ()  ()  \
  /*&*&*&*&*&*&*&*&*&*&*&\
 /                        \
/,.,.,.,.,.,.,.,.,.,.,.,.,.\
           |   |
          |`````|
          \_____/

"""

import numpy as np
import perceval as pcvl
from perceval.components import BS, PERM
import matplotlib.pyplot as plt

# Circuit definitions

M = 25  # Number of modes : 2*m+1
N = 10000  # Number of samples

# Tree aesthetics

HEIGHT_FIRST_CONE = 15
WIDTH_FIRST_CONE = 5

HEIGHT_SECOND_CONE = 22
WIDTH_SECOND_CONE = 10

HEIGHT_THIRD_CONE = 26
WIDTH_THIRD_CONE = 4

WIDTH_BOTTOM = 6

HEIGHT_BOTTOM = 5

NUM_LINES = (
    HEIGHT_FIRST_CONE
    + HEIGHT_SECOND_CONE
    + HEIGHT_THIRD_CONE
    + HEIGHT_BOTTOM
    - WIDTH_FIRST_CONE
    - WIDTH_SECOND_CONE
    + 4
)


def sampling_level_symmetric(level: int):
    """Draw two points at m - level and m + level
    with probability 50%, 50%
    """
    circuit = pcvl.Circuit(2 * M + 1)
    if level > 0:
        circuit.add(M, BS())
        for i in range(0, level):
            circuit.add(M - i - 1, PERM([1, 0]))
            if i > 0:
                circuit.add(M + i, PERM([1, 0]))

    p = pcvl.Processor("SLOS", circuit)
    p.with_input(pcvl.BasicState([0] * M + [1] + [0] * M))
    sampler = pcvl.algorithm.Sampler(p)
    sample_count = sampler.sample_count(N)
    return sample_count["results"]


def sampling_level_line(l: int, excl: int):
    """Draw two lines, each of them of length l,
    exluced by a region of 2*excl-1 in the the center

    l=2
    excl=2
    m=5

    ###**###**###

    with probability 1/(2*l)
    """

    # First draw a line from spatial mode 0 to spatial mode 2*l
    L = 2 * l + (2 * excl - 1)
    offset = (2 * M + 1 - L) // 2

    circuit = pcvl.Circuit(2 * M + 1)
    for i in range(2 * l):
        circuit.add(i, BS(BS.r_to_theta(1 / (2 * l - i))))

    # Use a nice permutation to get the good result
    perm = (
        [offset + i for i in range(l)]
        + [offset + l + (2 * excl - 1) + i for i in range(l)]
        + [i for i in range(offset)]
        + [offset + l + i for i in range(2 * excl - 1)]
    )
    circuit.add(0, PERM(perm))

    p = pcvl.Processor("SLOS", circuit)
    p.with_input(pcvl.BasicState([1] + [0] * 2 * M))
    sampler = pcvl.algorithm.Sampler(p)
    sample_count = sampler.sample_count(N)
    return sample_count["results"]


def update_matrix(matrix, results):
    for k, v in results.items():
        for i, t in enumerate(k):
            if t:
                matrix[l][i] += v


# Compute the matrix of the tree

l = 0
matrix = np.zeros((NUM_LINES, 2 * M + 1))
for level in range(HEIGHT_FIRST_CONE):
    update_matrix(matrix, sampling_level_symmetric(level))
    l += 1

update_matrix(
    matrix,
    sampling_level_line(HEIGHT_FIRST_CONE - WIDTH_FIRST_CONE, WIDTH_FIRST_CONE),
)
l += 1

for level in range(WIDTH_FIRST_CONE, HEIGHT_SECOND_CONE):
    update_matrix(matrix, sampling_level_symmetric(level))
    l += 1

update_matrix(
    matrix,
    sampling_level_line(HEIGHT_SECOND_CONE - WIDTH_SECOND_CONE, WIDTH_SECOND_CONE),
)
l += 1

for level in range(WIDTH_SECOND_CONE, HEIGHT_THIRD_CONE):
    update_matrix(matrix, sampling_level_symmetric(level))
    l += 1

update_matrix(
    matrix, sampling_level_line(HEIGHT_THIRD_CONE - WIDTH_THIRD_CONE, WIDTH_THIRD_CONE)
)
l += 1


for _ in range(HEIGHT_BOTTOM):
    update_matrix(matrix, sampling_level_symmetric(WIDTH_THIRD_CONE))
    l += 1

update_matrix(matrix, sampling_level_line(WIDTH_THIRD_CONE, 1))

# Plot the tree
# Use log trick to have less variations in the color map.
plt.figure()
plt.imshow(
    10 * np.log10(1 + np.log10(1 + matrix)), cmap="Greens", interpolation="nearest"
)
plt.xlabel("Probability of each spatial mode")
plt.ylabel("Circuit")
plt.title("Joyeux Noël FedeRez !")
plt.colorbar()
plt.show()
