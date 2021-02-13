# nndescent

This code implements following paper:

    Wei Dong et al., "Efficient K-Nearest Neighbor Graph Construction for Generic Similarity Measures", WWW11
    
Some additional join algorithms are added:
- join the center node to its nbd nodes
- random join (join random nodes)
- randomly break the tie

## Build Requirements

- C++ compiler (needed support for C++11 or later)
- Python3 with matplotlib

## Examples
I use main.cpp to generate 100 points and draw images for each step:
- image for initial 100 points
- image for initial graph with random neighbours
- image after iteration1 by nndescent
- image after iteration2 by nndescent
- image after iteration3 by nndescent
- image after iteration4 by nndescent
- image after iteration5 by nndescent
- image after iteration6 by nndescent
- image after iteration7 by nndescent
