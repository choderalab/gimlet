# GIN
Graph-based Interatomic Networks. A package for modelling, learning, and inference on molecular topological space written in Python and TensorFlow.

## Dependencies
`gin` doens't depend on _ANY_ pacakges except `TensorFlow 2.0 `.

## Examples
Minimal working example: https://github.com/choderalab/gin/blob/master/tonic/scripts/mini_example_ESOL.ipynb

## Authors
- yuanqing wang `<yuanqing.wang@choderalab.org>` (while at Chodera Lab at Memorial Sloan Kettering Cancer Center, Weill Cornell Medicine, and City University of New York -- City College of New York.)

## Manifest
*`gin/` the core (and fun) part of the package.
    *`i_o/` reading and writing popular molecule embedding/representing structures.
    *`deterministic/` simple property predictions based on molecular mechanics.
    *`probabilistic/` infrastructure for molecular machine learning, especially graph networks.
*`tonic/` auxiliary scripts.
    *`for_biologists/` ready-to-use modules and scripts that does not require knowledge, experience, or intelligence of any sort.
    *`architectures/` off-the-shelf model architectures developed elsewhere.
    *`scripts/` fun scripts we used to generate data and hypothesis.
    *`trained_models/` _Nomen est omen_.
