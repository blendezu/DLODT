# DenseNet — Summary

## Overview
DenseNet (Densely Connected Convolutional Networks) introduces dense connectivity between layers: each layer receives, as input, the feature-maps of all preceding layers within the same dense block. This design improves gradient flow, encourages feature reuse, and reduces the number of parameters required for a given accuracy.

## Core concept: dense connectivity
- Traditional CNN: each layer connects only to the next layer.
- DenseNet: in a dense block, every layer is connected to every other layer in a feed-forward fashion.
- Connection count: for an L-layer block, DenseNet has L(L + 1) / 2 direct connections (not just L).

## Mathematical formulation
- Standard layer: x_l = H_l(x_{l-1})
- ResNet-style (skip-add): x_l = H_l(x_{l-1}) + x_{l-1}
- DenseNet (concatenation):  
  x_l = H_l([x_0, x_1, ..., x_{l-1}])  
  where [x_0, ..., x_{l-1}] denotes concatenation of feature-maps from layers 0..l-1 into a single tensor.

Example: if H_1 outputs 12 feature-maps and the original input had 3 channels, H_2 receives 3 + 12 = 15 input channels.

## Architecture components
- Dense blocks: groups of layers that use concatenation to share features. All layers inside a block produce feature-maps that are concatenated and passed to subsequent layers within the block.
- Transition layers: placed between dense blocks to change feature-map size. A transition layer typically contains:
  - Batch Normalization
  - 1×1 Convolution (to adjust channel count)
  - 2×2 Average Pooling (for down-sampling)

## Growth rate (k)
- The growth rate k defines how many new feature-maps each layer contributes to the collective feature set.
- DenseNets often use small k (e.g., k = 12) because layers can reuse earlier features and do not need to relearn redundant information.

## Efficiency variants (DenseNet-B, DenseNet-C, DenseNet-BC)
- Bottleneck layers (DenseNet-B): insert a 1×1 convolution before a 3×3 convolution to reduce the number of input feature-maps, improving computation efficiency.
- Compression (DenseNet-C): in transition layers, reduce the number of feature-maps by a factor θ (0 < θ ≤ 1), e.g., θ = 0.5 halves channels.
- DenseNet-BC: combines bottleneck layers and compression for maximal compactness and efficiency.

## Advantages
- Improved gradient flow and reduced vanishing-gradient issues: each layer has direct access to loss gradients and the original input.
- Parameter efficiency: achieves competitive accuracy with fewer parameters than many alternatives (e.g., some ResNets).
- Feature reuse: earlier feature-maps are reused throughout the network, enabling more efficient representations.

## Quick implementation notes
- Typical settings: growth rate k ∈ {12, 24, 32}; compression θ ≈ 0.5 for compact models.
- When implementing, ensure matching spatial sizes before concatenation (use transition layers or matching pooling/upsampling).
- Bottleneck pattern: BN → ReLU → 1×1 Conv → BN → ReLU → 3×3 Conv (inside a dense layer).

## References and further reading
- For original experiments and exact architectures (DenseNet-121/169/201/264) consult the DenseNet paper and implementation notes in common deep learning libraries.
