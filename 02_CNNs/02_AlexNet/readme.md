# AlexNet - The Pioneer

AlexNet was the breakthrough architecture that won the ImageNet challenge in 2012, marking the beginning of the deep learning revolution. However, today it is widely considered **obsolete**. But it changed the AI Game as Jensen Huang said.

## Why is it Obsolete?
- **High Parameter Count**: Over 60 million parameters, mainly due to the massive fully connected (dense) layers at the end.
- **Inefficiency**: Modern architectures like ResNet (using skip connections) or EfficientNet achieve better performance with fewer parameters and lower compute costs.
- **Historical Techniques**: It relies on Local Response Normalization (LRN), which has been replaced by more effective methods like Batch Normalization.

Because this architecture is no longer state-of-the-art and is primarily studied for its historical significance, I have trained it for only **30 epochs**.

---

## Original vs. Modern Version

### Original: Dual-GPU Design
The original 2012 implementation was split across two NVIDIA GTX 580 GPUs because they only had 3GB of memory each. This split is visible in the diagram below:

![AlexNet Original Dual-GPU Architecture](./AlexNet_Dual_GPU.jpg)

### Modern: Single Stream
The modern version, used in standard libraries like PyTorch's `torchvision`, simplifies this into a single pipeline as seen here:

![AlexNet Modern Architecture](./AlexNet.png)

## Implementation
The implementation in [AlexNet.ipynb](./AlexNet.ipynb) has been significantly updated to more closely reflect the characteristics of the original 2012 paper:

- **Optimizer**: Switched from AdamW back to **SGD with Momentum (0.9)** and **Weight Decay (5e-4)**.
- **Normalization**: Restored **Local Response Normalization (LRN)** layers after the first two convolutional blocks.
- **Scheduler**: Implemented **ReduceLROnPlateau** to automate the original paper's heuristic of dividing the learning rate by 10 whenever validation error stops improving.
