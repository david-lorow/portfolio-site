The math for how the model works, comes out to a linear equation that is interpreted as a probability. The digits 0 to 9 yield 10 distinct **classes**. Each input image has pixels (or dimensions) 28 × 28, which results in 784 total **input features** after flattening. So essentially a flattened vector representing pixel values will be fed in and the predicted digit based on the input image will come out. Meaning, for each digit $k$ every input pixel's brightness is multiplied by a weight based on that particular pixel's importance to that digit. The sum of all those multiplications plus a bias term per class, produces the logit. Ten logits are produced from each input image, but ultimately only one will be chosen (after argmax) as the most probable. 

The logits for this linear classifier can be expressed as:

$$
z_k = b_k + \sum_{i=1}^{784} W_{k,i}\ x_i
$$

where:

- $x_i$ represents the $i$-th pixel value of the flattened input image
- $W_{k,i}$ is the weight associated with pixel $i$ for class $k$
- $b_k$ is the bias term for class $k$
- $z_k$ is the resulting logit (pre-softmax), after which argmax will provide de facto classification for class $k$