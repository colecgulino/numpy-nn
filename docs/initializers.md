# Initialization

Initialization is an important part of neural network training. Starting off with properly initialized weights ensures that training will be smooth and proper.

## Xavier Uniform Initialization

Xavier Uniform initialization is based on the neural network layer to ensure that the variance in the input and the output are the same and that the gradients have a reasonable scale.

So for some neural network with weights $W$, we want to ensure that for the function $y = Wx$ that:
$$
\text{Var}[y] = \text{Var}[x]
$$

Given this dense connection, we can see:
$$
y = \sum_{i=1}^{n_{in}}W_ix_i
$$

Given the definition of variance:
$$
\text{Var}[y] = E\left[y^2\right] -(E\left[y\right])^2
$$

Finding the expectation of the network is:
$$
E\left[y\right] = \sum_{i=1}^{n_{in}}E[W_i]E[x_i]
$$

If we assume zero mean for $W$ and $x$, we can say then $E[y] = 0$.

$$
\text{Var}[y] = E\left[
\left(
\sum_{i=1}^{n_{in}}W_ix_i
\right) ^ 2
\right]
$$

Exapanding the square:
$$
\text{Var}[y] = E\left[
\sum_{i=1}^{n_{in}}W_i^2x_i^2 + 
2\sum_{i\ne j}W_ix_iWjx_j
\right]
$$

Since they are independent and zero mean, we get rid of the second term.
$$
\text{Var}[y] = \sum_{i=1}^{n_{in}}E[W_i^2]
E[x_i^2]
$$
Because they are assumed zero mean, we can say that $\text{Var}[W] = E[W^2]$ and $\text{Var}[x] = E[x^2]$.

$$
\text{Var}[y] = \sum_{i=1}^{n_{in}}\text{Var}[W]\cdot \text{Var}[x]
$$
Since all terms are identical:
$$
\text{Var}[y] = n_{in}\cdot\text{Var}[W]\cdot\text{Var}[x]
$$

From this, we can find $\text{Var}[W]$ as:
$$
\text{Var}[W] = \frac{1}{n_{in}}
$$

For backwards, we assume something like the backwards gradient $\partial L / \partial Y$.

$$
\frac{\partial L}{\partial X} = W^{T}\frac{
    \partial L}{\partial Y}
$$

$$
\text{Var}\left[\frac{\partial L}{\partial X}\right] = 
\text{Var}[W] \text{Var}\left[
    \frac{\partial L}{\partial X}
\right] n_{out}
$$
$$
\text{Var}[W] = \frac{1}{n_{out}}
$$

To satisfiy both conditions, we can say:
$$
\text{Var}[W] = \frac{2}{n_{in} + n_{out}}
$$

So the distributions we we can express:
### Uniform
$$
W \sim U\left[
    -\frac{\sqrt{6}}{\sqrt{n_{in} + n_{out}}},
    \frac{\sqrt{6}}{\sqrt{n_{in} + n_{out}}}
\right]
$$
### Normal
$$
W \sim \mathcal{N}\left[
    0,
    \frac{2}{n_{in} + n_{out}}
\right]
$$
