# Layer

[github link](https://github.com/colecgulino/numpy-nn/blob/main/nn/layer.py)

This file lays out the interface to all of the other modules
that will be presented in the rest of the neural network layers.

This has two main functions which should be understandable to all of users of popular languages like Pytorch and Jax.

## Forward
```python
def forward(self, x: np.ndarray) -> tuple[np.ndarray, Cache]:
```
In the forward pass, we assume one input although `*args` and `**kwargs` can be passed in. The output is a tuple of the output tensor (assumed single) and a `Cache` object which is a dictionary that will be passed into the `backward` pass.

## Backward
```python
def backward(
    self,
    x: np.ndarray,
    cache: Cache,
    backwards_gradient: np.ndarray,
    gradients: dict[str, np.ndarray]
) -> np.ndarray:
```
In all of our backward passes, we assume that we have a gradient from the upstream level `backwards_gradient` or `din` which is the shape of the output layer of the forward pass. For us this will be the same as shape `y` of `[out_dim]`.

$$
backwards_gradient = din = [din_1, din_2]
$$

If we imagine three cascading functions
$$
y_1 = f_1(x)
$$
$$
y_2 = f_2(y_1)
$$
$$
y_3 = f_3(y_2)
$$

If we want to find the partial derivative $\partial y_3 / \partial x$, we can use the [chain rule](https://en.wikipedia.org/wiki/Chain_rule) to find the derivatives.

$$
\frac{\partial y_3}{\partial x} =
\frac{\partial y_3}{\partial y_2}
\frac{\partial y_2}{\partial y_1}
\frac{\partial y_1}{\partial x}
$$

If we think of this in an functional programming related style, the derivative of the first function $\partial y_1 / \partial x$ is completely defined within the function `f_1` itself if we know all of the previous gradients in the chain. So if we call the `backward` pass in a sequence starting from the last layer and going to first layer, as long as we maintain the rolling `backwards_gradient` which is just always $backwards_gradient = \partial out / \partial in$.

The input of `backwards_gradient` for all the functions are:
$$
f_2: din = \frac{\partial y_3}{\partial y_2}
$$
$$
f_3: din = \frac{\partial y_3}{\partial y_2}
\frac{\partial y_2}{\partial y_1}
$$


For all of our backwards gradients, we assume that the first layer is some loss function that returns a scalar, so that the the first node in the backwards graph always returns:
$$
\frac{\partial L}{\partial Y}
$$
Because $L$ is a scalar, the quantity $\partial L / \partial Y$ is a matrix of the same shape as $Y$.

For example, if we imagine that the matrix $Y$ is of shape `[M=2, C=2]` then we can write the expression as:
$$
\frac{\partial L}{\partial Y} = \begin{bmatrix}
 \frac{\partial L}{\partial y_{11}} && \frac{\partial L}{y_{12}} \\
 \frac{\partial L}{\partial y_{21}} && \frac{\partial L}{y_{22}}
\end{bmatrix}
$$ 

All of the quantities calculated will be based on this, for example $\partial L / \partial W$. Some of the intermediate quantities will be Jacobians (i.e. matrices with matrix derivatives) but we can avoid this with some tricks as will illustrate in other parts of the documentation.
## Other Interfaces

These two functions make up the bulk of the functionality of the `Layer` class. The thing to call out is the `Cache` dataclass `Cache = dict[str, np.ndarray]` which is the way we cache out information for the backwards pass.

Another function here is a recurssive function for updating the parameters of a `Layer` and all attributes of that `Layer`. This allows us to pass in specific parameters to update nested `Layer`s.
```python
def update_parameters(self, parameters: dict[str, np.ndarray]) -> None:
```