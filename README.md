# Writing a 2-layer MLP to solve MNIST for fun.

Everything is in mlp.cpp for now. This is just a personal project, so I'm not really expecting anyone to be looking here, but feel free too! :)

TODO:
- Add batching.
- Add more modularity; i.e. extend to more layers (?)

## Backward propagation notes. 

This of course requires writing the backward propagation step by hand, so I'll just write my derivations here for future reference.

### Output Layer. 

Loss function is simply $L(p, y) = -\sum_{i=1}^{o} y_{i} \log(p_{i})$, which is cross-entropy loss. The probability vector is defined component-wise as 

$$ p_{i} = \frac{\exp(z_{i}^{(2)})}{\sum_{j=1}^{o} \exp(z_{j}^{(2)})} $$

respectively. Hence, we have that 

$$ \log(p_{i}) = z_{i}^{(2)} - \log \left(\sum_{j=1}^{o} \exp(z_{j}^{(2)}) \right) $$ 

holds. We can further see that 

$$ \frac{\partial L(p, y)}{\partial z_{k}^{(2)}} = -\sum_{i=1}^{o} y_{i} \frac{\partial \log(p_{i})}{\partial z_{k}^{(2)}} $$

holds. So, it's obviously useful to compute $\frac{\partial \log(p_{i})}{\partial z_{k}^{(2)}}$. By above, we can see that 

$$ \frac{\partial \log(p_{i})}{\partial z_{k}^{(2)}} = \delta_{i, k} - \frac{\frac{\partial \left(\sum_{j=1}^{o} \exp(z_{j}^{(2)})\right)}{\partial z_{k}^{(2)}}}{\sum_{j=1}^{o} \exp(z_{j}^{(2)})} $$
 
From here, we can see that 

$$ \frac{\partial \left(\sum_{j=1}^{o} \exp(z_{j}^{(2)})\right)}{\partial z_{k}^{(2)}} = \exp(z_{k}^{(2)}) $$ 

holds. So, we have that 

$$ \frac{\partial \log(p_{i})}{\partial z_{k}^{(2)}} = \delta_{i, k} - \frac{\exp(z_{k}^{(2)})}{\sum_{j=1}^{o} \exp(z_{j}^{(2)})} = \delta_{i, k} - p_{k} $$ 

Thus, it follows that 

$$ \frac{\partial L(p, y)}{\partial z_{k}^{(2)}} = -\sum_{i=1}^{o} y_{i} \left(\delta_{i, k} - p_{k}\right) = -(y_{k} - p_{k}) = p_{k} - y_{k} $$ 

respectively. So, the gradient vector for this output layer is simply $p - y$ in vector form.

### Outer Parameters.

We now seek to find $\frac{\partial L(p, y)}{\partial W^{(2)}_{i, j}}$ as well as $\frac{\partial L(p, y)}{\partial b^{(2)}_i}$.

We have that 

$$ z_{i}^{(2)} = \sum_{j=1}^{d} W_{i, j}^{(2)} a_{j}^{(1)} + b^{(2)}_i $$

holds. So, by the chain rule, we have that 

$$ \frac{\partial L(p, y)}{\partial W_{i, j}^{(2)}} = \sum_{k=1}^{o} \frac{\partial L(p, y)}{\partial z_{k}^{(2)}} \frac{\partial z_{k}^{(2)}}{\partial W_{i, j}^{(2)}} = \frac{\partial L(p, y)}{\partial z_{i}^{(2)}} \frac{\partial z_{i}^{(2)}}{\partial W_{i, j}^{(2)}} $$ 

and

$$ \frac{\partial L(p, y)}{\partial b_{j}^{(2)}} = \sum_{k=1}^{o} \frac{\partial L(p, y)}{\partial z_{k}^{(2)}} \frac{\partial z_{k}^{(2)}}{\partial b_{j}^{(2)}} = \frac{\partial L(p, y)}{\partial z_{j}^{(2)}} \frac{\partial z_{j}^{(2)}}{\partial b_{j}^{(2)}} $$

holds. By above, we can see that 

$$ \frac{\partial z_{i}^{(2)}}{\partial W_{i, j}^{(2)}} = a_{j}^{(1)} $$ 

and 

$$ \frac{\partial z_{i}^{(2)}}{\partial b_{j}^{(2)}} = \delta_{i, j} $$ 

holds. Through the backpropagation algorithm, this is enough to update the gradients of this layer, with the obvious simplification here being that  

$$ \frac{\partial L(p, y)}{\partial b_{j}^{(2)}} = \frac{\partial L(p, y)}{\partial z_{j}^{(2)}} $$ 

holds respectively.



