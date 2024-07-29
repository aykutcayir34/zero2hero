#%%
from engine import Value, draw_dot
import random 

class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def parameters(self):
        return self.w + [self.b]

    def __call__(self, x):
        # w * x + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

    def __call__(self, x):
        outs = [neuron(x) for neuron in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
# %%
x = [2.0, 3.0, -1.0]
n = MLP(3, [4, 4, 1])
n(x)
# %%

# %%
draw_dot(n(x))
# %%
n(x).backward()
draw_dot(n(x))
# %%
n.parameters()
# %%
xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0]
      ]
ys = [1.0, -1.0, -1.0, 1.0]
n = MLP(3, [4, 4, 1])
# %%
for k in range(20):
    ypred = [n(x) for x in xs]
    loss = sum([(yp - y)**2 for yp, y in zip(ypred, ys)], Value(0.0))
    loss.backward()
    for p in n.parameters():
        p.data -= p.grad * 0.01

    print(k, loss.data)

# %%
ypred = [n(x) for x in xs]
# %%
ypred
# %%
