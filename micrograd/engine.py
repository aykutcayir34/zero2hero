#%%
import math
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
# %%
#def f(x):
#    return 3 * x ** 2 - 4 * x + 5
# %%
#f(3.0)
# %%
#xs = np.arange(-5, 5, 0.25)
#ys = f(xs)
#plt.plot(xs, ys)
# %%
class Value:
    def __init__(self, data, _children=(), _op='', label=''): 
        self.data = data
        self.prev = set(_children)
        self._op = _op
        self.label = label
        self.grad = 0.0
        self._backward = lambda: None 
    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self * other
    
    def __radd__(self, other): # other + self
        return self + other
    
    def __rsub__(self, other): # other - self
        return other + (-self)
    
    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float for now"
        out = Value(self.data ** other, (self,), f'**{other}')
        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self * other**-1
    
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)
    
    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward 
        return out
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
# %%
#a = Value(2.0, label='a')
#b = Value(-3.0, label='b')
#c = Value(10.0, label='c')
#e = a*b; e.label='e'
#d = e+c; d.label='d'
#f = Value(-2.0, label='f')
#L = d*f; L.label='L'
#L
# %%
from graphviz import Digraph
#%%
def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v.prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges
# %%
def draw_dot(root):
    nodes, edges = trace(root)
    dot = Digraph(format="svg", graph_attr={'rankdir': 'LR'})
    for n in nodes:
        uid = str(id(n))
        dot.node(name=uid, label="{%s | data %.4f | grad %.4f}" % (n.label, n.data, n.grad), shape='record')
        if n._op:
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    return dot
# %%
#draw_dot(L)
# %%
#x1 = Value(2.0, label='x1')
#x2 = Value(0.0, label='x2')
#w1 = Value(-3.0, label='w1')
#w2 = Value(1.0, label='w2')
#b = Value(6.7, label='b')
#x1w1 = x1*w1; x1w1.label='x1w1'
#x2w2 = x2*w2; x2w2.label='x2w2'
#x1w1x2w2 = x1w1+x2w2; x1w1x2w2.label='x1w1 + x2w2'
#n = x1w1x2w2+b; n.label='n'
#o = n.tanh(); o.label='o'
#draw_dot(o)
# %%
#o.grad = 1.0
#o._backward()
# %%
#draw_dot(o)
# %%
#n._backward()
# %%
#b._backward()
# %%
#x1w1x2w2._backward()
# %%
#x2w2._backward()
# %%
#x1w1._backward()
# %%
#o.backward()
#draw_dot(o)
# %%
#a = Value(2.0, label='a')
#b = Value(4.0, label='b')
#a - b
# %%
#x1 = Value(2.0, label='x1')
#x2 = Value(0.0, label='x2')
#w1 = Value(-3.0, label='w1')
#w2 = Value(1.0, label='w2')
#b = Value(6.7, label='b')
#x1w1 = x1*w1; x1w1.label='x1w1'
#x2w2 = x2*w2; x2w2.label='x2w2'
#x1w1x2w2 = x1w1+x2w2; x1w1x2w2.label='x1w1 + x2w2'
#n = x1w1x2w2+b; n.label='n'
#--------------------------
#o = n.tanh(); o.label='o'
#e = (2*n).exp(); e.label='e'
#o = (e + -1) / (e + 1); o.label='o'
#o.backward()
#draw_dot(o)
# %%
#import torch
#x1 = torch.Tensor([2.0]).double(); x1.requires_grad = True
#x2 = torch.Tensor([0.0]).double(); x2.requires_grad = True
#w1 = torch.Tensor([-3.0]).double(); w1.requires_grad = True
#w2 = torch.Tensor([1.0]).double(); w2.requires_grad = True
#b = torch.Tensor([6.8813735870195432]).double(); b.requires_grad = True
#n = x1*w1 + x2*w2 + b
#o = torch.tanh(n)
#print(o.data.item())
#o.backward()
#print('---')
#print(x2.grad.item())
#print(w2.grad.item())
#print(x1.grad.item())
#print(w1.grad.item())
# %%
