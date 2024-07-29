#%%
from engine import Value
import random 

class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        # w * x + b
        return 0.0
    
x = [2.0, 3.0]
n = Neuron(2)
# %%
