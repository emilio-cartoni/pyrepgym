import numpy as np
import os

np.set_printoptions(formatter={'float': '{:8.4f}'.format})


# %%
def relu(x):
    return np.maximum(0, x)


class Iknn:
    def __init__(self):

        fweights = os.path.realpath(os.path.dirname(__file__)) + \
            os.sep + "weights.npy"

        self.params = np.load(fweights, allow_pickle=True)
        self.biases = [self.params[s] for s in
                       range(1, len(self.params)+1, 2)]
        self.weights = [self.params[s] for s in
                        range(0, len(self.params), 2)]
        self.input_size = self.weights[0].shape[0]
        self.n_layers = len(self.biases)

    def get_joints(self, pos):
        assert len(pos) == self.input_size
        u = pos
        for l in range(self.n_layers-1):
            u = relu(np.dot(u, self.weights[l]) + self.biases[l])
        out = np.tanh(np.dot(u, self.weights[l+1]) + self.biases[l+1])
        joints = np.zeros(7)
        joints[[0, 1, 3, 5]] = out
        return joints*np.pi

# %%


if __name__ == "__main__":

    n = 10

    inp = np.random.uniform(-1, 1, [n, 3])

    ik = Iknn()
    for x in range(n):
        print(ik.get_joints(inp[x]))
