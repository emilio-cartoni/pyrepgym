import numpy as np

np.set_printoptions(formatter={'float': '{:8.4f}'.format})


# %%
def relu(x):
    return np.maximum(0, x)


class Iknn:
    def __init__(self):
        self.params = np.load("weights.npy", allow_pickle=True)
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

    data = np.loadtxt("data_filtered")
    
    np.random.shuffle(data)
    ik = Iknn()
    for d in data:
        print(ik.get_joints(d[7:10]))
        print(d[:7])
        print()
