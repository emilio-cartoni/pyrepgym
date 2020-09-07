#!/bin/bash

py2_load="
import numpy as np
import pickle

w = np.load('weights.npy', allow_pickle=True)
with open('/tmp/w', 'wb') as f:
    pickle.dump(w, f, protocol=2)
"
py3_save="
import numpy as np
import pickle

with open('/tmp/w', 'rb') as f:
    w = pickle.load(f)

np.save('weights_3', w, allow_pickle=True)
"

echo -E "$py2_load" | python3
echo -E "$py3_save" | python2

