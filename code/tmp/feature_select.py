# Mahmood Amintoosi
from itertools import combinations, chain
import numpy as np
features = np.array([[0, 1, 1],
                     [1, 0, 0],
                     [1, 1, 1]])

n = features.shape[1]

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

# get all combinations, we will use this as indices for the columns later
indices = list(powerset(range(n)))

# remove the empty subset
indices.pop(0)

print(indices)

data = []

for i in indices:

    print()
    print(i)
    # _ = features[:, i]
    # print(_)

    # x = np.prod(_, axis=1)
    # print(x)
    # data.append(x)

# print(np.column_stack(data))