import numpy as np

# Generate the spin configurations recursively
def generate_recursive(n):
    if n == 1:
        return [[1], [-1]]

    current = generate_recursive(n - 1)
    permutations = []

    for p in current:
        permutations.append(p + [1])
        permutations.append(p + [-1])

    return permutations

# "Helper" function to avoid too much shitty numpy code
def generate_sigma(n):
    arr = generate_recursive(n)
    return np.array(arr)

print(generate_sigma(5))