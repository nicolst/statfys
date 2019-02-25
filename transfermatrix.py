import numpy as np
import permutations
import matplotlib.pyplot as plt
import threading


class Spinerino(threading.Thread):
    def __init__(self, P, beta):
        self.beta = beta
        self.P = P
        super(Spinerino, self).__init__()

    def compute_eigenvalues(self):
        self.eigvalues = np.linalg.eigvals(self.P ** self.beta)

    def run(self):
        self.compute_eigenvalues()


def generate_P(sigma, Bb, J):
    sigma_len = len(sigma)
    P = np.zeros((sigma_len, sigma_len))

    for l in range(sigma_len):
        for m in range(sigma_len):
            A = J[0] * np.sum(sigma[l] * sigma[m])
            B = J[1] * np.sum(sigma[l] * np.roll(sigma[l], -1))
            C = Bb / 2 * np.sum(sigma[l] + sigma[m])
            P[l, m] = np.exp(A + B + C)

    return P


betavals = np.linspace(0.005, 1.0, 100, endpoint=True)

B, J = 0.5, (1.2, 1.2)
n = 8

sigma = permutations.generate_sigma(n)

P = generate_P(sigma, B, J)

spinerinos = [Spinerino(P, beta) for beta in betavals]

for s in spinerinos:
    s.start()
print("All threads started.")
for s in spinerinos:
    s.join()
    print("Beta {} done.".format(s.beta))

eigvals = [s.eigvalues for s in spinerinos]

plt.semilogy(betavals, eigvals)
plt.show()
