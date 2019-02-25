import transfermatrix
import permutations
import numpy as np
import matplotlib.pyplot as plt

k = 1.38e-23
betas = np.linspace(0.005, 1, 100, endpoint=True)

def calculate_specific_heat_per_nnk(n):
    sigma = permutations.generate_sigma(n)
    P = transfermatrix.generate_P(sigma, 0.5, (1.2, 1.2))

    spinners = [transfermatrix.Spinerino(P, b) for b in betas]

    for s in spinners:
        s.start()
    for s in spinners:
        s.join()

    logeigs = [np.log10(np.max(s.eigvalues)) for s in spinners]

    first_numerator = np.diff(logeigs)
    first_denominator = np.diff(betas)

    first_derivative = first_numerator / first_denominator

    second_numerator = np.diff(first_derivative)

    second_derivative = second_numerator / first_denominator[:-1]

    heat_per_unit_for_beta = second_derivative * betas[:-2]**2 / n
    heat_etc_for_T = np.flipud(heat_per_unit_for_beta)
    return heat_etc_for_T

T = np.flipud(np.array([1/(k*b) for b in betas]))
test = calculate_specific_heat_per_nnk(6)
plt.plot(T[:-2], test)
plt.title(r"$n={0}, B={1}, J_\perp={2}, J_\| = {3}$".format(6, 0.5, 1.2, 1.2))
plt.xlabel(r"$T$", size=14)
plt.ylabel(r"$C_B(T)/Nnk$", size=14)
plt.show()
