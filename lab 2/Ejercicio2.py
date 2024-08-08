import numpy as np

# Definir la función F(x, y) y su Jacobiano
def F(x):
    x1, x2 = x
    return np.array([
        3 * x1**2 - x2**2,
        3 * x1 * x2**2 - x1**3 - 1
    ])

def JF(x):
    x1, x2 = x
    return np.array([
        [6 * x1, -2 * x2],
        [3 * x2**2 - 3 * x1**2, 6 * x1 * x2]
    ])

# Definir la función G(x, y, z) y su Jacobiano
def G(x):
    x1, x2, x3 = x
    return np.array([
        12 * x1 - 3 * x2**2 - 4 * x3 - 7.17,
        x1 + 10 * x2 - x3 - 11.54,
        x2**3 - 7 * x3**3 - 7.631
    ])

def JG(x):
    x1, x2, x3 = x
    return np.array([
        [12, -6 * x2, -4],
        [1, 10, -1],
        [0, 3 * x2**2, -21 * x3**2]
    ])

# Implementación del método de Newton para sistemas de ecuaciones
def newton_multidim(F, JF, x0, tol=1e-7, max_iter=100):
    x = x0
    for i in range(max_iter):
        J = JF(x)
        F_val = F(x)
        delta_x = np.linalg.solve(J, -F_val)
        x = x + delta_x
        
        print(f"Iteración {i+1}: x = {x}, F(x) = {F_val}")
        
        if np.linalg.norm(delta_x) < tol:
            print(f"Convergencia en {i+1} iteraciones.")
            return x
    print("No llegó a convergir")
    return x

# Programa principal
def main():
    # Resolver el sistema para F(x, y)
    print("Resolviendo F(x, y):")
    x0_F = np.array([0.1, 0.1])  # Valores iniciales
    sol_F = newton_multidim(F, JF, x0_F)
    print(f"Solución para F(x, y): x* = {sol_F}\n")

    # Resolver el sistema para G(x, y, z)
    print("Resolviendo G(x, y, z):")
    x0_G = np.array([0.1, 0.1, 0.1])  # Valores iniciales
    sol_G = newton_multidim(G, JG, x0_G)
    print(f"Solución para G(x, y, z): x* = {sol_G}")

if __name__ == "__main__":
    main()
