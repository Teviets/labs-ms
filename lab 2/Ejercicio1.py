import numpy as np

def F(x):
    x1, x2, x3 = x
    return np.array([
        3 * x1 - np.cos(x2 * x3) - 0.5,
        x1**2 - 81 * (x2 + 0.1)**2 + np.sin(x3) + 1.06,
        np.exp(-x1 * x2) + 20 * x3 + (10 * np.pi - 3) / 3
    ])

def JF(x):
    x1, x2, x3 = x
    return np.array([
        [3, x3 * np.sin(x2 * x3), x2 * np.sin(x2 * x3)],
        [2 * x1, -162 * (x2 + 0.1), np.cos(x3)],
        [-x2 * np.exp(-x1 * x2), -x1 * np.exp(-x1 * x2), 20]
    ])

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
    print("No llego a convergir")
    return x

# Parámetros iniciales
x0 = np.array([0.1, 0.1, 0.1])

# Encontrar la solución
sol = newton_multidim(F, JF, x0)

print(f"Solución encontrada: x* = {sol}")
