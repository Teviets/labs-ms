import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Definir la función f(x, y) para la ecuación diferencial dada
def f(x, y):
    return (4*x - 3*y + 7*x*y) / (4*x + 2*y + 2*x**2 - 3*y**2)

# Derivadas parciales numéricas para encontrar puntos de equilibrio
def df_dx(x, y):
    h = 1e-6
    return (f(x+h, y) - f(x, y)) / h

def df_dy(x, y):
    h = 1e-6
    return (f(x, y+h) - f(x, y)) / h

# Resolver el sistema numéricamente usando varios estimados iniciales
def sistema(p):
    x, y = p
    return [df_dx(x, y), df_dy(x, y)]

# Probar con diferentes puntos iniciales para encontrar puntos de equilibrio
puntos_iniciales = [[1, 1], [-1, -1], [1, -1], [-1, 1]]
soluciones = []

for p_inicial in puntos_iniciales:
    sol = fsolve(sistema, p_inicial)
    if sol.tolist() not in soluciones:  # Evitar duplicados
        soluciones.append(sol.tolist())

print("\nPuntos de equilibrio encontrados numéricamente:", soluciones)

# Visualizar el campo vectorial y los puntos de equilibrio
X, Y = np.meshgrid(np.linspace(-10, 10, 400), np.linspace(-10, 10, 400))
U = np.ones_like(X)
V = f(X, Y)

plt.streamplot(X, Y, U, V, color='blue')
plt.title('Campo vectorial del sistema')
for sol in soluciones:
    plt.plot(sol[0], sol[1], 'ro')  # Marcar los puntos de equilibrio en rojo
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.show()
