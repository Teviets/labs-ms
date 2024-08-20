import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def edo(y, x):
    return (x - 3*y - 3*(x**2 - y**2) + 3*x*y) / (2*x - y + 3*(x**2 - y**2) + 2*x*y)

# Crear la malla para el campo de direcciones
x = np.linspace(-5, 5, 20)
y = np.linspace(-5, 5, 20)
X, Y = np.meshgrid(x, y)

# Calcular las pendientes
dY = edo(Y, X)
dX = np.ones(X.shape)

# Normalizar las flechas
norm = np.sqrt(dX**2 + dY**2)
dX = dX / norm
dY = dY / norm

# Graficar el campo de direcciones
plt.figure(figsize=(12, 8))
plt.quiver(X, Y, dX, dY, color='b', alpha=0.5)

# Resolver la EDO para diferentes condiciones iniciales
condiciones_iniciales = [2, -2, 1, -3]
x_span = np.linspace(0, 5, 100)

for y0 in condiciones_iniciales:
    solucion = odeint(edo, y0, x_span)
    plt.plot(x_span, solucion, label=f'y({x_span[0]:.1f})={y0}')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Campo de direcciones y soluciones de la EDO')
plt.legend()
plt.grid(True)
plt.axhline(y=0, color='k', linestyle='--')
plt.axvline(x=0, color='k', linestyle='--')
plt.show()