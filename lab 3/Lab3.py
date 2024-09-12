"""
ModSim Laboratorio 3
"""


# Importar librerías
import numpy as np
import matplotlib.pyplot as plt


# Algotirmo de Runge-Kutta de cuarto orden para una EDO:
def runge_kutta_4(f, y0, t0, tf, h):
    n = int((tf - t0) / h)
    t = np.linspace(t0, tf, n + 1)
    y = np.zeros(n + 1)
    y[0] = y0

    for i in range(n):
        k1 = h * f(t[i], y[i])
        k2 = h * f(t[i] + h / 2, y[i] + k1 / 2)
        k3 = h * f(t[i] + h / 2, y[i] + k2 / 2)
        k4 = h * f(t[i] + h, y[i] + k3)
        y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return t, y


# Algoritmo de Runge-Kutta de cuarto orden para un sistema de EDOs:
def runge_kutta_4_system(f, y0, t0, tf, h):
    n = int((tf - t0) / h)
    t = np.linspace(t0, tf, n + 1)
    y = np.zeros((n + 1, len(y0)))
    y[0] = y0

    for i in range(n):
        k1 = h * f(t[i], y[i])
        k2 = h * f(t[i] + h / 2, y[i] + k1 / 2)
        k3 = h * f(t[i] + h / 2, y[i] + k2 / 2)
        k4 = h * f(t[i] + h, y[i] + k3)
        y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return t, y

"""
---------------------------------------------------------------------------------------------
Ejercicio 2
---------------------------------------------------------------------------------------------
"""

# a) Graficar el campo vectorial de la ecuación diferencial

# Definimos las EDOs
def dx_dt(x, y):
    return 0.2 * x - 0.005 * x * y

def dy_dt(x, y):
    return -0.5 * y + 0.01 * x * y

x = np.linspace(0, 150, 20)
y = np.linspace(0, 150, 20)
X, Y = np.meshgrid(x, y)

# Calcular los componentes del campo vectorial
U = dx_dt(X, Y)
V = dy_dt(X, Y)

plt.figure(figsize=(8, 8))
plt.quiver(X, Y, U, V, color="blue")
plt.xlabel("x(t)")
plt.ylabel("y(t)")
plt.title("Campo vectorial del sistema de EDOs")
plt.xlim([0, 150])
plt.ylim([0, 150])
plt.grid()
plt.show()


# b) Encontrar los puntos de Equilibrio y clasificarlos
from sympy import symbols, Eq, solve
from sympy import Matrix

# Definir las variables
x, y = symbols('x y')

# Definir el sistema de ecuaciones
eq1 = Eq(0.2 * x - 0.005 * x * y, 0)
eq2 = Eq(-0.5 * y + 0.01 * x * y, 0)

# Resolver el sistema
equilibria = solve((eq1, eq2), (x, y))
equilibria = [(float(e[0]), float(e[1])) for e in equilibria]
print(f"Puntos de equilibrio en el primer cuadrante: {equilibria}")

# Definir las funciones
f1 = 0.2 * x - 0.005 * x * y
f2 = -0.5 * y + 0.01 * x * y

# Matriz Jacobiana
J = Matrix([[f1.diff(x), f1.diff(y)],
            [f2.diff(x), f2.diff(y)]])

# Evaluar el Jacobiano en los puntos de equilibrio
for eq_point in equilibria:
    J_eval = J.subs({x: eq_point[0], y: eq_point[1]})
    eigenvals = J_eval.eigenvals()
    print(f"En el punto {eq_point}: valores propios = {eigenvals}")


# c) Resolver el sistema de EDOs con Runge-Kutta de cuarto orden
def sistema(t, Y):
    x, y = Y
    dxdt = 0.2 * x - 0.005 * x * y
    dydt = -0.5 * y + 0.01 * x * y
    return np.array([dxdt, dydt])

y0 = np.array([70, 30])  # Condiciones iniciales
t0 = 0
tf = 60  # 5 años son 60 meses
h = 0.1

t, y = runge_kutta_4_system(sistema, y0, t0, tf, h)

# Graficar la solución
plt.plot(t, y[:, 0], label="x(t) - Población de x")
plt.plot(t, y[:, 1], label="y(t) - Población de y")
plt.xlabel("Tiempo (meses)")
plt.ylabel("Población")
plt.title("Solución del sistema usando Runge-Kutta de Orden 4")
plt.legend()
plt.grid()
plt.show()

# Estimar las poblaciones después de 5 años
print(f"Población de x después de 5 años: {y[-1, 0]:.2f}")
print(f"Población de y después de 5 años: {y[-1, 1]:.2f}")

# Estimar el período o ciclo de repetición
from scipy.signal import find_peaks

peaks_x, _ = find_peaks(y[:, 0])
period_x = np.diff(t[peaks_x]).mean()
print(f"Período estimado del ciclo de x: {period_x:.2f} meses")


# d) Resolver el sistema de EDOs con Runge-Kutta de cuarto orden para diferentes condiciones iniciales
# Nuevas condiciones iniciales
y0_new = np.array([100, 10])  # x(0) = 100, y(0) = 10

t_new, y_new = runge_kutta_4_system(sistema, y0_new, t0, tf, h)

# Graficar la solución
plt.plot(t_new, y_new[:, 0], label="x(t) - Población de x")
plt.plot(t_new, y_new[:, 1], label="y(t) - Población de y")
plt.xlabel("Tiempo (meses)")
plt.ylabel("Población")
plt.title("Solución del sistema usando Runge-Kutta de Orden 4")
plt.legend()
plt.grid()
plt.show()

# Estimar las poblaciones después de 5 años
print(f"Población de x después de 5 años: {y_new[-1, 0]:.2f}")
print(f"Población de y después de 5 años: {y_new[-1, 1]:.2f}")

# Estimar el período o ciclo de repetición
peaks_x_new, _ = find_peaks(y_new[:, 0])
period_x_new = np.diff(t_new[peaks_x_new]).mean()
print(f"Período estimado del ciclo de x: {period_x_new:.2f} meses")


# e) Graficar ambas soluciones 

plt.figure(figsize=(8, 8))

# Graficar el campo vectorial
plt.quiver(X, Y, U, V, color="blue", alpha=0.5)

# Graficar la primera trayectoria
plt.plot(y[:, 0], y[:, 1], label="Trayectoria 1: x(0)=70, y(0)=30", color="red")
plt.scatter([y[0, 0]], [y[0, 1]], color="red", label="Inicial 1")
plt.scatter([y[-1, 0]], [y[-1, 1]], color="darkred", label="Final 1")

# Graficar la segunda trayectoria
plt.plot(y_new[:, 0], y_new[:, 1], label="Trayectoria 2: x(0)=100, y(0)=10", color="green")
plt.scatter([y_new[0, 0]], [y_new[0, 1]], color="green", label="Inicial 2")
plt.scatter([y_new[-1, 0]], [y_new[-1, 1]], color="darkgreen", label="Final 2")

plt.xlabel("x(t)")
plt.ylabel("y(t)")
plt.title("Trayectorias en el plano de fase")
plt.xlim([0, 150])
plt.ylim([0, 150])
plt.legend()
plt.grid()
plt.show()


# f) Descripcion del sistema: 
"""

"""