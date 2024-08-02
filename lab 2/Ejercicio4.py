import numpy as np
import matplotlib.pyplot as plt

# Funciones para las EDOs dadas
def f1(x, y):
    return 2 * np.exp(x / 5) - 5 * x - 25

def f2(t, u):
    return -np.sin(t)

# Derivada de f1 para el método de Taylor de orden 2
def f1_prime(x, y):
    return (2 / 5) * np.exp(x / 5) - 5

# Métodos Numéricos
def euler(f, y0, t0, tf, h):
    t = np.arange(t0, tf, h)
    y = np.zeros(len(t))
    y[0] = y0
    for i in range(1, len(t)):
        y[i] = y[i-1] + h * f(t[i-1], y[i-1])
    return t, y

def heun(f, y0, t0, tf, h):
    t = np.arange(t0, tf, h)
    y = np.zeros(len(t))
    y[0] = y0
    for i in range(1, len(t)):
        k1 = h * f(t[i-1], y[i-1])
        k2 = h * f(t[i-1] + h, y[i-1] + k1)
        y[i] = y[i-1] + 0.5 * (k1 + k2)
    return t, y

def taylor2(f, f_prime, y0, t0, tf, h):
    t = np.arange(t0, tf, h)
    y = np.zeros(len(t))
    y[0] = y0
    for i in range(1, len(t)):
        y[i] = y[i-1] + h * f(t[i-1], y[i-1]) + (h**2 / 2) * f_prime(t[i-1], y[i-1])
    return t, y

# Soluciones exactas
def exact_solution1(x):
    return -3 + 10 * np.exp(x / 5) - 5 * x**2 - 25 * x

def exact_solution2(t):
    return 1 + np.cos(t)

# Parámetros
y0_a = -3
x0_a = 0
xf_a = 5
h_a = 0.1

u0_b = 1
t0_b = 0
tf_b = 6 * np.pi
h_b = 0.1

# Resolver EDOs
t_euler_a, y_euler_a = euler(f1, y0_a, x0_a, xf_a, h_a)
t_heun_a, y_heun_a = heun(f1, y0_a, x0_a, xf_a, h_a)
t_taylor_a, y_taylor_a = taylor2(f1, f1_prime, y0_a, x0_a, xf_a, h_a)

t_euler_b, u_euler_b = euler(f2, u0_b, t0_b, tf_b, h_b)
t_heun_b, u_heun_b = heun(f2, u0_b, t0_b, tf_b, h_b)
t_taylor_b, u_taylor_b = taylor2(f2, lambda t, u: -np.cos(t), u0_b, t0_b, tf_b, h_b)

# Gráfica de la EDO a)
plt.figure(figsize=(12, 6))
plt.plot(t_euler_a, y_euler_a, label="Euler")
plt.plot(t_heun_a, y_heun_a, label="Heun")
plt.plot(t_taylor_a, y_taylor_a, label="Taylor de orden 2")
plt.plot(t_euler_a, exact_solution1(t_euler_a), 'k--', label="Solución exacta")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Solución numérica vs exacta para la EDO a)")
plt.show()

# Gráfica de la EDO b)
plt.figure(figsize=(12, 6))
plt.plot(t_euler_b, u_euler_b, label="Euler")
plt.plot(t_heun_b, u_heun_b, label="Heun")
plt.plot(t_taylor_b, u_taylor_b, label="Taylor de orden 2")
plt.plot(t_euler_b, exact_solution2(t_euler_b), 'k--', label="Solución exacta")
plt.xlabel("t")
plt.ylabel("u")
plt.legend()
plt.title("Solución numérica vs exacta para la EDO b)")
plt.show()

# Campo vectorial de la EDO a)
X, Y = np.meshgrid(np.linspace(x0_a, xf_a, 20), np.linspace(min(y_euler_a), max(y_euler_a), 20))
U = np.ones_like(X)
V = f1(X, Y)

plt.figure(figsize=(12, 6))
plt.quiver(X, Y, U, V, angles='xy')
plt.plot(t_euler_a, y_euler_a, 'r', label="Solución Euler")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Campo vectorial y solución numérica para la EDO a)")
plt.show()

# Campo vectorial de la EDO b)
T, U = np.meshgrid(np.linspace(t0_b, tf_b, 20), np.linspace(min(u_euler_b), max(u_euler_b), 20))
V = f2(T, U)

plt.figure(figsize=(12, 6))
plt.quiver(T, U, np.ones_like(T), V, angles='xy')
plt.plot(t_euler_b, u_euler_b, 'r', label="Solución Euler")
plt.xlabel("t")
plt.ylabel("u")
plt.legend()
plt.title("Campo vectorial y solución numérica para la EDO b)")
plt.show()
