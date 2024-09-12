import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

mu = 4 * np.pi**2

# Función para calcular la distancia r
def r(x, y, z):
    return np.sqrt(x**2 + y**2 + z**2)

# Definición del sistema de ecuaciones
def sistema(t, Y):
    x, y, z, vx, vy, vz = Y
    dxdt = vx
    dydt = vy
    dzdt = vz
    dvxdt = -mu * x / r(x, y, z)**3
    dvydt = -mu * y / r(x, y, z)**3
    dvzdt = -mu * z / r(x, y, z)**3
    return np.array([dxdt, dydt, dzdt, dvxdt, dvydt, dvzdt])

# Condiciones iniciales
y0 = np.array([0.325514, -0.459460, 0.166229, -9.096111, -6.916686, -1.305721])
t0 = 0  # Tiempo inicial
tf = 76  # Tiempo final en años (aproximadamente un período orbital)
h = 0.01  # Paso de tiempo

# Implementación del método de Runge-Kutta de cuarto orden
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

# Resolver el sistema
t, y = runge_kutta_4_system(sistema, y0, t0, tf, h)

# Graficar las proyecciones de la trayectoria en 2D
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.plot(y[:, 0], y[:, 1])
plt.xlabel("x (UA)")
plt.ylabel("y (UA)")
plt.title("Proyección xy")

plt.subplot(1, 3, 2)
plt.plot(y[:, 0], y[:, 2])
plt.xlabel("x (UA)")
plt.ylabel("z (UA)")
plt.title("Proyección xz")

plt.subplot(1, 3, 3)
plt.plot(y[:, 1], y[:, 2])
plt.xlabel("y (UA)")
plt.ylabel("z (UA)")
plt.title("Proyección yz")

plt.tight_layout()
plt.show()

# Cálculo de r(t) en función del tiempo
r_t = np.sqrt(y[:, 0]**2 + y[:, 1]**2 + y[:, 2]**2)

# Graficar t contra r(t)
plt.figure(figsize=(8, 6))
plt.plot(t, r_t)
plt.xlabel("Tiempo (años)")
plt.ylabel("Distancia r(t) (UA)")
plt.title("Gráfico de t contra r(t)")
plt.grid()
plt.show()

# Intentar encontrar picos con ajustes
peaks, _ = find_peaks(r_t, height=30, distance=50)  # Ajustar según sea necesario

# Si no se encuentran picos, se puede proceder manualmente
if len(peaks) > 1:
    periodos = np.diff(t[peaks])
    periodo_medio = periodos.mean()
else:
    periodo_medio = "No se encontraron picos suficientes para estimar el período."

print(f"Período estimado del cometa: {periodo_medio} años")

# Visualización de los picos encontrados
plt.figure(figsize=(8, 6))
plt.plot(t, r_t)
plt.plot(t[peaks], r_t[peaks], "x")  # Marca los picos
plt.xlabel("Tiempo (años)")
plt.ylabel("Distancia r(t) (UA)")
plt.title("Gráfico de t contra r(t) con picos marcados")
plt.grid()
plt.show()

# Aumentar el tiempo final para intentar capturar más ciclos
tf_extended = 200  # Probar con 200 años

# Resolver el sistema con el tiempo extendido
t_extended, y_extended = runge_kutta_4_system(sistema, y0, t0, tf_extended, h)

# Calcular r(t) extendido
r_t_extended = np.sqrt(y_extended[:, 0]**2 + y_extended[:, 1]**2 + y_extended[:, 2]**2)

# Graficar t contra r(t) extendido
plt.figure(figsize=(8, 6))
plt.plot(t_extended, r_t_extended)
plt.xlabel("Tiempo (años)")
plt.ylabel("Distancia r(t) (UA)")
plt.title("Gráfico de t contra r(t) con intervalo extendido")
plt.grid()
plt.show()

# Intentar encontrar picos en el intervalo extendido
peaks_extended, _ = find_peaks(r_t_extended, height=30, distance=50)

# Estimar el período si se encuentran suficientes picos
if len(peaks_extended) > 1:
    periodos_extended = np.diff(t_extended[peaks_extended])
    periodo_medio_extended = periodos_extended.mean()
else:
    periodo_medio_extended = "No se encontraron picos suficientes para estimar el período."

print(f"Período estimado del cometa (extendido): {periodo_medio_extended} años")

# Visualización de los picos encontrados en el intervalo extendido
plt.figure(figsize=(8, 6))
plt.plot(t_extended, r_t_extended)
plt.plot(t_extended[peaks_extended], r_t_extended[peaks_extended], "x")  # Marca los picos
plt.xlabel("Tiempo (años)")
plt.ylabel("Distancia r(t) (UA)")
plt.title("Gráfico de t contra r(t) con picos marcados (intervalo extendido)")
plt.grid()
plt.show()

# Estimación para 2086 (100 años después del último perihelio)
tf_100 = 100  # tiempo en años desde 1986

t_100, y_100 = runge_kutta_4_system(sistema, y0, t0, tf_100, h)

posicion_100 = y_100[-1, :3]
velocidad_100 = y_100[-1, 3:]

print(f"Posición del cometa en 2086: {posicion_100}")
print(f"Velocidad del cometa en 2086: {velocidad_100}")

# Estimación para 2186 (200 años después del último perihelio)
tf_200 = 200  # tiempo en años desde 1986

t_200, y_200 = runge_kutta_4_system(sistema, y0, t0, tf_200, h)

posicion_200 = y_200[-1, :3]
velocidad_200 = y_200[-1, 3:]

print(f"Posición del cometa en 2186: {posicion_200}")
print(f"Velocidad del cometa en 2186: {velocidad_200}")
