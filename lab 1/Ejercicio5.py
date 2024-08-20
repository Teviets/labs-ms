import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Definir la EDO
def dP_dt(P, t):
    return 0.0004 * P**2 - 0.06 * P

# Tiempo
t = np.linspace(0, 50, 500)

# Condiciones iniciales
P0_200 = 200
P0_100 = 100

# Resolver la EDO
P_200 = odeint(dP_dt, P0_200, t)
P_100 = odeint(dP_dt, P0_100, t)

# Graficar las soluciones
plt.figure(figsize=(12, 8))
plt.plot(t, P_200, label='P(0) = 200', color='r')
plt.plot(t, P_100, label='P(0) = 100', color='b')
plt.axhline(y=150, color='g', linestyle='--', label='Equilibrio P = 150')
plt.axhline(y=0, color='k', linestyle='--')
plt.xlabel('Tiempo (semanas)')
plt.ylabel('Población (individuos)')
plt.title('Curvas de Solución de la EDO')
plt.legend()
plt.grid(True)
plt.show()
