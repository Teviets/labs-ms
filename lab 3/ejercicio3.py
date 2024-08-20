import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Definir el sistema de ecuaciones diferenciales
def system(populations):
    x, y = populations
    dxdt = 0.5 * x - 0.001 * x**2 - x * y
    dydt = -0.2 * y + 0.1 * x * y
    return np.array([dxdt, dydt])

# Implementar el método de Runge-Kutta de cuarto orden
def runge_kutta_4th(system, initial_conditions, t):
    n = len(t)
    populations = np.zeros((n, len(initial_conditions)))
    populations[0] = initial_conditions
    
    for i in range(1, n):
        h = t[i] - t[i - 1]
        k1 = system(populations[i - 1])
        k2 = system(populations[i - 1] + 0.5 * h * k1)
        k3 = system(populations[i - 1] + 0.5 * h * k2)
        k4 = system(populations[i - 1] + h * k3)
        
        populations[i] = populations[i - 1] + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    
    return populations

# Encontrar puntos de equilibrio
def equilibrium_equations(populations):
    x, y = populations
    dxdt = 0.5 * x - 0.001 * x**2 - x * y
    dydt = -0.2 * y + 0.1 * x * y
    return [dxdt, dydt]

# Graficar el campo vectorial
def plot_phase_plane():
    x_values = np.linspace(0, 100, 20)
    y_values = np.linspace(0, 100, 20)
    X, Y = np.meshgrid(x_values, y_values)

    U = np.zeros(X.shape)
    V = np.zeros(Y.shape)

    for i in range(len(x_values)):
        for j in range(len(y_values)):
            dxdt, dydt = system([X[i, j], Y[i, j]])
            U[i, j] = dxdt
            V[i, j] = dydt

    plt.quiver(X, Y, U, V, color='b')

# Graficar individualmente cada ecuación
def plot_individual_equations(t, x_sol, y_sol):
    plt.figure(figsize=(10, 6))
    plt.plot(t, x_sol, 'r', label='Población de la especie x')
    plt.xlabel('Tiempo (meses)')
    plt.ylabel('Población de la especie x')
    plt.title('Evolución de la población de la especie x en el tiempo')
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(t, y_sol, 'b', label='Población de la especie y')
    plt.xlabel('Tiempo (meses)')
    plt.ylabel('Población de la especie y')
    plt.title('Evolución de la población de la especie y en el tiempo')
    plt.grid(True)
    plt.legend()
    plt.show()

def main():
    # b) Encontrar los puntos de equilibrio
    initial_guesses = [[0, 0], [100, 0], [0, 100], [50, 50]]
    equilibrium_points = []

    for guess in initial_guesses:
        equilibrium = fsolve(equilibrium_equations, guess)
        equilibrium_points.append(equilibrium)

    # Imprimir los puntos de equilibrio
    print("Puntos de equilibrio:")
    for point in equilibrium_points:
        print(f"x = {point[0]:.3f}, y = {point[1]:.3f}")

    # c) Resolver el sistema de EDO usando Runge-Kutta de cuarto orden
    initial_conditions = [10, 10]
    t = np.linspace(0, 60, 500)  # 60 meses (5 años)

    solution = runge_kutta_4th(system, initial_conditions, t)
    x_sol, y_sol = solution[:, 0], solution[:, 1]

    # Imprimir las poblaciones finales después de 5 años
    print(f"\nPoblación después de 5 años (60 meses):")
    print(f"x(60) = {x_sol[-1]:.3f}, y(60) = {y_sol[-1]:.3f}")

    # Primera gráfica: Solo el campo vectorial
    plt.figure(figsize=(10, 6))
    plot_phase_plane()
    plt.xlabel('Población de la especie x')
    plt.ylabel('Población de la especie y')
    plt.title('Plano de fase (Campo vectorial)')
    plt.grid(True)
    plt.xlim([0, 100])
    plt.ylim([0, 100])
    plt.show()

    # Segunda gráfica: Solo la trayectoria de la solución con puntos de equilibrio
    plt.figure(figsize=(10, 6))
    plt.plot(x_sol, y_sol, 'r', label='Trayectoria de la solución')
    plt.scatter(*zip(*equilibrium_points), color='k', zorder=5, label='Puntos de equilibrio')
    plt.xlabel('Población de la especie x')
    plt.ylabel('Población de la especie y')
    plt.title('Trayectoria de la solución con puntos de equilibrio')
    plt.legend()
    plt.grid(True)
    plt.xlim([0, 100])
    plt.ylim([0, 100])
    plt.show()

    # Graficar individualmente las ecuaciones
    plot_individual_equations(t, x_sol, y_sol)

    print("\n\nEXPLICACIÓN DE LOS RESULTADOS:")
    print("""
            El sistema tiene un comportamiento de extinción para las dos especies. Pero existe un punto en el que las dos especies 
            pueden coexistir (x = 2, y = 0.498). Este punto es de equilibrio e inestable. En consecuencia, las trayectorias del sistema 
            tienden a moverse hacia el origen, donde ambas poblaciones se extinguen.
    """)

if __name__ == "__main__":
    main()
