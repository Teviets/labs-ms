import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def initialize_grid(M, N, infected_positions):
    grid = np.zeros((M, N), dtype=int)
    for (i, j) in infected_positions:
        grid[i, j] = 1
    return grid

def simulate_sir(grid, M, N, T, rad, beta, gamma):
    history = []
    S_count, I_count, R_count = [], [], []

    for t in range(T + 1):
        new_grid = grid.copy()
        history.append(new_grid.copy())

        S, I, R = np.sum(grid == 0), np.sum(grid == 1), np.sum(grid == 2)
        S_count.append(S)
        I_count.append(I)
        R_count.append(R)

        for i in range(M):
            for j in range(N):
                if grid[i, j] == 0:  # susceptible
                    infected_neighbors = 0
                    for x in range(max(0, i - rad), min(M, i + rad + 1)):
                        for y in range(max(0, j - rad), min(N, j + rad + 1)):
                            if grid[x, y] == 1:
                                infected_neighbors += 1
                    if np.random.rand() < 1 - (1 - beta) ** infected_neighbors:
                        new_grid[i, j] = 1
                elif grid[i, j] == 1:  # infected
                    if np.random.rand() < gamma:
                        new_grid[i, j] = 2

        grid = new_grid

    return history, S_count, I_count, R_count

def average_grids(histories):
    Nexp, T, M, N = len(histories), len(histories[0]), len(histories[0][0]), len(histories[0][0][0])
    avg_grid = np.zeros((T, M, N), dtype=float)

    for exp in histories:
        for t in range(T):
            avg_grid[t] += exp[t]
    avg_grid /= Nexp

    return avg_grid

def plot_snapshots(avg_grid, T, M, N, num_snapshots=10):
    # Seleccionar 10 snapshots de forma equidistante
    T_snapshots = np.linspace(0, T-1, num_snapshots, dtype=int)
    
    fig, axes = plt.subplots(1, len(T_snapshots), figsize=(15, 5))
    for i, t in enumerate(T_snapshots):
        axes[i].imshow(avg_grid[t], cmap='viridis', vmin=0, vmax=2)
        axes[i].set_title(f'Time = {t}')
    plt.show()

def create_animation(avg_grid, M, N):
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(avg_grid[0], cmap='viridis', vmin=0, vmax=2)

    def update(frame):
        im.set_array(avg_grid[frame])
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=range(len(avg_grid)), blit=True)

    # Guardar la animación como un archivo gif
    ani.save('sir_simulation.gif', writer='pillow', fps=5)

    plt.show()

# Parámetros de la simulación
M, N = 50, 50
T = 50
rad = 1
beta = 0.1
gamma = 0.25
Nexp = 10
infected_positions = [(10, 10), (15, 15), (20, 20)]  # Posiciones predefinidas de infecciones iniciales

# Simulación
histories = []
for _ in range(Nexp):
    grid = initialize_grid(M, N, infected_positions)
    history, S_count, I_count, R_count = simulate_sir(grid, M, N, T, rad, beta, gamma)
    histories.append(history)

# Calcular el grid promedio
avg_grid = average_grids(histories)

# Crear snapshots (opcional, para visualizar 10 momentos específicos)
plot_snapshots(avg_grid, T, M, N, num_snapshots=10)

# Crear y guardar la animación
create_animation(avg_grid, M, N)
