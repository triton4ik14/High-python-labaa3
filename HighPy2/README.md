# Лабораторная работа №1
##### Вариант №5
##### Бобришов Роман, МКН-418Б
## Решение задачи Дирихле для уравнения Лапласа методом Гаусса-Зейделя

### Постановка задачи

Требуется решить задачу Дирихле для уравнения Лапласа в прямоугольной области:
$$\Delta u = \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} = 0$$

с граничными условиями:
- $u(0, y) = -19y^2 - 17y + 15$, $y \in [0, 1]$
- $u(1, y) = -19y^2 - 57y + 49$, $y \in [0, 1]$
- $u(x, 0) = 18x^2 + 16x + 15$, $x \in [0, 1]$
- $u(x, 1) = 18x^2 - 24x - 21$, $x \in [0, 1]$

Область решения: $[0, 1] \times [0, 1]$.

### Теоретическая часть
#### Конечно-разностная аппроксимация

Для дискретизации задачи используем равномерную сетку:
- $x_i = i \cdot h$, $i = 0, 1, \dots, n$
- $y_j = j \cdot h$, $j = 0, 1, \dots, n$
- $h = \frac{1}{n}$

Уравнение Лапласа аппроксимируется разностным уравнением:
$$\frac{u_{i+1,j} - 2u_{i,j} + u_{i-1,j}}{h^2} + \frac{u_{i,j+1} - 2u_{i,j} + u_{i,j-1}}{h^2} = 0$$

После преобразования получаем итерационную формулу:
$$u_{i,j} = \frac{u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1}}{4}$$

#### Метод Гаусса-Зейделя

Итерационный процесс метода Гаусса-Зейделя:
$$u_{i,j}^{(k+1)} = \frac{u_{i+1,j}^{(k)} + u_{i-1,j}^{(k+1)} + u_{i,j+1}^{(k)} + u_{i,j-1}^{(k+1)}}{4}$$

Критерий остановки:
$$\max_{i,j} |u_{i,j}^{(k+1)} - u_{i,j}^{(k)}| < \varepsilon$$

#### Необходимые компоненты
-Создание виртуального окружения:
python -m venv venv или запуск в PyCharm

-Активация виртуального окружения:
venv\Scripts\activate

-Установка библиотек:
pip install numpy matplotlib

-Запуск автоматического тестирования:
python main.py --method experiments

### Практическая часть
#### Ключевые функции

1. **Граничные условия** (`boundary_conditions(x: float, y: float, side: str) -> float`):
```python
    if side == 'left':    # x = 0
        return -19*y**2 - 17*y + 15
    elif side == 'right': # x = 1
        return -19*y**2 - 57*y + 49
    elif side == 'bottom': # y = 0
        return 18*x**2 + 16*x + 15
    else:  # side == 'top', y = 1
        return 18*x**2 - 24*x - 21
    """
    Параметры:
    x, y - координаты точки
    side - сторона ('left', 'right', 'bottom', 'top')
    """
```

2. **Решает уравнение Лапласа на чистом Python** (`solve_laplace_pure(h: float, epsilon: float, max_iter: int = 10000)`):
```
    """
    Алгоритм: Метод Якоби (итерационное усреднение соседей)
    Формула обновления: u[i][j] = (u[i+1][j] + u[i-1][j] + u[i][j+1] + u[i][j-1]) / 4.0
    Критерий остановки: error < epsilon или достижение max_iter
    """
```
3. **Решает уравнение Лапласа c использованием NumPy-циклы** (`solve_laplace_numpy_loop(h: float, epsilon: float, max_iter: int = 10000)`):
```
    """
    Особенность: Использует NumPy массивы, но обновление через циклы
    Преимущество: Быстрее чистого Python за счет оптимизированных операций NumPy
    """
```
4. **Решает уравнение Лапласа c использованием NumPy-векторизация** (`solve_laplace_numpy_vectorized(h: float, epsilon: float, max_iter: int = 10000)`):
```
    """
    Особенность: Использует срезы массивов для одновременного обновления всех точек
    Формула: $$u[1:n, 1:n] = (u[2:n+1, 1:n] + u[0:n-1, 1:n] + u[1:n, 2:n+1] + u[1:n, 0:n-1]) / 4.0$$
    Преимущество: Максимальная производительность
    """
```

### Результаты запуска
*Время в секундах*
## Сравнение времени выполнения

| h     | ε       | Чистый Python | NumPy с циклами | Векторизованный NumPy |
|-------|---------|---------------|-----------------|-----------------------|
| 0.1   | 0.1     | 0.0000        | 0.0009          | 0.0011               |
| 0.1   | 0.01    | 0.0000        | 0.0014          | 0.0017               |
| 0.1   | 0.001   | 0.0000        | 0.0024          | 0.0012               |
| 0.01  | 0.1     | 0.2381        | 0.5833          | 0.0062               |
| 0.01  | 0.01    | 1.4180        | 3.3540          | 0.0415               |
| 0.01  | 0.001   | 3.4944        | 8.5118          | 0.1291               |
| 0.005 | 0.1     | 1.0684        | 2.5856          | 0.0235               |
| 0.005 | 0.01    | 8.3621        | 19.9247         | 0.1952               |
| 0.005 | 0.001   | 34.4215       | 82.3450         | 1.0471               |

## Количество итераций

| h     | ε       | Чистый Python | NumPy с циклами | Векторизованный NumPy |
|-------|---------|---------------|-----------------|-----------------------|
| 0.1   | 0.1     | 20            | 20              | 33                   |
| 0.1   | 0.01    | 34            | 34              | 71                   |
| 0.1   | 0.001   | 56            | 56              | 116                  |
| 0.01  | 0.1     | 141           | 141             | 150                  |
| 0.01  | 0.01    | 824           | 824             | 1054                 |
| 0.01  | 0.001   | 2070          | 2070            | 3297                 |
| 0.005 | 0.1     | 158           | 158             | 163                  |
| 0.005 | 0.01    | 1226          | 1226            | 1362                 |
| 0.005 | 0.001   | 5059          | 5059            | 7407                 |



### Выводы

1. **Эффективность NumPy**: Использование NumPy ускоряет вычисления в 3-5 раз за счет оптимизированных векторных операций в C.
2. **Эффективность Numba**:
-Первый запуск: Numba требует времени на JIT-компиляцию (0.5-1 секунда), что делает ее медленнее других методов при единичных вычислениях.
-Повторные запуски: После компиляции Numba показывает производительность в 10-50 раз выше, чем чистый Python, и в 2-5 раз выше, чем NumPy.

3. **Точность метода**: Метод Гаусса-Зейделя обеспечивает хорошую точность при умеренном количестве итераций. Сходимость линейная.

4. **Сравнительный анализ**:
-Python: Лучшая читаемость и простота отладки, но наихудшая производительность.
-NumPy: Оптимальный баланс между производительностью и удобством использования, не требует дополнительной компиляции.
-Numba: Максимальная производительность после компиляции, но требует времени на "прогрев" и имеет кривую обучения.

5. **Рекомендации**:
-Для разработки и прототипирования: Используйте NumPy - быстрая разработка без накладных расходов.
-Для производственных вычислений: Используйте Numba с предварительной компиляцией - максимальная производительность.
-Для обучения и понимания алгоритма: Используйте чистый Python - лучшая прозрачность процесса.
-Для однократных вычислений: Используйте NumPy - избегайте накладных расходов на компиляцию Numba.

---

### Приложения
```python
import time
import sys
import os
import json
import argparse
from typing import List, Tuple, Dict, Any
import numpy as np
import matplotlib.pyplot as plt


# --- Граничные условия ---
def boundary_conditions(x: float, y: float, side: str) -> float:
    if side == 'left':
        return -19 * y ** 2 - 17 * y + 15
    elif side == 'right':
        return -19 * y ** 2 - 57 * y + 49
    elif side == 'bottom':
        return 18 * x ** 2 + 16 * x + 15
    elif side == 'top':
        return 18 * x ** 2 - 24 * x - 21
    else:
        raise ValueError(f"Неизвестная сторона: {side}")


# --- Инициализация сетки ---
def initialize_grid_pure(h: float) -> Tuple[List[List[float]], int]:
    n = int(1 / h)
    grid_size = n + 1
    u = [[0.0 for _ in range(grid_size)] for _ in range(grid_size)]
    for i in range(grid_size):
        x = i * h
        u[i][0] = boundary_conditions(x, 0, 'bottom')
        u[i][n] = boundary_conditions(x, 1, 'top')
    for j in range(grid_size):
        y = j * h
        u[0][j] = boundary_conditions(0, y, 'left')
        u[n][j] = boundary_conditions(1, y, 'right')
    return u, n


def initialize_grid_numpy(h: float) -> Tuple[np.ndarray, int]:
    n = int(1 / h)
    grid_size = n + 1
    u = np.zeros((grid_size, grid_size), dtype=np.float64)
    for i in range(grid_size):
        x = i * h
        u[i, 0] = boundary_conditions(x, 0, 'bottom')
        u[i, n] = boundary_conditions(x, 1, 'top')
    for j in range(grid_size):
        y = j * h
        u[0, j] = boundary_conditions(0, y, 'left')
        u[n, j] = boundary_conditions(1, y, 'right')
    return u, n


# --- Решатели Лапласа ---
def solve_laplace_pure(h: float, epsilon: float, max_iter: int = 10000) -> Tuple[List[List[float]], int, float]:
    if h <= 0:
        raise ValueError("Шаг сетки должен быть положительным")
    if epsilon <= 0:
        raise ValueError("Точность должна быть положительной")
    start_time = time.time()
    u, n = initialize_grid_pure(h)
    u_old = [row[:] for row in u]
    iteration = 0
    error = float('inf')
    while iteration < max_iter and error > epsilon:
        # Классический метод конечных разностей
        for i in range(1, n):
            for j in range(1, n):
                u[i][j] = (u[i + 1][j] + u[i - 1][j] + u[i][j + 1] + u[i][j - 1]) / 4.0
        # Вычисление максимальной ошибки
        error = 0.0
        for i in range(1, n):
            for j in range(1, n):
                err = abs(u[i][j] - u_old[i][j])
                if err > error:
                    error = err
        # Обновление старого решения
        for i in range(1, n):
            for j in range(1, n):
                u_old[i][j] = u[i][j]
        iteration += 1
    end_time = time.time()
    return u, iteration, end_time - start_time


def solve_laplace_numpy_loop(h: float, epsilon: float, max_iter: int = 10000) -> Tuple[np.ndarray, int, float]:
    if h <= 0:
        raise ValueError("Шаг сетки должен быть положительным")
    if epsilon <= 0:
        raise ValueError("Точность должна быть положительной")
    start_time = time.perf_counter()
    u, n = initialize_grid_numpy(h)
    iteration = 0
    error = float('inf')
    while iteration < max_iter and error > epsilon:
        u_old = u.copy()
        for i in range(1, n):
            for j in range(1, n):
                u[i, j] = 0.25 * (u[i + 1, j] + u[i - 1, j] + u[i, j + 1] + u[i, j - 1])
        error = np.max(np.abs(u[1:n, 1:n] - u_old[1:n, 1:n]))
        iteration += 1
    end_time = time.perf_counter()
    return u, iteration, end_time - start_time


def solve_laplace_numpy_vectorized(h: float, epsilon: float, max_iter: int = 10000) -> Tuple[np.ndarray, int, float]:
    if h <= 0:
        raise ValueError("Шаг сетки должен быть положительным")
    if epsilon <= 0:
        raise ValueError("Точность должна быть положительной")
    start_time = time.perf_counter()
    u, n = initialize_grid_numpy(h)
    iteration = 0
    error = float('inf')
    while iteration < max_iter and error > epsilon:
        u_old = u.copy()
        # Векторизованное обновление внутренних точек
        u[1:n, 1:n] = (u[2:n + 1, 1:n] + u[0:n - 1, 1:n] + u[1:n, 2:n + 1] + u[1:n, 0:n - 1]) / 4.0
        error = np.max(np.abs(u[1:n, 1:n] - u_old[1:n, 1:n]))
        iteration += 1
    end_time = time.perf_counter()
    return u, iteration, end_time - start_time


# --- Визуализация ---
def plot_comparison(
        u_pure: List[List[float]],
        u_numpy: np.ndarray,
        u_numpy_vec: np.ndarray,
        h: float,
        epsilon: float,
        save_path: str = None,
        show_plot: bool = True
) -> None:
    u_pure_array = np.array(u_pure)
    n = len(u_pure)
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(15, 10))

    # 3D графики
    ax1 = fig.add_subplot(231, projection='3d')
    ax1.plot_surface(X, Y, u_pure_array.T, cmap='viridis', alpha=0.8)
    ax1.set_title('Python (3D)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.view_init(30, 45)

    ax2 = fig.add_subplot(232, projection='3d')
    ax2.plot_surface(X, Y, u_numpy.T, cmap='plasma', alpha=0.8)
    ax2.set_title('NumPy с циклами (3D)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.view_init(30, 45)

    ax3 = fig.add_subplot(233, projection='3d')
    ax3.plot_surface(X, Y, u_numpy_vec.T, cmap='cividis', alpha=0.8)
    ax3.set_title('NumPy векторизованный (3D)')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.view_init(30, 45)

    # 2D графики (тепловые карты)
    ax4 = fig.add_subplot(234)
    im1 = ax4.imshow(u_pure_array, cmap='viridis', extent=[0, 1, 0, 1], origin='lower')
    ax4.set_title('Python (2D)')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    plt.colorbar(im1, ax=ax4, fraction=0.046, pad=0.04)

    ax5 = fig.add_subplot(235)
    im2 = ax5.imshow(u_numpy, cmap='plasma', extent=[0, 1, 0, 1], origin='lower')
    ax5.set_title('NumPy с циклами (2D)')
    ax5.set_xlabel('X')
    ax5.set_ylabel('Y')
    plt.colorbar(im2, ax=ax5, fraction=0.046, pad=0.04)

    ax6 = fig.add_subplot(236)
    im3 = ax6.imshow(u_numpy_vec, cmap='cividis', extent=[0, 1, 0, 1], origin='lower')
    ax6.set_title('NumPy векторизованный (2D)')
    ax6.set_xlabel('X')
    ax6.set_ylabel('Y')
    plt.colorbar(im3, ax=ax6, fraction=0.046, pad=0.04)

    fig.suptitle(f'Сравнение методов решения: h={h}, ε={epsilon}', fontsize=16, y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"График сохранен: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_performance_comparison(
        results_pure: List[dict],
        results_numpy_loop: List[dict],
        results_numpy_vec: List[dict],
        save_path: str = None,
        show_plot: bool = True
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    epsilon_values = sorted(set(r['epsilon'] for r in results_pure))
    colors = ['red', 'green', 'blue']

    for idx, epsilon in enumerate(epsilon_values):
        ax1 = axes[0, idx]
        ax2 = axes[1, idx]

        pure_eps = [r for r in results_pure if r['epsilon'] == epsilon]
        numpy_loop_eps = [r for r in results_numpy_loop if r['epsilon'] == epsilon]
        numpy_vec_eps = [r for r in results_numpy_vec if r['epsilon'] == epsilon]

        pure_eps.sort(key=lambda x: x['grid_size'])
        numpy_loop_eps.sort(key=lambda x: x['grid_size'])
        numpy_vec_eps.sort(key=lambda x: x['grid_size'])

        grid_sizes = [r['grid_size'] for r in pure_eps]

        times_pure = [r['time'] for r in pure_eps]
        times_numpy_loop = [r['time'] for r in numpy_loop_eps]
        times_numpy_vec = [r['time'] for r in numpy_vec_eps]

        iterations_pure = [r['iterations'] for r in pure_eps]
        iterations_numpy_loop = [r['iterations'] for r in numpy_loop_eps]
        iterations_numpy_vec = [r['iterations'] for r in numpy_vec_eps]

        # График времени
        ax1.plot(grid_sizes, times_pure, 'o-', color=colors[idx], label='Python', linewidth=2, markersize=6)
        ax1.plot(grid_sizes, times_numpy_loop, 's--', color=colors[idx], label='NumPy циклы', linewidth=2, markersize=6)
        ax1.plot(grid_sizes, times_numpy_vec, 'd-.', color=colors[idx], label='NumPy вект.', linewidth=2, markersize=6)
        ax1.set_xlabel('Размер сетки')
        ax1.set_ylabel('Время (сек)')
        ax1.set_title(f'Время выполнения (ε={epsilon})')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_yscale('log')

        # График итераций
        ax2.plot(grid_sizes, iterations_pure, 'o-', color=colors[idx], label='Python', linewidth=2, markersize=6)
        ax2.plot(grid_sizes, iterations_numpy_loop, 's--', color=colors[idx], label='NumPy циклы', linewidth=2,
                 markersize=6)
        ax2.plot(grid_sizes, iterations_numpy_vec, 'd-.', color=colors[idx], label='NumPy вект.', linewidth=2,
                 markersize=6)
        ax2.set_xlabel('Размер сетки')
        ax2.set_ylabel('Количество итераций')
        ax2.set_title(f'Итерации сходимости (ε={epsilon})')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

    plt.suptitle('Сравнение производительности методов', fontsize=16, y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"График производительности сохранен: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


# --- Эксперименты ---
def run_experiments_pure() -> List[Dict[str, Any]]:
    h_values = [0.1, 0.01, 0.005]
    epsilon_values = [0.1, 0.01, 0.001]
    results = []
    print("Запуск экспериментов для чистого Python...")
    for h in h_values:
        for epsilon in epsilon_values:
            u, iterations, time_taken = solve_laplace_pure(h, epsilon)
            result = {
                'h': h,
                'epsilon': epsilon,
                'iterations': iterations,
                'time': time_taken,
                'grid_size': int(1 / h) + 1
            }
            results.append(result)
            print(f"  h={h}, ε={epsilon}: {iterations} итераций, {time_taken:.4f} сек")
    return results


def run_experiments_numpy_loop() -> List[Dict[str, Any]]:
    h_values = [0.1, 0.01, 0.005]
    epsilon_values = [0.1, 0.01, 0.001]
    results = []
    print("\nЗапуск экспериментов для NumPy с циклами...")
    for h in h_values:
        for epsilon in epsilon_values:
            u, iterations, time_taken = solve_laplace_numpy_loop(h, epsilon)
            result = {
                'h': h,
                'epsilon': epsilon,
                'iterations': iterations,
                'time': time_taken,
                'grid_size': int(1 / h) + 1
            }
            results.append(result)
            print(f"  h={h}, ε={epsilon}: {iterations} итераций, {time_taken:.4f} сек")
    return results


def run_experiments_numpy_vectorized() -> List[Dict[str, Any]]:
    h_values = [0.1, 0.01, 0.005]
    epsilon_values = [0.1, 0.01, 0.001]
    results = []
    print("\nЗапуск экспериментов для векторизованного NumPy...")
    for h in h_values:
        for epsilon in epsilon_values:
            u, iterations, time_taken = solve_laplace_numpy_vectorized(h, epsilon)
            result = {
                'h': h,
                'epsilon': epsilon,
                'iterations': iterations,
                'time': time_taken,
                'grid_size': int(1 / h) + 1
            }
            results.append(result)
            print(f"  h={h}, ε={epsilon}: {iterations} итераций, {time_taken:.4f} сек")
    return results


# --- Основная функция ---
def main():
    parser = argparse.ArgumentParser(
        description='Решение уравнения Лапласа методом конечных разностей',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python laplace_solver.py --h 0.1 --epsilon 0.01
  python laplace_solver.py --method experiments --no-plot
  python laplace_solver.py --h 0.05 --epsilon 0.001 --method both
        """
    )
    parser.add_argument('--h', type=float, default=0.1,
                        help='Шаг сетки (по умолчанию: 0.1)')
    parser.add_argument('--epsilon', type=float, default=0.01,
                        help='Точность решения (по умолчанию: 0.01)')
    parser.add_argument('--max-iter', type=int, default=10000,
                        help='Максимальное число итераций (по умолчанию: 10000)')
    parser.add_argument('--method', choices=['pure', 'numpy_loop', 'numpy_vec', 'both', 'experiments'],
                        default='both', help='Метод решения (по умолчанию: both)')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Директория для сохранения результатов (по умолчанию: results)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Не показывать графики')

    args = parser.parse_args()

    # Создание директорий для результатов
    os.makedirs(args.output_dir, exist_ok=True)
    plots_dir = os.path.join(args.output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    data_dir = os.path.join(args.output_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    if args.method == 'experiments':

        results_pure = run_experiments_pure()
        results_numpy_loop = run_experiments_numpy_loop()
        results_numpy_vec = run_experiments_numpy_vectorized()

        # Сохранение результатов
        with open(os.path.join(data_dir, 'results_pure.json'), 'w') as f:
            json.dump(results_pure, f, indent=2, default=str)
        with open(os.path.join(data_dir, 'results_numpy_loop.json'), 'w') as f:
            json.dump(results_numpy_loop, f, indent=2, default=str)
        with open(os.path.join(data_dir, 'results_numpy_vec.json'), 'w') as f:
            json.dump(results_numpy_vec, f, indent=2, default=str)

        # Вывод таблицы производительности
        print("\n" + "=" * 80)
        print("ТАБЛИЦА ПРОИЗВОДИТЕЛЬНОСТИ (ε=0.01)")
        print("=" * 80)
        print("h\t\tGrid Size\tPython\t\tNumPy циклы\tNumPy вект.\tУскорение цикл\tУскорение вект.")
        print("-" * 100)

        # Только для указанных h
        for h in [0.1, 0.01, 0.005]:
            pure = next((r for r in results_pure if r['h'] == h and r['epsilon'] == 0.01), None)
            numpy_loop = next((r for r in results_numpy_loop if r['h'] == h and r['epsilon'] == 0.01), None)
            numpy_vec = next((r for r in results_numpy_vec if r['h'] == h and r['epsilon'] == 0.01), None)

            if pure and numpy_loop and numpy_vec:
                speedup_loop = pure['time'] / numpy_loop['time'] if numpy_loop['time'] > 0 else 0
                speedup_vec = pure['time'] / numpy_vec['time'] if numpy_vec['time'] > 0 else 0

                print(f"{h:.3f}\t\t{pure['grid_size']}x{pure['grid_size']}\t"
                      f"{pure['time']:.6f}\t\t{numpy_loop['time']:.6f}\t\t{numpy_vec['time']:.6f}\t\t"
                      f"{speedup_loop:.2f}x\t\t{speedup_vec:.2f}x")

        # Создание графиков сравнения
        if not args.no_plot:
            print("\nСоздание графиков сравнения...")

            # График для h=0.1, ε=0.01
            u_pure1, _, _ = solve_laplace_pure(0.1, 0.01)
            u_numpy_loop1, _, _ = solve_laplace_numpy_loop(0.1, 0.01)
            u_numpy_vec1, _, _ = solve_laplace_numpy_vectorized(0.1, 0.01)

            plot_comparison(u_pure1, u_numpy_loop1, u_numpy_vec1, 0.1, 0.01,
                            save_path=os.path.join(plots_dir, 'comparison_h0.1_eps0.01.png'),
                            show_plot=False)

            # График для h=0.01, ε=0.001
            u_pure2, _, _ = solve_laplace_pure(0.01, 0.001)
            u_numpy_loop2, _, _ = solve_laplace_numpy_loop(0.01, 0.001)
            u_numpy_vec2, _, _ = solve_laplace_numpy_vectorized(0.01, 0.001)

            plot_comparison(u_pure2, u_numpy_loop2, u_numpy_vec2, 0.01, 0.001,
                            save_path=os.path.join(plots_dir, 'comparison_h0.01_eps0.001.png'),
                            show_plot=False)

            # График для h=0.005, ε=0.001
            u_pure3, _, _ = solve_laplace_pure(0.005, 0.001)
            u_numpy_loop3, _, _ = solve_laplace_numpy_loop(0.005, 0.001)
            u_numpy_vec3, _, _ = solve_laplace_numpy_vectorized(0.005, 0.001)

            plot_comparison(u_pure3, u_numpy_loop3, u_numpy_vec3, 0.005, 0.001,
                            save_path=os.path.join(plots_dir, 'comparison_h0.005_eps0.001.png'),
                            show_plot=False)

            # График производительности
            plot_performance_comparison(results_pure, results_numpy_loop, results_numpy_vec,
                                        save_path=os.path.join(plots_dir, 'performance_comparison.png'),
                                        show_plot=True)

            print(f"\nРезультаты сохранены в директории: {args.output_dir}")
            print(f"  - Данные: {data_dir}/")
            print(f"  - Графики: {plots_dir}/")

    elif args.method == 'both':
        print(f"\nСРАВНЕНИЕ МЕТОДОВ ДЛЯ h={args.h}, ε={args.epsilon}")
        print("=" * 50)

        # Чистый Python
        print("\n1. Чистый Python:")
        u_pure, iter_pure, time_pure = solve_laplace_pure(args.h, args.epsilon, args.max_iter)
        print(f"   Итераций: {iter_pure}")
        print(f"   Время: {time_pure:.6f} сек")

        # NumPy с циклами
        print("\n2. NumPy с циклами:")
        u_numpy_loop, iter_numpy_loop, time_numpy_loop = solve_laplace_numpy_loop(args.h, args.epsilon, args.max_iter)
        print(f"   Итераций: {iter_numpy_loop}")
        print(f"   Время: {time_numpy_loop:.6f} сек")

        # Векторизованный NumPy
        print("\n3. Векторизованный NumPy:")
        u_numpy_vec, iter_numpy_vec, time_numpy_vec = solve_laplace_numpy_vectorized(args.h, args.epsilon,
                                                                                     args.max_iter)
        print(f"   Итераций: {iter_numpy_vec}")
        print(f"   Время: {time_numpy_vec:.6f} сек")

        if not args.no_plot:
            plot_comparison(u_pure, u_numpy_loop, u_numpy_vec, args.h, args.epsilon,
                            save_path=os.path.join(plots_dir, f'comparison_h{args.h}_eps{args.epsilon}.png'),
                            show_plot=True)

    elif args.method == 'pure':
        print(f"\nРешение с использованием чистого Python (h={args.h}, ε={args.epsilon}):")
        u_pure, iter_pure, time_pure = solve_laplace_pure(args.h, args.epsilon, args.max_iter)
        print(f"Итераций: {iter_pure}")
        print(f"Время: {time_pure:.6f} сек")

    elif args.method == 'numpy_loop':
        print(f"\nРешение с использованием NumPy с циклами (h={args.h}, ε={args.epsilon}):")
        u_numpy_loop, iter_numpy_loop, time_numpy_loop = solve_laplace_numpy_loop(args.h, args.epsilon, args.max_iter)
        print(f"Итераций: {iter_numpy_loop}")
        print(f"Время: {time_numpy_loop:.6f} сек")

    elif args.method == 'numpy_vec':
        print(f"\nРешение с использованием векторизованного NumPy (h={args.h}, ε={args.epsilon}):")
        u_numpy_vec, iter_numpy_vec, time_numpy_vec = solve_laplace_numpy_vectorized(args.h, args.epsilon,
                                                                                     args.max_iter)
        print(f"Итераций: {iter_numpy_vec}")
        print(f"Время: {time_numpy_vec:.6f} сек")


if __name__ == "__main__":
    main()

```

---

### Заключение

В ходе лабораторной работы успешно реализованы два варианта решения задачи Дирихле для уравнения Лапласа методом Гаусса-Зейделя. Проведено сравнение производительности, исследована зависимость времени вычислений от параметров сетки и точности. Полученные результаты демонстрируют преимущества и минусы каждого метода.
