import numpy as np

def determinant(matrix):
    """Вычисляет определитель матрицы."""
    return np.linalg.det(matrix)

def kramer_method(coefficients, results):
    """
    Решает систему линейных уравнений методом Крамера.
    
    Parameters:
    coefficients (numpy.ndarray): Матрица коэффициентов.
    results (numpy.ndarray): Вектор правых частей уравнений. 
    
    Returns:
    numpy.ndarray: Вектор решений.
    """
    det_main = determinant(coefficients)
    
    if det_main == 0:
        raise ValueError("Определитель матрицы коэффициентов равен нулю. Решений нет или бесконечно много.")
    
    n = coefficients.shape[0]
    solutions = np.zeros(n, dtype=complex)
    
    for i in range(n):
        # Создаём матрицу, где i-я колонка заменена вектором правых частей
        modified_matrix = np.copy(coefficients)
        modified_matrix[:, i] = results
        
        det_modified = determinant(modified_matrix)
        
        # Вычисляем решение для i-й переменной
        solutions[i] = det_modified / det_main
    
    # Если все решения являются действительными числами без мнимой части,
    # выводим их как действительные числа
    if np.allclose(solutions.imag, 0):
        # Проверяем, являются ли решения целыми числами
        if np.allclose(solutions.real, np.round(solutions.real)):
            solutions = np.round(solutions.real).astype(int)
        else:
            solutions = solutions.real
    
    return solutions

# Пример использования для матрицы 2х2
if __name__ == "__main__":
    # Матрица коэффициентов
    coefficients_2x2 = np.array([[complex(1, 2), 2],
                                 [3, complex(4, 4)]])
    
    # Вектор правых частей
    results_2x2 = np.array([5, 11])
    
    try:
        solutions_2x2 = kramer_method(coefficients_2x2, results_2x2)
        print("Решения для матрицы 2х2:", solutions_2x2)
    except ValueError as e:
        print(e)

    # Матрица коэффициентов 5х5
    coefficients_5x5 = np.array([
        [complex(1, 3), 2, complex(4, 1), 5, 6],
        [7, complex(8, 2), 9, complex(10, 3), 11],
        [12, 13, complex(14, 4), complex(15, 5), 16],
        [complex(17, 6), 18, 19, complex(20, 7), complex(21, 8)],
        [22, complex(23, 9), complex(24, 10), 25, complex(26, 11)]
    ])

    # Вектор правых частей
    results_5x5 = np.array([27, complex(28, 12), 29, complex(30, 13), 31])

    try:
        solutions_5x5 = kramer_method(coefficients_5x5, results_5x5)
        print("Решения для матрицы 5х5:", solutions_5x5)
    except ValueError as e:
        print(e)
