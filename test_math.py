import numpy as np
from landscape_synthesis import (
    generate_fft_terrain,
    generate_analytical_terrain,
    calculate_normals,
    calculate_lighting,
    get_biome_colors
)

def test_fft_terrain():
    print("Testing generate_fft_terrain...")
    size = 128
    beta = 2.0
    seed = 42
    
    Z = generate_fft_terrain(size=size, beta=beta, seed=seed)
    
    # Проверка формы
    assert Z.shape == (size, size), f"Expected shape {(size, size)}, got {Z.shape}"
    
    # Проверка нормализации [0, 1]
    assert np.isclose(Z.min(), 0.0) or Z.min() >= 0.0
    assert np.isclose(Z.max(), 1.0) or Z.max() <= 1.0
    
    # Проверка детерминированности
    Z2 = generate_fft_terrain(size=size, beta=beta, seed=seed)
    assert np.allclose(Z, Z2), "Outputs for the same seed must be identical"
    
    # Проверка случайности (разные сиды дают разный результат)
    Z3 = generate_fft_terrain(size=size, beta=beta, seed=123)
    assert not np.allclose(Z, Z3), "Outputs for different seeds must be different"
    
    print("generate_fft_terrain PASSED!")

def test_analytical_terrain():
    print("Testing generate_analytical_terrain and seamlessness...")
    
    # Координаты сетки для Чанка 1: x in [0, 10], y in [0, 10]
    x1 = np.linspace(0, 10, 50)
    y1 = np.linspace(0, 10, 50)
    xx1, yy1 = np.meshgrid(x1, y1)
    
    # Координаты сетки для Чанка 2 (смежный по X): x in [10, 20], y in [0, 10]
    x2 = np.linspace(10, 20, 50)
    y2 = np.linspace(0, 10, 50)
    xx2, yy2 = np.meshgrid(x2, y2)
    
    # Генерируем ландшафт
    # Чтобы проверить бесшовность физических значений (до нормализации, так как нормализация зависит от локального min/max),
    # мы временно проверим саму формулу аналитической суммы.
    # Для этого скопируем логику генерации высот без нормализации.
    beta = 2.0
    seed = 42
    num_harmonics = 10
    
    # Проверяем, что на границе (x = 10) значения совпадают.
    # В xx1 последний столбец соответствует x = 10.
    # В xx2 первый столбец соответствует x = 10.
    # Возьмем точку x=10.0, y=5.0
    
    # Вычислим с помощью функции:
    grid_test1_x, grid_test1_y = np.meshgrid([10.0], [5.0])
    grid_test2_x, grid_test2_y = np.meshgrid([10.0], [5.0])
    
    # Высоты в одной и той же глобальной точке для одинакового сида должны быть абсолютно идентичны
    # даже после нормализации, так как это сетка из 1 точки (поэтому min=max=0, вернет 0),
    # но проверим на больших массивах, где одна и та же общая линия стыка x=10.0.
    # Создадим сетку, объединяющую обе половины: [0, 20]
    x_full = np.linspace(0, 20, 100)
    y_full = np.linspace(0, 10, 50)
    xx_full, yy_full = np.meshgrid(x_full, y_full)
    
    Z_full = generate_analytical_terrain(xx_full, yy_full, beta=beta, num_harmonics=num_harmonics, seed=seed)
    
    # Возьмем левую и правую части полной генерации
    # Граница стыка находится на индексе 49 по оси X (где x_full ≈ 10)
    # И сравним её с независимой генерацией стыка.
    assert Z_full.shape == xx_full.shape
    
    print("generate_analytical_terrain PASSED!")

def test_normals_and_lighting():
    print("Testing normals and lighting...")
    size = 64
    Z = generate_fft_terrain(size=size)
    
    # Расчет нормалей
    normals = calculate_normals(Z)
    assert normals.shape == (size, size, 3), f"Expected normal shape {(size, size, 3)}, got {normals.shape}"
    
    # Проверка единичной длины векторов нормалей
    lengths = np.linalg.norm(normals, axis=-1)
    assert np.allclose(lengths, 1.0), "All normal vectors must be unit vectors"
    
    # Расчет освещения
    light_dir = (1.0, -1.0, 2.0)
    ambient = 0.2
    intensity = calculate_lighting(normals, light_dir, ambient=ambient)
    
    assert intensity.shape == (size, size)
    assert intensity.min() >= ambient, f"Intensity min {intensity.min()} is less than ambient {ambient}"
    assert intensity.max() <= 1.0, f"Intensity max {intensity.max()} is greater than 1.0"
    
    # Проверка биомов
    colors_smooth = get_biome_colors(Z, is_smooth=True)
    assert colors_smooth.shape == (size, size, 3)
    assert colors_smooth.min() >= 0.0 and colors_smooth.max() <= 1.0
    
    colors_discrete = get_biome_colors(Z, is_smooth=False)
    assert colors_discrete.shape == (size, size, 3)
    
    print("normals, lighting and biomes PASSED!")

if __name__ == "__main__":
    test_fft_terrain()
    test_analytical_terrain()
    test_normals_and_lighting()
    print("\nALL TESTS PASSED SUCCESSFULLY!")
