import numpy as np

def generate_fft_terrain(size=256, beta=2.0, seed=42):
    """
    Генерирует 2D карту высот методом спектрального синтеза с использованием 2D FFT.
    
    :param size: Размер сетки ландшафта (желательно степень двойки)
    :param beta: Спектральный индекс (шероховатость: ~1.0 - скалы, ~2.5 - пологие холмы)
    :param seed: Зерно для генератора случайных чисел
    :return: 2D массив высот Z размерностью (size, size), нормализованный в диапазон [0, 1]
    """
    rng = np.random.default_rng(seed)
    
    # 1. Генерация матрицы белого шума
    noise = rng.normal(size=(size, size))
    
    # 2. Прямое быстрое преобразование Фурье
    F = np.fft.fft2(noise)
    F_shifted = np.fft.fftshift(F)
    
    # 3. Частотная фильтрация (1/f^beta)
    # Создаем сетку частот от -0.5 до 0.5
    freq_u = np.fft.fftshift(np.fft.fftfreq(size))
    freq_v = np.fft.fftshift(np.fft.fftfreq(size))
    u, v = np.meshgrid(freq_u, freq_v)
    
    # Вычисляем модули частот
    f = np.sqrt(u**2 + v**2)
    
    # Избегаем деления на ноль в центре (нулевая частота / постоянная составляющая)
    zero_mask = (u == 0) & (v == 0)
    f_safe = np.where(zero_mask, 1.0, f)
    
    # Фильтр спадания амплитуды
    filter_h = 1.0 / (f_safe ** beta)
    # Зануляем постоянную составляющую (среднюю высоту рельефа), чтобы она не влияла на дисперсию
    filter_h[zero_mask] = 0.0
    
    # Применяем фильтр
    H = F_shifted * filter_h
    
    # 4. Обратное преобразование Фурье
    Z = np.real(np.fft.ifft2(np.fft.ifftshift(H)))
    
    # 5. Нормализация в диапазон [0, 1]
    z_min, z_max = Z.min(), Z.max()
    if z_max > z_min:
        Z = (Z - z_min) / (z_max - z_min)
    else:
        Z = np.zeros_like(Z)
        
    return Z

def generate_analytical_terrain(grid_x, grid_y, beta=2.0, num_harmonics=50, seed=42):
    """
    Генерирует ландшафт аналитическим методом гармоник (суммированием синусоид).
    Подходит для бесконечной бесшовной генерации по любым координатам.
    
    :param grid_x: 2D массив координат X
    :param grid_y: 2D массив координат Y
    :param beta: Спектральный индекс (шероховатость)
    :param num_harmonics: Количество суммируемых гармоник
    :param seed: Зерно генератора случайных чисел
    :return: 2D массив высот Z того же размера, что и grid_x, нормализованный в [0, 1]
    """
    rng = np.random.default_rng(seed)
    
    # Генерируем случайные направления и частоты для гармоник
    # Ограничиваем диапазон частот, чтобы рельеф выглядел гармонично
    k_min, k_max = 0.1, 8.0
    
    # Распределяем частоты логарифмически для охвата разных масштабов (октав)
    log_k = rng.uniform(np.log(k_min), np.log(k_max), size=num_harmonics)
    k_mags = np.exp(log_k)
    
    # Случайные углы направлений волн
    angles = rng.uniform(0, 2 * np.pi, size=num_harmonics)
    
    k_x = k_mags * np.cos(angles)
    k_y = k_mags * np.sin(angles)
    
    # Случайные фазовые сдвиги
    phases = rng.uniform(0, 2 * np.pi, size=num_harmonics)
    
    # Вычисляем амплитуды по закону 1/f^beta
    amplitudes = 1.0 / (k_mags ** beta)
    
    # Нормализуем амплитуды для предотвращения переполнения
    if np.sum(amplitudes) > 0:
        amplitudes /= np.sum(amplitudes)
        
    # Аналитическое суммирование гармоник
    Z = np.zeros_like(grid_x, dtype=float)
    for i in range(num_harmonics):
        Z += amplitudes[i] * np.sin(k_x[i] * grid_x + k_y[i] * grid_y + phases[i])
        
    # Нормализация в [0, 1]
    z_min, z_max = Z.min(), Z.max()
    if z_max > z_min:
        Z = (Z - z_min) / (z_max - z_min)
    else:
        Z = np.zeros_like(Z)
        
    return Z

def calculate_normals(Z, dx=1.0, dy=1.0):
    """
    Вычисляет единичные векторы нормалей для каждой точки высотной карты Z.
    
    :param Z: 2D массив высот
    :param dx: Шаг сетки по оси X
    :param dy: Шаг сетки по оси Y
    :return: 3D массив нормалей размера (H, W, 3)
    """
    # np.gradient возвращает производные (по строкам / Y, по столбцам / X)
    grad_y, grad_x = np.gradient(Z, dy, dx)
    
    # Вектор нормали: n = (-dZ/dx, -dZ/dy, 1)
    nx = -grad_x
    ny = -grad_y
    nz = np.ones_like(Z)
    
    # Нормируем векторы
    norm = np.sqrt(nx**2 + ny**2 + nz**2)
    nx /= norm
    ny /= norm
    nz /= norm
    
    return np.stack([nx, ny, nz], axis=-1)

def calculate_lighting(normals, light_dir=(1.0, 1.0, 1.0), ambient=0.15):
    """
    Вычисляет интенсивность освещения по закону Ламберта с добавлением фонового (ambient) света.
    
    :param normals: 3D массив нормалей (H, W, 3)
    :param light_dir: Вектор направления света (источник солнца)
    :param ambient: Интенсивность рассеянного света (0.0 - 1.0)
    :return: 2D массив интенсивностей освещения (H, W) в диапазоне [ambient, 1.0]
    """
    # Нормируем вектор направления света
    l_vec = np.array(light_dir, dtype=float)
    l_norm = np.linalg.norm(l_vec)
    if l_norm > 0:
        l_vec /= l_norm
    else:
        l_vec = np.array([0.0, 0.0, 1.0])
        
    # Скалярное произведение в каждой точке (H, W, 3) x (3,) -> (H, W)
    dot = np.tensordot(normals, l_vec, axes=(-1, 0))
    
    # Применяем модель Ламберта: I = ambient + (1 - ambient) * max(0, cos(theta))
    intensity = ambient + (1.0 - ambient) * np.clip(dot, 0.0, 1.0)
    return intensity

def get_biome_colors(Z, is_smooth=True):
    """
    Определяет цвета биомов для каждой вершины на основе её высоты.
    
    :param Z: 2D массив высот [0, 1]
    :param is_smooth: Если True, то цвета интерполируются плавно. Иначе делятся по жестким порогам.
    :return: 3D массив цветов RGB размера (H, W, 3) со значениями в диапазоне [0, 1]
    """
    if is_smooth:
        # Плавные переходы цветов по точкам интерполяции
        xp = [0.0, 0.24, 0.25, 0.32, 0.45, 0.65, 0.85, 1.0]
        # Цвета: Глубокая вода -> Мелководье -> Песок -> Равнина -> Лес/Холмы -> Скалы -> Снег
        fp_r = [10,  20,  235,  46,   34,  110, 130, 245]
        fp_g = [46,  80,  200, 139,  100,  110, 130, 245]
        fp_b = [92, 160,  128,  87,   60,  115, 135, 250]
        
        r = np.interp(Z, xp, fp_r)
        g = np.interp(Z, xp, fp_g)
        b = np.interp(Z, xp, fp_b)
        
        return np.stack([r, g, b], axis=-1) / 255.0
    else:
        # Дискретные пороги согласно ТЗ
        colors = np.zeros(Z.shape + (3,))
        
        # Глубоководный океан (Темно-синий)
        mask_ocean = Z < 0.25
        colors[mask_ocean] = [10/255, 46/255, 92/255]
        
        # Мелководье / Пляж (Желтый песок)
        mask_beach = (Z >= 0.25) & (Z < 0.32)
        colors[mask_beach] = [230/255, 194/255, 128/255]
        
        # Долины / Равнины (Насыщенный зеленый)
        mask_plains = (Z >= 0.32) & (Z < 0.65)
        colors[mask_plains] = [46/255, 139/255, 87/255]
        
        # Горные склоны / Скалы (Серый)
        mask_mountains = (Z >= 0.65) & (Z < 0.85)
        colors[mask_mountains] = [128/255, 128/255, 128/255]
        
        # Ледники и вечные снега (Чисто белый)
        mask_snow = Z >= 0.85
        colors[mask_snow] = [255/255, 255/255, 255/255]
        
        return colors

def get_shaded_terrain_colors(Z, light_dir=(1.0, 1.0, 1.0), is_smooth=True, ambient=0.15):
    """
    Выполняет полный расчет затенения и биомов, возвращая финальную раскраску.
    
    :return: 3D массив RGB цветов (H, W, 3)
    """
    normals = calculate_normals(Z)
    intensity = calculate_lighting(normals, light_dir, ambient)
    base_colors = get_biome_colors(Z, is_smooth)
    
    # Накладываем тени (умножаем базовый цвет на интенсивность света)
    shaded_colors = base_colors * intensity[..., np.newaxis]
    return np.clip(shaded_colors, 0.0, 1.0)
