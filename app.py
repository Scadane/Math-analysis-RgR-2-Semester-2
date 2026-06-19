import streamlit as st
import numpy as np
import plotly.graph_objects as go
from landscape_synthesis import (
    generate_fft_terrain,
    generate_analytical_terrain,
    calculate_normals,
    calculate_lighting,
    get_biome_colors
)

# Настройка страницы
st.set_page_config(
    page_title="Процедурная генерация 3D-ландшафта",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Пользовательские стили для красивого премиального интерфейса (Dark theme friendly)
st.markdown("""
<style>
    .main-title {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FF6B6B, #4ECDC4, #45B6FE);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
        text-align: center;
    }
    .subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        color: #888888;
        margin-bottom: 2rem;
        text-align: center;
    }
    .math-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background: linear-gradient(135deg, #4ECDC4, #45B6FE) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(78, 205, 196, 0.4) !important;
    }
</style>
""", unsafe_allow_html=True)

# Заголовок
st.markdown('<div class="main-title">🏔️ Процедурная генерация 3D-ландшафта</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Интерактивный генератор рельефа на основе спектрального и аналитического методов</div>', unsafe_allow_html=True)

# Инициализация сессии для случайного Seed
if "seed" not in st.session_state:
    st.session_state.seed = 42

# Функция генерации случайного сида
def randomize_seed():
    st.session_state.seed = int(np.random.randint(1, 9999))

# Названия режимов (единый источник — используется и в логике, и в текстах)
MODE_SPECTRAL = "Спектральный метод"
MODE_ANALYTICAL = "Аналитический метод"

# Боковая панель
st.sidebar.header("⚙️ Параметры генерации")

# Кнопка генерации нового сида
col_seed1, col_seed2 = st.sidebar.columns([3, 1])
with col_seed1:
    seed = st.number_input("Зерно генератора (Seed):", min_value=1, max_value=99999, value=st.session_state.seed, key="seed_input")
    st.session_state.seed = seed
with col_seed2:
    st.write("")  # Отступ
    st.write("")
    if st.button("🎲"):
        randomize_seed()
        st.rerun()

# Выбор режима
mode = st.sidebar.radio(
    "Метод генерации:",
    [MODE_SPECTRAL, MODE_ANALYTICAL]
)

st.sidebar.markdown("---")
st.sidebar.subheader("🎨 Свойства рельефа")

beta = st.sidebar.slider(
    "Шероховатость (Спектральный индекс β):",
    min_value=0.5,
    max_value=3.5,
    value=1.8,
    step=0.1,
    help="β определяет скорость затухания высоких частот. Больше β — более гладкий рельеф (холмы), меньше β — острый и скалистый (горы)."
)

# Описание характера рельефа на основе бета
if beta < 1.0:
    roughness_desc = "🌋 Экстремально острые фрактальные скалы"
elif beta < 1.6:
    roughness_desc = "⛰️ Острые горные хребты"
elif beta < 2.2:
    roughness_desc = "🏞️ Сбалансированный холмистый ландшафт"
else:
    roughness_desc = "🌳 Плавные старые холмы / равнины"
st.sidebar.caption(roughness_desc)

z_scale = st.sidebar.slider("Масштаб высоты Z:", min_value=5.0, max_value=80.0, value=30.0, step=2.5)

smooth_biomes = st.sidebar.checkbox("Плавные биомы (интерполяция)", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("🗺️ Размер мира")

radius = st.sidebar.slider(
    "Радиус генерации (R):",
    min_value=1,
    max_value=3,
    value=1,
    step=1,
    help="Мир разбивается на чанки. Радиус R даёт (2R−1)×(2R−1) чанков: R=1 → 1 чанк, R=2 → 9 чанков, R=3 → 25 чанков. Чанки бесшовно сшиваются."
)
n_side = 2 * radius - 1
st.sidebar.caption(f"Текущий мир: {n_side}×{n_side} = {n_side * n_side} чанк(ов/а).")

st.sidebar.markdown("---")
st.sidebar.subheader("☀️ Освещение солнца")
sun_alt = st.sidebar.slider("Высота солнца над горизонтом (°):", min_value=10, max_value=90, value=45)
sun_az = st.sidebar.slider("Азимут направления солнца (°):", min_value=0, max_value=360, value=135)

# Расчет вектора солнца
alt_rad = np.radians(sun_alt)
az_rad = np.radians(sun_az)
lx = np.cos(alt_rad) * np.cos(az_rad)
ly = np.cos(alt_rad) * np.sin(az_rad)
lz = np.sin(alt_rad)
light_dir = (lx, ly, lz)

# ────────────────────────────────────────────────────────────────
# Генерация ОДНОЙ большой бесшовной поверхности для всего мира.
# Идея бесшовности: мир строится как единый массив, который затем
# нарезается на чанки с шагом (grid_size-1). Соседние чанки разделяют
# ровно одну строку/столбец точек — значения на стыке идентичны.
# ────────────────────────────────────────────────────────────────

# Специфичные параметры и генерация по методам
if mode == MODE_SPECTRAL:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📐 Параметры спектрального метода")
    grid_size = st.sidebar.selectbox("Разрешение сетки (на чанк):", [64, 128, 256], index=1)

    # Размер единой поверхности: (2R-1) чанков с общей границей
    big = n_side * (grid_size - 1) + 1

    @st.cache_data
    def get_fft_world(big_size, b, s):
        return generate_fft_terrain(size=big_size, beta=b, seed=s)

    Z_big = get_fft_world(big, beta, st.session_state.seed)

    # Координаты (шаг сетки = 1.0)
    step = 1.0
    coords = np.arange(big) * step
    X_big, Y_big = np.meshgrid(coords, coords)
    dx = step
    dy = step
    chunk_size_spatial = (grid_size - 1) * step

    # Отдельно — спектры для вкладки фильтрации (на одном чанке)
    @st.cache_data
    def get_fft_data(size, b, s):
        rng = np.random.default_rng(s)
        noise = rng.normal(size=(size, size))
        F = np.fft.fft2(noise)
        F_shifted = np.fft.fftshift(F)

        freq_u = np.fft.fftshift(np.fft.fftfreq(size))
        freq_v = np.fft.fftshift(np.fft.fftfreq(size))
        u, v = np.meshgrid(freq_u, freq_v)
        f_grid = np.sqrt(u**2 + v**2)
        zero_mask = (u == 0) & (v == 0)
        f_safe = np.where(zero_mask, 1.0, f_grid)

        filter_h = 1.0 / (f_safe ** b)
        filter_h[zero_mask] = 0.0

        H = F_shifted * filter_h

        log_spec_noise = np.log10(np.abs(F_shifted) + 1e-6)
        log_filter = np.log10(filter_h + 1e-6)
        log_spec_filtered = np.log10(np.abs(H) + 1e-6)

        return log_spec_noise, log_filter, log_spec_filtered

    spec_noise, spec_filter, spec_filtered = get_fft_data(grid_size, beta, st.session_state.seed)

else:  # MODE_ANALYTICAL
    st.sidebar.markdown("---")
    st.sidebar.subheader("📐 Параметры аналитического метода")
    num_harmonics = st.sidebar.slider("Количество гармоник (K):", min_value=5, max_value=120, value=50, step=5)

    grid_size = 100
    chunk_size_spatial = 10.0

    # Размер единой поверхности: (2R-1) чанков с общей границей
    big = n_side * (grid_size - 1) + 1
    total_size = n_side * chunk_size_spatial

    # Симметрично вокруг центра мира (0,0)
    coords = np.linspace(-total_size / 2, total_size / 2, big)
    X_big, Y_big = np.meshgrid(coords, coords)
    dx = chunk_size_spatial / (grid_size - 1)
    dy = chunk_size_spatial / (grid_size - 1)

    @st.cache_data
    def get_analytical_world(cx, cy, b, harm, s):
        return generate_analytical_terrain(cx, cy, beta=b, num_harmonics=harm, seed=s)

    Z_big = get_analytical_world(X_big, Y_big, beta, num_harmonics, st.session_state.seed)

# ────────────────────────────────────────────────────────────────
# Расчёт освещения. КЛЮЧЕВОЙ МОМЕНТ ФИКСА СОЛНЦА:
# нормали считаются от ЭФФЕКТИВНОЙ высоты (Z * z_scale) с корректным
# шагом сетки, чтобы уклоны соответствовали тому, что видит глаз.
# Раньше нормали были почти вертикальны (уклон ~0.004), поэтому
# ползунки солнца не давали видимого эффекта.
# Биомы при этом берутся от нормализованной высоты Z (корректные пороги).
# ────────────────────────────────────────────────────────────────
Z_eff = Z_big * z_scale
normals = calculate_normals(Z_eff, dx=dx, dy=dy)
intensity = calculate_lighting(normals, light_dir)
base_colors = get_biome_colors(Z_big, is_smooth=smooth_biomes)
colors_big = np.clip(base_colors * intensity[..., np.newaxis], 0.0, 1.0)

# Переводим в rgb строки для Plotly
c_int = (colors_big * 255).astype(int)
rgb_big = np.char.add(
    np.char.add(
        np.char.add(
            np.char.add("rgb(", c_int[..., 0].astype(str)),
            ","
        ),
        np.char.add(c_int[..., 1].astype(str), ",")
    ),
    np.char.add(c_int[..., 2].astype(str), ")")
)

# ── Нарезка на чанки с общей границей (шаг grid_size-1) ──────────
def slice_chunk(arr, n, i, j):
    """Вырезает чанк (i, j) размера n×n. Соседние чанки делят границу."""
    return arr[i * (n - 1): i * (n - 1) + n, j * (n - 1): j * (n - 1) + n]

# Вкладки на основном экране
tab_3d, tab_spectra, tab_math = st.tabs(["🖥️ Интерактивный 3D-рендер", "📊 Частотная фильтрация", "📖 Математическая справка"])

with tab_3d:
    st.markdown('<div class="math-card">', unsafe_allow_html=True)
    chunks_count = n_side * n_side
    st.markdown(f"**Текущий режим:** {mode}. "
                f"Сгенерировано {chunks_count} чанк(ов/а) при радиусе R={radius}. "
                f"Высота вершин отмасштабирована в {z_scale} раз. "
                "Вращайте, приближайте и исследуйте ландшафт мышью. "
                "Меняйте ползунки солнца — тени перемещаются по склонам.", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Сборка 3D-сцены: отдельные Surface для каждого чанка + линии границ
    traces = []
    for i in range(n_side):
        for j in range(n_side):
            Zc = slice_chunk(Z_big, grid_size, i, j)
            Xc = slice_chunk(X_big, grid_size, i, j)
            Yc = slice_chunk(Y_big, grid_size, i, j)
            Cc = slice_chunk(rgb_big, grid_size, i, j)
            traces.append(go.Surface(
                x=Xc, y=Yc, z=Zc * z_scale,
                surfacecolor=Cc,
                showscale=False,
                hoverinfo='none',
                contours=dict(
                    z=dict(show=False),
                    x=dict(show=False),
                    y=dict(show=False),
                )
            ))

    # Линии границ чанков: сетка (n_side+1)×(n_side+1), слегка приподнята
    if radius > 1:
        border_lift = 0.5
        border_line = dict(color="rgba(15,15,15,0.55)", width=2)
        for k in range(n_side + 1):
            idx = min(k * (grid_size - 1), big - 1)
            # Вертикальная линия (фиксированный X-индекс, вдоль Y)
            traces.append(go.Scatter3d(
                x=X_big[idx, :], y=Y_big[idx, :], z=Z_big[idx, :] * z_scale + border_lift,
                mode="lines", line=border_line, hoverinfo="skip", showlegend=False
            ))
            # Горизонтальная линия (фиксированный Y-индекс, вдоль X)
            traces.append(go.Scatter3d(
                x=X_big[:, idx], y=Y_big[:, idx], z=Z_big[:, idx] * z_scale + border_lift,
                mode="lines", line=border_line, hoverinfo="skip", showlegend=False
            ))

    fig3d = go.Figure(data=traces)
    fig3d.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis=dict(title='X', showgrid=False, showticklabels=False),
            yaxis=dict(title='Y', showgrid=False, showticklabels=False),
            zaxis=dict(title='Высота Z', showgrid=False, showticklabels=False),
            aspectratio=dict(x=1, y=1, z=0.15 + z_scale / 120.0)
        ),
        height=650,
        showlegend=False
    )

    st.plotly_chart(fig3d, use_container_width=True)

    if radius >= 3:
        st.caption("ℹ️ При большом радиусе рендеринг может занимать больше времени из-за числа чанков.")

with tab_spectra:
    if mode == MODE_SPECTRAL:
        st.subheader("Визуализация процесса фильтрации в частотной области")
        st.write("Каждый пиксель на спектральной карте ниже представляет амплитуду определенной пространственной частоты. "
                 "Постоянная составляющая (нулевая частота) находится строго в центре карт.")

        col_s1, col_s2, col_s3 = st.columns(3)

        # Функция для создания красивых 2D спектрограмм
        def plot_spectrum_heatmap(data, title, colorscale="Viridis"):
            fig = go.Figure(data=go.Heatmap(
                z=data,
                colorscale=colorscale,
                showscale=True
            ))
            fig.update_layout(
                title=dict(text=title, x=0.5),
                margin=dict(l=10, r=10, b=10, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=350
            )
            return fig

        with col_s1:
            st.plotly_chart(
                plot_spectrum_heatmap(spec_noise, "1. Спектр белого шума (Равномерный)"),
                use_container_width=True
            )
            st.caption("Амплитуды частот распределены равномерно по всем направлениям и масштабам.")

        with col_s2:
            st.plotly_chart(
                plot_spectrum_heatmap(spec_filter, "2. Фильтр спада амплитуды 1/f^β", "Plasma"),
                use_container_width=True
            )
            st.caption(f"Фильтр спадания частот при β = {beta:.1f}. Высокие частоты (по краям) сильно ослабляются.")

        with col_s3:
            st.plotly_chart(
                plot_spectrum_heatmap(spec_filtered, "3. Отфильтрованный спектр"),
                use_container_width=True
            )
            st.caption("Результат умножения. Энергия сосредоточена на низких частотах (в центре), формирующих горы.")

    else:
        st.info("Визуализация частотных спектров доступна только в режиме «Спектральный метод». "
                "В аналитическом режиме мы напрямую суммируем заданный набор гармоник.")

with tab_math:
    st.markdown('<div class="math-card">', unsafe_allow_html=True)
    st.markdown("""
    ### 📓 Математический фундамент проекта

    #### 1. Представление ландшафта через ряды Фурье
    В основе генерации лежит спектральное разложение функции высоты рельефа $Z(x, y)$. Согласно теории Фурье, любая достаточно гладкая функция на ограниченной области может быть представлена в виде двойного ряда Фурье:

    $$Z(x, y) = \\sum_{n=-\\infty}^{\\infty} \\sum_{m=-\\infty}^{\\infty} C_{nm} e^{i (k_x x + k_y y)}$$

    где $C_{nm}$ — комплексные спектральные коэффициенты (амплитуды гармоник), а $\\vec{k} = (k_x, k_y)$ — волновой вектор пространственной частоты.

    Каждая отдельная синусоида (гармоника) задает волновой фронт:

    $$w_i(x, y) = A_i \\cdot \\sin(k_{xi} x + k_{yi} y + \\phi_i)$$

    где:
    * $\\vec{k}_i = (k_{xi}, k_{yi})$ — волновой вектор, определяющий направление распространения волны и ее пространственную частоту (длина вектора $|\\vec{k}_i| = 2\\pi/\\lambda$).
    * $\\phi_i$ — фазовый сдвиг, определяющий смещение холмов.
    * $A_i$ — амплитуда, задающая высоту волны.

    #### 2. Закон $1/f^\\beta$ (фрактальный шум)
    Чтобы поверхность выглядела реалистично, амплитуда гармоники должна уменьшаться при увеличении частоты (уменьшении длины волны). Это реализуется законом:

    $$A_i = \\frac{1}{|\\vec{k}_i|^\\beta}$$

    Где спектральный индекс $\\beta$ отвечает за фрактальную шероховатость.

    #### 3. Быстрое преобразование Фурье (FFT)
    Спектральный метод использует **двумерное дискретное преобразование Фурье (2D DFT)** для эффективного вычисления рядов Фурье на сетке:

    $$Z(x_m, y_n) = \\text{Re}\\left(\\text{IFFT2}\\left(\\text{FFT2}(W) \\cdot H(u, v)\\right)\\right)$$

    где $W$ — матрица белого шума, $H(u, v) = 1 / (u^2 + v^2)^{\\beta/2}$ — частотный фильтр, ослабляющий высокие пространственные частоты. Алгоритм FFT позволяет вычислить преобразование за $O(N^2 \\log N)$ вместо наивных $O(N^4)$, что делает генерацию практически мгновенной.

    #### 4. Дискретное дифференцирование и освещение Ламберта
    Для визуального отображения рельефа рассчитываются тени. Сила освещения в каждой вершине определяется углом падения солнечного луча.

    Вектор нормали к поверхности $Z(x, y)$ в каждой точке находится как:

    $$\\vec{n} = \\left(-\\frac{\\partial Z}{\\partial x}, -\\frac{\\partial Z}{\\partial y}, 1\\right)$$

    Для дискретной карты высот частные производные вычисляются методом центральных разностей:

    $$\\frac{\\partial Z}{\\partial x} \\approx \\frac{Z(x+dx, y) - Z(x-dx, y)}{2dx}$$

    $$\\frac{\\partial Z}{\\partial y} \\approx \\frac{Z(x, y+dy) - Z(x, y-dy)}{2dy}$$

    Затем нормаль нормируется до единичной длины: $\\vec{n}_{unit} = \\frac{\\vec{n}}{|\\vec{n}|}$.

    Интенсивность освещенности $I$ по закону Ламберта при направлении света $\\vec{L}$ рассчитывается через скалярное произведение:

    $$I = \\max(0, \\vec{n}_{unit} \\cdot \\vec{L})$$

    Финальный цвет вершины вычисляется как произведение цвета биома (пляж, лес, гора) на интенсивность $I$ с небольшим рассеянным светом (ambient):

    $$Color_{final} = Color_{biome} \\cdot (ambient + (1 - ambient) \\cdot I)$$

    #### 4. Бесшовная сшивка чанков
    Мир разбивается на квадратные чанки. Радиус $R$ задаёт область $(2R-1)\\times(2R-1)$ чанков вокруг центрального. Бесшовность достигается тем, что вся область строится как единая поверхность, которая затем нарезается с шагом $(N-1)$, где $N$ — разрешение чанка. Благодаря этому соседние чанки разделяют ровно одну строку и один столбец точек, и значения высот на стыке совпадают тождественно — швов не возникает.
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
