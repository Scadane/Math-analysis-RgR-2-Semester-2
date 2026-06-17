import streamlit as st
import numpy as np
import plotly.graph_objects as go
from landscape_synthesis import (
    generate_fft_terrain,
    generate_analytical_terrain,
    calculate_normals,
    calculate_lighting,
    get_shaded_terrain_colors
)

# Настройка страницы
st.set_page_config(
    page_title="Процедурный ландшафт Фурье",
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
st.markdown('<div class="main-title">🏔️ Спектральный синтез 3D-ландшафтов</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Интерактивный генератор рельефа на основе рядов и преобразований Фурье</div>', unsafe_allow_html=True)

# Инициализация сессии для случайного Seed
if "seed" not in st.session_state:
    st.session_state.seed = 42

# Функция генерации случайного сида
def randomize_seed():
    st.session_state.seed = int(np.random.randint(1, 9999))

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
    ["Быстрое преобразование Фурье (2D FFT)", "Аналитический метод гармоник"]
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

# Специфичные параметры для методов
if mode == "Быстрое преобразование Фурье (2D FFT)":
    st.sidebar.markdown("---")
    st.sidebar.subheader("📐 Параметры FFT")
    grid_size = st.sidebar.selectbox("Разрешение сетки:", [64, 128, 256], index=1)
    
    # Генерация данных FFT
    @st.cache_data
    def get_fft_data(size, b, s):
        # Внутри вызываем шаги для построения спектров
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
        
        Z = np.real(np.fft.ifft2(np.fft.ifftshift(H)))
        z_min, z_max = Z.min(), Z.max()
        if z_max > z_min:
            Z = (Z - z_min) / (z_max - z_min)
        else:
            Z = np.zeros_like(Z)
            
        # Спектры (логарифмическая шкала для визуализации)
        log_spec_noise = np.log10(np.abs(F_shifted) + 1e-6)
        log_filter = np.log10(filter_h + 1e-6)
        log_spec_filtered = np.log10(np.abs(H) + 1e-6)
        
        return Z, log_spec_noise, log_filter, log_spec_filtered
        
    Z, spec_noise, spec_filter, spec_filtered = get_fft_data(grid_size, beta, st.session_state.seed)
    
    # Координаты для 3D
    x = np.arange(grid_size)
    y = np.arange(grid_size)
    X, Y = np.meshgrid(x, y)
    dx = 1.0
    dy = 1.0

else:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📐 Параметры гармоник")
    num_harmonics = st.sidebar.slider("Количество гармоник (K):", min_value=5, max_value=120, value=50, step=5)
    
    st.sidebar.markdown("**Координаты чанка:**")
    col_chunk_x, col_chunk_y = st.sidebar.columns(2)
    with col_chunk_x:
        offset_x = st.number_input("Chunk X:", value=0, step=1)
    with col_chunk_y:
        offset_y = st.number_input("Chunk Y:", value=0, step=1)
        
    chunk_size_spatial = 10.0
    grid_size = 100
    
    # Расчет глобальных координат
    x_start = offset_x * chunk_size_spatial
    y_start = offset_y * chunk_size_spatial
    
    x = np.linspace(x_start, x_start + chunk_size_spatial, grid_size)
    y = np.linspace(y_start, y_start + chunk_size_spatial, grid_size)
    X, Y = np.meshgrid(x, y)
    
    # Генерация данных аналитически
    @st.cache_data
    def get_analytical_data(coords_x, coords_y, b, harm, s):
        return generate_analytical_terrain(coords_x, coords_y, beta=b, num_harmonics=harm, seed=s)
        
    Z = get_analytical_data(X, Y, beta, num_harmonics, st.session_state.seed)
    
    dx = chunk_size_spatial / (grid_size - 1)
    dy = chunk_size_spatial / (grid_size - 1)

# Получаем финальные затененные цвета
normals = calculate_normals(Z, dx=dx, dy=dy)
intensity = calculate_lighting(normals, light_dir)
colors = get_shaded_terrain_colors(Z, light_dir=light_dir, is_smooth=smooth_biomes)

# Переводим в rgb строки для Plotly
c_int = (colors * 255).astype(int)
rgb_strings = np.char.add(
    np.char.add(
        np.char.add(
            np.char.add("rgb(", c_int[..., 0].astype(str)), 
            ","
        ), 
        np.char.add(c_int[..., 1].astype(str), ",")
    ),
    np.char.add(c_int[..., 2].astype(str), ")")
)

# Вкладки на основном экране
tab_3d, tab_spectra, tab_math = st.tabs(["🖥️ Интерактивный 3D-рендер", "📊 Фильтрация спектров Фурье", "📖 Математическая справка"])

with tab_3d:
    st.markdown('<div class="math-card">', unsafe_allow_html=True)
    st.markdown(f"**Текущий режим:** {mode}. "
                f"Высота вершин отмасштабирована в {z_scale} раз для наглядности рельефа. "
                "Вы можете свободно вращать, приближать и исследовать ландшафт с помощью мыши.", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Построение 3D графика в Plotly
    fig3d = go.Figure(data=[go.Surface(
        x=X, y=Y, z=Z * z_scale,
        surfacecolor=rgb_strings,
        showscale=False,
        hoverinfo='none'
    )])
    
    fig3d.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis=dict(title='X', showgrid=False, showticklabels=False),
            yaxis=dict(title='Y', showgrid=False, showticklabels=False),
            zaxis=dict(title='Высота Z', showgrid=False, showticklabels=False),
            aspectratio=dict(x=1, y=1, z=0.45)
        ),
        height=650
    )
    
    st.plotly_chart(fig3d, width="stretch")

with tab_spectra:
    if mode == "Быстрое преобразование Фурье (2D FFT)":
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
                width="stretch"
            )
            st.caption("Амплитуды частот распределены равномерно по всем направлениям и масштабам.")
            
        with col_s2:
            st.plotly_chart(
                plot_spectrum_heatmap(spec_filter, "2. Фильтр спада амплитуды 1/f^β", "Plasma"),
                width="stretch"
            )
            st.caption(f"Фильтр спадания частот при β = {beta:.1f}. Высокие частоты (по краям) сильно ослабляются.")
            
        with col_s3:
            st.plotly_chart(
                plot_spectrum_heatmap(spec_filtered, "3. Отфильтрованный спектр"),
                width="stretch"
            )
            st.caption("Результат умножения. Энергия сосредоточена на низких частотах (в центре), формирующих горы.")
            
    else:
        st.info("Визуализация частотных спектров доступна только в режиме 'Быстрое преобразование Фурье (2D FFT)'. "
                "В аналитическом режиме мы напрямую суммируем заданный набор гармоник.")

with tab_math:
    st.markdown('<div class="math-card">', unsafe_allow_html=True)
    st.markdown("""
    ### 📓 Математический фундамент проекта
    
    #### 1. Двумерные волны и ряды
    В основе генерации лежит разложение функции высоты рельефа $Z(x, y)$ в спектр. Каждая отдельная синусоида (гармоника) задает волновой фронт с определенной пространственной частотой:
    
    $$w_i(x, y) = A_i \cdot \sin(k_{xi} x + k_{yi} y + \phi_i)$$
    
    где:
    * $\vec{k}_i = (k_{xi}, k_{yi})$ — волновой вектор, определяющий направление распространения волны и ее пространственную частоту (длина вектора $|\vec{k}_i| = 2\pi/\lambda$).
    * $\phi_i$ — фазовый сдвиг, определяющий смещение холмов.
    * $A_i$ — амплитуда, задающая высоту волны.
    
    #### 2. Закон $1/f^\beta$ (Шум с фиолетовым/розовым спектром)
    Чтобы поверхность выглядела реалистично, амплитуда гармоники должна уменьшаться при увеличении частоты (уменьшении длины волны). Это реализуется законом:
    
    $$A_i = \frac{1}{|\vec{k}_i|^\beta}$$
    
    Где спектральный индекс $\beta$ отвечает за фрактальную шероховатость.
    
    #### 3. Дискретное дифференцирование и освещение Ламберта
    Для визуального отображения рельефа рассчитываются тени. Сила освещения в каждой вершине определяется углом падения солнечного луча.
    
    Вектор нормали к поверхности $Z(x, y)$ в каждой точке находится как:
    
    $$\vec{n} = \left(-\frac{\partial Z}{\partial x}, -\frac{\partial Z}{\partial y}, 1\right)$$
    
    Для дискретной карты высот частные производные вычисляются методом центральных разностей:
    
    $$\frac{\partial Z}{\partial x} \approx \frac{Z(x+dx, y) - Z(x-dx, y)}{2dx}$$
    
    $$\frac{\partial Z}{\partial y} \approx \frac{Z(x, y+dy) - Z(x, y-dy)}{2dy}$$
    
    Затем нормаль нормируется до единичной длины: $\vec{n}_{unit} = \frac{\vec{n}}{|\vec{n}|}$.
    
    Интенсивность освещенности $I$ по закону Ламберта при направлении света $\vec{L}$ рассчитывается через скалярное произведение:
    
    $$I = \max(0, \vec{n}_{unit} \cdot \vec{L})$$
    
    Финальный цвет вершины вычисляется как произведение цвета биома (пляж, лес, гора) на интенсивность $I$ с небольшим рассеянным светом (ambient):
    
    $$Color_{final} = Color_{biome} \cdot (ambient + (1 - ambient) \cdot I)$$
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
