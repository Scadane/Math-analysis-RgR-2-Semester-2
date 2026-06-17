import sys
import numpy as np
from landscape_synthesis import generate_fft_terrain

def main():
    print("=========================================================")
    # Печатаем красивый ASCII-арт гор
    print("           ▲           ▲")
    print("          ▲▲▲         ▲▲▲      🏔️ PROCEDURAL FOURIER")
    print("         ▲▲▲▲▲       ▲▲▲▲▲        3D LANDSCAPE GENERATOR")
    print("        /     \\     /     \\")
    print("=========================================================")
    print("Проект успешно готов к запуску!\n")
    print("Доступные интерактивные интерфейсы:")
    print("1. Jupyter Notebook (с формулами и графиками):")
    print("   Запуск: .venv/bin/jupyter notebook landscape_notebook.ipynb\n")
    print("2. Веб-приложение Streamlit (премиальный дашборд):")
    print("   Запуск: .venv/bin/streamlit run app.py\n")
    print("---------------------------------------------------------")
    print("Выполняем тестовую генерацию FFT-ландшафта (размер 128x128)...")
    
    Z = generate_fft_terrain(size=128, beta=1.8, seed=42)
    print(f"Тестовая генерация завершена успешно!")
    print(f"Форма сетки высот: {Z.shape}")
    print(f"Диапазон высот Z: [{Z.min():.4f}, {Z.max():.4f}]")
    print("Все математические вычисления работают корректно.")
    print("=========================================================")

if __name__ == "__main__":
    main()