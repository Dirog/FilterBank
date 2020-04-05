# Банк цифровых фильтров (Filter Bank)
Данная программа реализует алгоритм вычисления F канального [банка фильтров](https://en.wikipedia.org/wiki/Filter_bank) для C физических каналов. Алгоритм основан на применении быстрого преобразования Фурье в режиме “скользящего окна” размера F. Длина импульсной характеристики фильтров T кратна F. Шаг скользящего окна K задаётся, и он определяет частоту дискретизации для каждого канала. Вычисления распараллелены и производится на графическом процессоре. Используется фреймворк CUDA и библиотека cuFFT.
