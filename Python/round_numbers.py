"""Round numbers."""

import random

# +
number_1 = random.random()  # генерирование числа от 0 до 1
number_1 = round(number_1, 2)  # округление значения сразу в переменной

print(number_1)  # вывод на печать уже округленного числа


# -


def generate_report(rounded_number: float) -> None:
    """Формирует отчет с использованием переданного числа.

    Args:
        rounded_number (float): Округленное число для отчета.

    Returns:
        None
    """
    report = f"Отчет о вычислениях (округлено): {rounded_number}"
    print(report)
