"""Round numbers."""

import io
import random
import sys

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


# +
def test_generate_report_prints_correct_output() -> None:
    """Тестирует корректность вывода отчета."""
    captured_output = io.StringIO()  # Создаем буфер для перехвата stdout
    sys.stdout = captured_output  # Перенаправляем stdout в буфер

    generate_report(0.57)  # Вызываем функцию с тестовым значением

    sys.stdout = sys.__stdout__  # Возвращаем stdout обратно
    output = captured_output.getvalue()  # Получаем текст, который был выведен

    assert "Отчет о вычислениях (округлено): 0.57" in output

    print("test_generate_report_prints_correct_output — PASSED")


def test_generate_report_with_zero() -> None:
    """Тестирует функцию при передаче нуля."""
    captured_output = io.StringIO()
    sys.stdout = captured_output

    generate_report(0.0)

    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()

    assert "Отчет о вычислениях (округлено): 0.0" in output

    print("test_generate_report_with_zero — PASSED")
