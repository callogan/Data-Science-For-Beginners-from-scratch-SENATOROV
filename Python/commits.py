"""Ответы на вопросы по коммитам."""

# 1. Опишите своими словами назначение каждого из этих типов коммитов:
#
#     ```bash
#     - feat - коммит, который добавляет определённую фичу в наш код;
#     ```
#     ```bash 
#     - fix - коммит, который исправляет баг в нашем коде; 
#     ```
#     ```bash 
#     - docs - коммит, указывающий на изменения, связанные с документацией проекта; 
#     ```
#     ```bash 
#     - style - коммит, обозначающий изменения, связанные с оформлением кода (не влияя не его логику); 
#     ```
#     ```bash 
#     - refactor - коммит, указывающий на формальное изменение кода без изменения его логики
#     (например, разделение больших функций на маленькие, улучшение алгоритмов и т.п.); 
#     ```
#     ```bash 
#     - test - коммит, обозначающий изменения, связанные с тестированием кода; 
#     ```
#     ```bash 
#     - build - коммит связан с изменениями, которые влияют на процесс сборки проекта или его зависимости; 
#     ```
#     ```bash 
#     - ci - коммит связан с изменениями в процессах непрерывной интеграции и развертывания (CI/CD); 
#     ```
#     ```bash 
#     - perf - коммит улучшает скорость работы или эффективность использования ресурсов
#     (например, оптимизация алгоритмов, снижение потребление памяти и т.п.);
#     ```
#     ```bash 
#     - chore - коммит используется для решения технических задач, которые не влияют на код приложения и его функциональность (например, обновление зависимостей, очистка ненужных файлов и т.п.).`
#     ```

# 2. Представьте, что вы исправили баг в функции, которая некорректно округляет числа. Сделайте фиктивный коммит и напишите для него сообщение в соответствии с Conventional Commits (используя тип fix).
#
#     ```bash
#     git commit -m "fix: correct rounding issue in calculate_total function"
#     ```

# 3. Добавление новой функциональности:
# Допустим, вы реализовали новую функцию generateReport в проекте. Сделайте фиктивный коммит с типом feat, отражающий добавление этой функциональности.
#
#     ```bash
#     git commit -m "feat: add generateReport function to create detailed reports"
#     ```

# 4. Модификация формата кода или стилей docs:
# Представьте, что вы поправили отступы и форматирование во всём проекте, не меняя логики кода. Сделайте фиктивный коммит с типом style.
#
#     ```bash
#     git commit -m "style: fixed indentation and formatting across the project"
#     ```

# 5. Документация и тестирование:
# - Сделайте фиктивный коммит с типом, добавляющий или улучшающий документацию для вашей новой функции.
#
#     ```bash
#     git commit -m "docs: add documentation for generateReport function"
#     ```
#
# - Сделайте фиктивный коммит с типом test, добавляющий тесты для этой же функции.
#
#     ```bash
#     git commit -m "test: add unit tests for generateReport function"
#     ```
