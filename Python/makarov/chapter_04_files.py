"""Working with files in Google Colab."""

# ## Работа с файлами в Google Colab

# ### Этап 1. Подгрузка файлов

# Способ 1. Вручную через вкладку 'Файлы'

# +
# см. материалы урока на сайте
# -

# Способ 2. Через модуль files библиотеки google.colab

# +
# выполняем все необходимы импорты
import os

import pandas as pd
import seaborn as sns
from google.colab import files

# импортируем логистическую регрессию из модуля linear_model библиотеки sklearn
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# -

# создаем объект этого класса, применяем метод .upload()
uploaded: dict[str, bytes] = files.upload()

# +
# посмотрим на содержимое словаря uploaded
# uploaded
# -

# ### Этап 2. Чтение файлов

# #### Просмотр содержимого папки /content/

# ##### Модуль os и метод .walk()

# выводим пути к папкам (dirpath) и наименования файлов (filenames)
# и после этого
for dirpath, _, filenames in os.walk("/content/"):

    # во вложенном цикле проходимся по названиям файлов
    for filename in filenames:

        # и соединяем путь до папок и входящие в эти папки файлы
        # с помощью метода path.join()
        print(os.path.join(dirpath, filename))

# ##### Команда `!ls`

# посмотрим на содержимое папки content
# !ls

# заглянем внутрь sample_data
# !ls /content/sample_data/

# #### Чтение из переменной uploaded

# посмотрим на тип значений словаря uploaded
type(uploaded["test.csv"])

# Пример работы с объектом bytes

# +
# обратимся к ключу словаря uploaded и применим метод .decode()
uploaded_str: str = uploaded["test.csv"].decode()

# на выходе получаем обычную строку
print(type(uploaded_str))
# -

# выведем первые 35 значений
print(uploaded_str[:35])

# +
# если разбить строку методом .split() по символам \r
# (возврат к началу строки) и \n (новая строка)
uploaded_list: list[str] = uploaded_str.split("\r\n")

# на выходе мы получим список
type(uploaded_list)
# -

# пройдемся по этому списку, не забыв создать индекс
# с помощью функции enumerate()
for i, line in enumerate(uploaded_list):

    # начнем выводить записи
    print(line)

    # когда дойдем до четвертой строки
    if i == 3:

        # прервемся
        break

# #### Использование функции open() и конструкции with open()

# +
# передадим функции open() адрес файла
# параметр 'r' означает, что мы хотим прочитать (read) файл
# f1: TextIO = open("/content/train.csv")

# метод .read() помещает весь файл в одну строку
# выведем первые 142 символа (если параметр не указывать,
# выведется все содержимое)
# print(f1.read(142))

# в конце файл необходимо закрыть
# f1.close()

# учитывая требования линтеров код был скорретирован
# следующим образом:
with open("file.txt", encoding="utf-8") as f1:
    data = f1.read()

# +
# снова откроем файл
# f2: TextIO = open("/content/train.csv")
with open("/content/train.csv", encoding="utf-8") as f2:

    # пройдемся по нашему объекту в цикле for и параллельно создадим индекс
    for i, line in enumerate(f2):

        # выведем строки без служебных символов по краям
        print(line.strip())

        # дойдя до четвертой строки, прервемся
        if i == 3:
            break

# не забудем закрыть файл
# f2.close()
# -

# скажем Питону: "открой файл  и назови его f3"
with open("/content/test.csv", encoding="utf-8") as f3:

    # "пройдись по строкам без служебных символов"
    for i, line in enumerate(f3):
        print(line.strip())

        # и "прервись на четвертой строке"
        if i == 3:
            break

# #### Чтение через библиотеку Pandas

# применим функцию read_csv() и посмотрим
# на первые три записи файла train.csv
train: pd.DataFrame = pd.read_csv("/content/train.csv")
train.head(3)

# сделаем то же самое с файлом test.csv
test: pd.DataFrame = pd.read_csv("/content/test.csv")
test.head(3)

# ### Этап 3. Построение модели и прогноз

# #### **Шаг 1**. Обработка и анализ данных

# Исследовательский анализ данных (EDA)

# посмотрим на данные в целом
train.info()

# посмотрим насколько значим класс билета для выживания пассажира
# с помощью x и hue мы можем уместить две категориальные переменные
# на одном графике
sns.countplot(x="Pclass", hue="Survived", data=train)

# ![image.png](attachment:image.png)

# Пропущенные значения

# выявим пропущенные значения с помощью .isnull()
# и посчитаем их количество через sum()
train.isnull().sum()

# переменная Cabin (номер каюты), скорее всего, не является самой важной
# избавимся от нее с помощью метода .drop()
# (параметр axis = 1 отвечает за столбцы, inplace = True сохраняет изменения)
train.drop(columns="Cabin", axis=1, inplace=True)

# а вот Age (возраст) скорее важен, заменим пустые значения
# средним арифметическим
train["Age"] = train["Age"].fillna(train["Age"].mean())

# у нас остаются две пустые строки в Embarked, удалим их
train.dropna(inplace=True)

# посмотрим на результат
train.isnull().sum()

# Категориальные переменные

# применим one-hot encoding к переменной Sex (пол)
# с помощью функции pd.get_dummies()
pd.get_dummies(train["Sex"]).head(3)

# снова скачаем столбец Sex из датасета train в формате датафрейма
previous: pd.DataFrame = pd.read_csv("/content/train.csv")[["Sex"]]
previous.head()

# закодируем переменную через 0 и 1
pd.get_dummies(previous["Sex"], dtype=int).head(3)

# удалим первый столбец, он избыточен
sex: pd.DataFrame = pd.get_dummies(train["Sex"], drop_first=True)
sex.head(3)

# сделаем то же самое для переменных Pclass и Embarked
embarked: pd.DataFrame = pd.get_dummies(train["Embarked'"], drop_first=True)
pclass: pd.DataFrame = pd.get_dummies(train["Pclass"], drop_first=True)

# присоединим закодированные через one-hot encoding переменные
# к исходному датафрейму через функцию .concat()
train = pd.concat([train, pclass, sex, embarked], axis=1)
train.head(3)

# Отбор признаков

# удалим те столбцы, которые нам теперь не нужны
id_columns = ["PassengerId", "Name", "Ticket"]
categorical_columns = ["Pclass", "Sex", "Embarked"]
columns_to_drop = id_columns + categorical_columns
train.drop(columns_to_drop, axis=1, inplace=True)
train.head(3)

# Нормализация данных

# +
# создадим объект этого класса
scaler: StandardScaler = StandardScaler()

# выберем те столбцы, которые мы хотим масштабировать
cols_to_scale: list[str] = ["Age", "Fare"]

# рассчитаем среднее арифметическое и СКО для масштабирования данных
scaler.fit(train[cols_to_scale])

# применим их
train[cols_to_scale] = scaler.transform(train[cols_to_scale])

# посмотрим на результат
train.head(3)
# -

# некоторые названия столбцов теперь представляют собой числа,
# так быть не должно
train.columns
