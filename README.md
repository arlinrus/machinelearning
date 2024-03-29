# machinelearning

# Направления методов машинного обучения: методов главных компонентов 
МГК - это уменьшение размерности с минимальными потерями в информативности 

Рассмотрим пример с таблицей:

<img width="484" alt="Снимок экрана 2024-01-10 в 15 23 54" src="https://github.com/arlinrus/machinelearning/assets/111064731/a45e8110-171c-4395-bd90-28ece3ca447d">

Для начала найдем среднее арифметичнское для каждого из объектов и произведем центрирование для объектов x1 и x2

<img width="769" alt="Снимок экрана 2024-01-10 в 15 29 06" src="https://github.com/arlinrus/machinelearning/assets/111064731/0abc8dbe-f3d2-4a51-8545-57f527df92f9">

Где новые координат векторов будут иметь данные заданные значения: 0,591 и 0,807 или тоже самое с минусами.


venv/od_1_ml.py

# Линейная регрессия(изучение с учителем)

***Что такое линейная регрессия простыми словами?
Линейная регрессия — это метод анализа данных, который предсказывает ценность неизвестных данных с помощью другого связанного и известного значения данных. Он математически моделирует неизвестную или зависимую переменную и известную или независимую переменную в виде линейного уравнения.***

Данные о проведенном времени в мгз

<img width="373" alt="Снимок экрана 2024-01-16 в 17 34 11" src="https://github.com/arlinrus/machinelearning/assets/111064731/c95dc953-69a9-4b77-994d-fb9b919e7be4">

Выберем переменные:

Номер наблюдения в мрдели не участвует - наблюдение

Предиктор - колво товаров(х1)

Отклик - время в мгх(y)
<img width="600" alt="Снимок экрана 2024-01-16 в 17 36 23" src="https://github.com/arlinrus/machinelearning/assets/111064731/4a49a141-03be-47bc-99df-efb57578ad01">


<img width="1052" alt="Снимок экрана 2024-01-16 в 17 40 42" src="https://github.com/arlinrus/machinelearning/assets/111064731/8fec1e08-a3e4-4355-bf96-2914ed511904">


<img width="693" alt="Снимок экрана 2024-01-16 в 17 37 36" src="https://github.com/arlinrus/machinelearning/assets/111064731/30fcbcc5-a3b3-4500-bef4-9f8efe2d7c34">

Уравнение линейной регрессии имеет следующий вид:

y = 4.06 + 0.93x1

Сколько временинам потребуеся, если мы захотим приобрести 27 товаров?
Подставим все в x1 и получим 29.17

LinearReгрессия соответствует линейной модели с коэффициентами w = (w1,…, wp), чтобы минимизировать остаточную сумму квадратов между наблюдаемыми целями в наборе данных и целями, предсказанными с помощью линейного приближения.

С точки зрения реализации, это просто обычные методы наименьших квадратов (scipy.linalg.lstsq) или неотрицательные наименьшие квадраты (scipy.optimize.nnls), завернутые в объект предиктора.
***Ссылка на изучение sklearn***: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

```from sklearn.linear_model import LinearRegression
import numpy as np
import io #для работы с различными типами ввода и вывода
import pandas as pd
from sklearn.metrics import r2_score

data = '''id,X,Y
1,6,19
2,3,11
3,10,24
4,4,14
5,2,7
6,11,26
7,23,62
8,21,47
9,20,48
10,14,27'''

data = pd.read_csv(io.StringIO(data), index_col = "id")
print(data)

#print(data.Y.mean()) #определили выборочные срение из столбцов
#print(data.X.mean())

X_train = pd.DataFrame(data.X)
Y_train = pd.DataFrame(data.Y)

reg_model = LinearRegression().fit(X_train, Y_train)#обучили модель
#print(reg_model.coef_) #teta 1
#print(reg_model.intercept_)#teta 0

y_predict = reg_model.predict(X_train)
print(y_predict) #to predict we need to make model with training datas

print(r2_score(Y_train, y_predict)) #оценка точности модели
```

# Задача классификаций с учителем
## Метод k ближайших соседей

Пусть X какое то множество. Метрикой на множестве X называют функцию d(x,y): X*X -- R, которая для любых x,y,z удовлетворяет трем свойствам:

1)она неотрицательна(расстяние не может быть отрицательно); 2)она симметрична; 3)выполняется неравенство треугольника(длина любой стороны не больше суммы двух других)

![Снимок экрана (11)](https://github.com/arlinrus/machinelearning/assets/111064731/4d7c3cf7-feaa-428a-8068-0ab7dbaa19b3)

Посмотреть на точки, который находятся рядом с черном объектом. Посмотри ближайших соседей и выбери тот класс, чьих представителей оказалось больше. То есть смотрим на ближайшие объекты и считаем их колличиство. Тех, что больше , к той и относим черный объект.

## Наивный байсовский классификатор
Пусть событие B таково, что P(B)>0. Условной вероятностью события А при условии, что произошло событие B, называется число P(A/B)=P(A пересекается B)/P(B) - формула Байса или P(A/B)=P(B/A)P(A)/P(B)
Хороший пример спам и не спам.

Наивный классификатор:
![Снимок экрана (13)](https://github.com/arlinrus/machinelearning/assets/111064731/fab2b648-13fd-4db8-b0e2-b93b3bee6572)

![Снимок экрана (14)](https://github.com/arlinrus/machinelearning/assets/111064731/84f5f2d5-759f-4196-90e2-bf51acadea09)

## Сглаживание по Лапласу 
Предположить, что мы видели каждое слово на один раз больше.

![Снимок экрана (15)](https://github.com/arlinrus/machinelearning/assets/111064731/c90bc324-a191-47b6-830a-6108c8810d58)




