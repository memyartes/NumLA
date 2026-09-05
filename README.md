# Лабораторная работа №1. PCA

Весь ваш код должен лежать в `src/`. Отчёт — в корне репозитория.

| Часть | Файл |
|:-|:-|
| Реализация PCA | [`src/pca.py`](src/pca.py) |
| Задание I. Морфометрия | [`src/morphometry.py`](src/morphometry.py) |
| Задание II. Собственные лица | [`src/eigenfaces.py`](src/eigenfaces.py) |
| Задание III. Многообразие данных | [`src/manifold.py`](src/manifold.py) |

Публичные тесты: [`tests/test_public.py`](tests/test_public.py).
## Локальное тестирование

Из корня репозитория:

```shell
pip install -r requirements.txt
pytest
```

Тесты нужно запускать из корня репозитория. Отдельные тесты — по имени:
```shell
pytest -k gram
```

### Закрытые тесты

В CI дополнительно запускаются закрытые тесты. Они находятся в специальном Docker-образе и не входят в репозиторий с заданием.

Для локального запуска закрытого чекера требуется установленный Docker. Инструкции по установке доступны в [официальной документации](https://docs.docker.com/engine/install/).

Из корня репозитория выполните:


```shell
docker run --rm \
  -v "$(pwd)/src:/workspace/src:ro" \
  cr.ct.itmo.ru/numla/hw-checker/pca:latest
```

Чекер содержит как публичные, так и закрытые тесты и запускает их вместе.

## Кодстайл

Проверяется `ruff` (неиспользуемые импорты и переменные):

```shell
ruff check src --select F401,F841
```

