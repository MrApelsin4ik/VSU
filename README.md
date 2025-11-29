# Проект «Новый ВГУ»

**Веб-сервис + Telegram-бот для абитуриентов и их семей.**
Помогает структурировать и находить актуальную информацию о новостях, событиях и курсах ВГУ. Встроенный интеллектуальный помощник (LLM) ускоряет и улучшает поиск сложной информации.

---

## Описание проекта

**Цель:**
Повысить доступность и узнаваемость информации для абитуриентов, усилить коммуникацию между университетом и потенциальными студентами, улучшить обратную связь.

**Кто пользуется:**

* Школьники-абитуриенты
* Их родители
* Иностранные студенты

---

## Компоненты проекта

* **Веб-сервис**: новости, события, платформа для курсов
* **Telegram-бот**: оповещения и ответы на вопросы
* **Интеллектуальный помощник**: OpenAI/LLM — «умный поиск» и помощь в сложных вопросах

---

## Проблематика (что решаем)

* **Низкая популярность образования и недостаточная узнаваемость бренда**
* **Слабая работа с абитуриентами**: информирование, вовлечение
* **Отсутствие или недостаточный фидбэк от студентов**

Проект структурирует информацию, делает её доступной и персонализированной, повышает вовлечённость.

---

## Ключевые возможности

* **Поиск и фильтрация актуальных новостей и событий**
* **Публикация и прохождение курсов**: платформа для курсов
* **Telegram-оповещения** о важных событиях
* **Интеллектуальный помощник**, умеющий отвечать на сложные вопросы и делать «смарт-поиск» по базе и внешним источникам (через LLM)
* **Поддержка нескольких категорий пользователей**: школьники, родители, иностранные студенты

---

## Технологии (стек)

* **Backend**: Python, Django, Gunicorn
* **Frontend**: HTML, CSS, JavaScript
* **База данных**: MySQL
* **Telegram**: aiogram
* **Интеграция LLM**: OpenAI (API) для смарт-поиска
* **Дополнительно**: nginx (рекомендуется для продакшена), systemd / Docker (опционально)


# Быстрый запуск и деплой



## 1. Клонирование репозитория

```bash
# в директорию, где хотите хранить проект
git clone https://github.com/MrApelsin4ik/VSU
cd VSU
```

## 2. Виртуальное окружение и зависимости

```bash
# создать и активировать venv
python3 -m venv .venv
source .venv/bin/activate

# установить зависимости
pip install --upgrade pip
pip install -r requirements.txt
```


## 3. Настройка базы данных и статика

```bash
# перейти в папку с manage.py
cd VGU_site

# выполнить миграции
python manage.py migrate

# (если нужно) создать суперпользователя
python manage.py createsuperuser

# собрать статические файлы (для nginx)
python manage.py collectstatic --noinput
```

> Статические файлы будут собраны в `STATIC_ROOT` (проверьте `VGU/settings.py`).

## 4. Запуск локального dev-сервера (быстрый тест)

```bash
# из VGU_site
python manage.py runserver 0.0.0.0:8000
```

## 5. Запуск Telegram-бота

```bash
# в корне проекта
# запустить в текущей сессии
python VGU_tg/tg_main.py

# запустить в фоне (простая опция)
nohup python VGU_tg/tg_main.py > /var/log/vgu_tg.log 2>&1 &
```

## 6. Продакшен: запуск через Gunicorn

```bash
# из корня проекта
gunicorn --bind 0.0.0.0:9000 --access-logfile - --error-logfile - VGU.wsgi:application
```

Рекомендуется запускать gunicorn через systemd (unit-файл) или supervisor для автозапуска и рестартов.

## 7. Запуск кода, который формирует/пишет описания в БД

В репозитории есть скрипт `AI/writing_descr.py`. Для запуска:

```bash
# из корня проекта
source .venv/bin/activate
python AI/writing_descr.py
```

## 8. Пример краткой конфигурации nginx

Сохраняйте как `/etc/nginx/sites-available/vgu` и сделайте symlink в `sites-enabled`.

```
server {
    listen 80;
    server_name example.com; # замените на ваше доменное имя или IP

    # Максимальный размер для загрузок (при необходимости)
    client_max_body_size 50M;

    # Статика (путь укажите тот, который настроен в settings.py)
    location /static/ {
        alias /home/dev/VGU/static/;  # STATIC_ROOT
        expires 30d;
        add_header Cache-Control "public";
    }

    location /media/ {
        alias /home/dev/VGU/media/;   # MEDIA_ROOT
        expires 30d;
        add_header Cache-Control "public";
    }

    # Проксирование запросов к Gunicorn
    location / {
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        proxy_pass http://127.0.0.1:9000;  # gunicorn
        proxy_read_timeout 300;
        proxy_connect_timeout 300;
    }

    # Логи (опционально)
    access_log /var/log/nginx/vgu_access.log;
    error_log  /var/log/nginx/vgu_error.log;
}
```

