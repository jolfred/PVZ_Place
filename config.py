import os
from dotenv import load_dotenv

# Загрузка переменных окружения из .env файла
load_dotenv()

# Загрузка переменных окружения
YANDEX_MAPS_API_KEY = os.getenv('YANDEX_MAPS_API_KEY', '')
if not YANDEX_MAPS_API_KEY:
    raise ValueError('Необходимо указать YANDEX_MAPS_API_KEY в файле .env')

# Настройки приложения
class Config:
    SECRET_KEY = os.urandom(24)
    YANDEX_MAPS_API_KEY = YANDEX_MAPS_API_KEY 