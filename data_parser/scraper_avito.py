import os
import time
import random
import csv
from collections import defaultdict
from urllib.parse import urljoin
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup


class AvitoParser:
    def __init__(self, base_url, output_dir="output", headless=False):
        print(f"Инициализация парсера для {base_url}")
        self.base_url = base_url
        self.output_dir = output_dir
        self.driver = self._init_driver(headless)
        self.parsed_ads = set()
        os.makedirs(output_dir, exist_ok=True)
        print(f"Выходная директория: {os.path.abspath(output_dir)}")

    def _init_driver(self, headless):
        """Инициализация ChromeDriver с настройками против обнаружения"""
        chrome_options = Options()

        if headless:
            chrome_options.add_argument("--headless=new")

        # Настройки против обнаружения
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_argument("--start-maximized")
        chrome_options.add_argument("--disable-infobars")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36")

        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option("useAutomationExtension", False)

        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)

        # Маскировка WebDriver
        driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
            "source": """
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                })
                window.navigator.chrome = {
                    runtime: {},
                };
            """
        })
        return driver

    def _random_delay(self, min_sec=2, max_sec=5):
        """Случайная задержка между действиями"""
        delay = random.uniform(min_sec, max_sec)
        print(f"Ожидание {delay:.2f} секунд...")
        time.sleep(delay)

    def load_page(self, url):
        """Загрузка страницы с обработкой ошибок"""
        print(f"Загрузка страницы: {url}")
        try:
            self.driver.get(url)
            self._random_delay(3, 7)
            return True
        except Exception as e:
            print(f"Ошибка загрузки страницы: {e}")
            return False

    def _check_captcha(self):
        """Проверка наличия капчи"""
        try:
            return any([
                self.driver.find_element(By.CSS_SELECTOR, "div.captcha__container").is_displayed(),
                "captcha" in self.driver.page_source.lower(),
                "анти-робот" in self.driver.page_source.lower()
            ])
        except:
            return False

    def parse_listing_page(self, url):
        """Парсинг страницы со списком объявлений"""
        try:
            if not self.load_page(url):
                return None

            # Проверка капчи
            if self._check_captcha():
                print("Обнаружена капча! Пожалуйста:")
                print("1. Решите капчу вручную в браузере")
                print("2. Смените IP-адрес")
                print("3. Подождите 10-15 минут")
                input("Нажмите Enter после решения капчи...")
                self._random_delay(5, 10)

            # Ожидание загрузки объявлений
            list=WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'div[data-marker="item"]')))
            print(list.text)
            # Прокрутка для загрузки всех объявлений
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            self._random_delay(1, 3)

            return self.driver.page_source
        except Exception as e:
            print(f"Ошибка парсинга страницы: {e}")
            return None


    def parse_ad_page(self, url):
        """Парсинг страницы конкретного объявления с сохранением всех параметров"""
        try:
            if not self.load_page(url):
                return None

            # Проверка капчи
            if self._check_captcha():
                print("Обнаружена капча! Пожалуйста:")
                print("1. Решите капчу вручную в браузере")
                print("2. Смените IP-адрес")
                print("3. Подождите 10-15 минут")
                input("Нажмите Enter после решения капчи...")
                self._random_delay(5, 10)

            # Ожидание загрузки основной информации
            main_info_place = WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.XPATH , '//div[@data-marker="item-view/item-params"]')))
            main_info_building = self.driver.find_element(By.XPATH, '//div[@data-marker="item-view/item-params"]')
            price = self.driver.find_element(By.XPATH, '//span[@itemprop="price"]')
            address = self.driver.find_element(By.XPATH, '//span[@class="style-item-address__string-wt61A"]')
            coordinates = self.driver.find_element(By.XPATH, '//div[@class="style-item-map-wrapper-ElFsX '
                                                             'style-expanded-x335n"]')


            # Базовые данные
            base_data = {
                'price': price.get_attribute("content"),
                'address': address.text,
                'coordinates_lat': coordinates.get_attribute("data-map-lat"),
                'coordinates_lon': coordinates.get_attribute("data-map-lon")
            }

            print(base_data)

            # Параметры помещения
            property_params = self._extract_property_params(soup)

            # Объединяем данные
            ad_data = {**base_data, **property_params}

            return ad_data

        except Exception as e:
            print(f"Ошибка парсинга объявления {url}: {e}")
            return None

    def save_to_dataframe(self, ads_list):
        """
        Сохраняет список объявлений в DataFrame и CSV
        автоматически обрабатывая все возможные параметры
        """
        if not ads_list:
            print("Нет данных для сохранения")
            return None

        # Создаем DataFrame
        df = pd.DataFrame(ads_list)

        # Заполняем пропущенные значения
        df = df.fillna('')

        # Сохраняем в CSV
        csv_path = os.path.join(self.output_dir, 'ads_data.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"Данные сохранены в {csv_path}")

        return df

    def run(self, max_ads=50, ads_per_file=20):
        """Основной метод запуска парсера"""
        print(f"\nЗапуск парсера (максимум {max_ads} объявлений)")
        total_ads = 0
        page = 1
        all_ads = []

        try:
            while total_ads < max_ads:
                print(f"\nСтраница {page}:")
                url = f"{self.base_url}&p={page}"
                html = self.parse_listing_page(url)

                if not html:
                    print("Не удалось получить данные, остановка")
                    break

                ad_links = self.extract_ad_links(html)
                if not ad_links:
                    print("Объявления не найдены, остановка")
                    break

                for link in ad_links:
                    if total_ads >= max_ads:
                        break

                    print(f"\nПарсинг объявления {total_ads + 1}/{max_ads}")
                    ad_data = self.parse_ad_page(link)

                    if ad_data:
                        all_ads.append(ad_data)
                        total_ads += 1
                        self._random_delay(3, 6)

                if total_ads >= max_ads:
                    break

                page += 1
                self._random_delay(5, 10)

            # Сохраняем все данные в DataFrame и CSV
            if all_ads:
                df = self.save_to_dataframe(all_ads)
                print("\nСтатистика собранных данных:")
                print(df.info())
                print("\nПример данных:")
                print(df.head())

            print(f"\nГотово! Всего собрано {total_ads} объявлений")
            return df

        except KeyboardInterrupt:
            print("\nПарсер остановлен пользователем")
            if all_ads:
                return self.save_to_dataframe(all_ads)
        except Exception as e:
            print(f"\nКритическая ошибка: {e}")
            if all_ads:
                return self.save_to_dataframe(all_ads)
        finally:
            self.driver.quit()


if __name__ == "__main__":
    # URL для коммерческой недвижимости в Казани
    url = "https://www.avito.ru/kazan/kommercheskaya_nedvizhimost/sdam-ASgBAgICAUSwCNRW?cd=1&f=ASgBAQECAkSwCNRW9BKk2gECQJ7DDTSI2TmG2TmK2TmI9BE0zIGLA8qBiwPIgYsDAkW2ExZ7ImZyb20iOm51bGwsInRvIjoxNTB9gqESHSLQvtGC0LTQtdC70YzQvdGL0Lkg0LLRhdC~0LQi&s=104"

    parser = AvitoParser(url)
    df = parser.run(max_ads=10)  # Парсим 10 объявлений

    if df is not None:
        # Можно работать с DataFrame
        if 'Общая площадь' in df.columns:
            try:
                avg_area = pd.to_numeric(df['Общая площадь'], errors='coerce').mean()
                print("\nСредняя площадь помещений:", avg_area)
            except:
                print("\nНе удалось вычислить среднюю площадь")