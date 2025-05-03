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
        self.current_items_list_page = base_url
        self.output_dir = output_dir
        self.driver = self._init_driver(headless)
        self.parsed_ads = set()
        os.makedirs(output_dir, exist_ok=True)
        self.full_data = pd.DataFrame()
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
            self._random_delay(5, 10)
            return True
        except Exception as e:
            print(f"Ошибка загрузки страницы: {e}")
            return False

    def scroll_items_list_page(self):
        """Прокрутка страницы для подгрузки всех объявлений"""
        print("Прокрутка страницы для полной загрузки...")
        self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")


    def scrap_items_list_page(self):
        """Парсинг страницы со списком объявлений"""
        try:
            if not self.load_page(self.current_items_list_page):
                return None

            # Ожидание загрузки объявлений
            WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'div[data-marker="item"]'))
            )

            # # Прокрутка для загрузки всех объявлений
            # self.scroll_items_list_page()
            return self.driver.page_source

        except Exception as e:
            print(f"Ошибка парсинга страницы со списком объявлений: {e}")
            return None

    def parse_items_list_page(self):
        """Извлечение данных объявлений из HTML"""
        html = self.scrap_items_list_page()


        soup = BeautifulSoup(html, 'html.parser')
        ads = []

        items_list = soup.select('div[data-marker="item"]')
        print(f"Найдено карточек: {len(items_list)}")

        for item in items_list:
            try:
                ads.append({
                    'title': item.select_one('[itemprop="name"]').text.strip(),
                    'price': item.select_one('[itemprop="price"]')['content'],
                    'address': item.select_one('[data-marker="item-address"]').text.strip(),
                    'item_url': urljoin(self.base_url, item.select_one('a[itemprop="url"]')['href']),
                    'date': item.select_one('[data-marker="item-date"]').text.strip()
                })

            except Exception as e:
                print(f"Ошибка в карточке сбор информации с общей страницы объявлений: {e}")
                continue

        return ads



    def _extract_property_params(self, soup):
        """
        Извлекает все доступные параметры помещения в виде словаря
        """
        params = defaultdict(str)

        try:
            params_block = soup.find('div', {'data-marker': 'item-view/item-params'})
            if not params_block:
                return dict(params)

            # Извлекаем все пункты параметров
            items = params_block.find_all('li', class_='params-paramsList__item-_2Y2O')

            for item in items:
                # Разделяем название параметра и значение
                parts = [p.strip() for p in item.get_text(strip=True, separator=':').split(':')]
                if len(parts) >= 2:
                    param_name = parts[0]
                    param_value = ':'.join(parts[1:])  # На случай, если в значении есть двоеточие
                    params[param_name] = param_value.replace('м²', '').strip()

        except Exception as e:
            print(f"Ошибка при извлечении параметров: {e}")

        return dict(params)

    def parse_about_inf(self, html):
        """Парсит HTML объявления и возвращает данные в виде словаря"""
        soup = BeautifulSoup(html, 'html.parser')
        result = {}

        params_items = soup.find_all('li', class_='params-paramsList__item-_2Y2O')

        for item in params_items:
            name = item.find('span', class_='styles-module-noAccent-l9CMS').text.strip().replace(':', '').strip()
            value = item.contents[-1].strip()
            result[name] = value

        return result

    def parse_item_page(self, url):
        """Парсинг страницы конкретного объявления с сохранением всех параметров"""
        try:
            if not self.load_page(url):
                return None


            # Ожидание загрузки основной информации
            main_info_place = self.driver.find_element(By.XPATH, '//div[@data-marker="item-view/item-params"]')
            main_info_building = self.driver.find_element(By.XPATH, '//div[@class="styles-params-A5_I4"]')
            # price = self.driver.find_element(By.XPATH, '//span[@itemprop="price"]')
            # address = self.driver.find_element(By.XPATH, '//span[@class="style-item-address__string-wt61A"]')
            coordinates = self.driver.find_element(By.XPATH, '//div[@class="style-item-map-wrapper-ElFsX style-expanded-x335n"]')


            place_html = main_info_place.get_attribute('innerHTML')
            building_html = main_info_building.get_attribute('innerHTML')
            place = self.parse_about_inf(place_html)
            building = self.parse_about_inf(building_html)
            all_data = place | building | {
                                        "lat": coordinates.get_attribute("data-map-lat"),
                                        "lon": coordinates.get_attribute("data-map-lon")
                                        }

            return all_data

        except Exception as e:
            print(f"Ошибка парсинга объявления {url}: {e}")
            return None



    def save_to_dataframe(self):
        """
        Сохраняет список объявлений в DataFrame и CSV
        автоматически обрабатывая все возможные параметры
        """
        if self.full_data.empty:
            print("Нет данных для сохранения")
            return None

        # Создаем DataFrame

        # Сохраняем в CSV
        csv_path = os.path.join(self.output_dir, 'ads_data.csv')
        self.full_data.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"Данные сохранены в {csv_path}")

        return True

    def next_list_items_page(self):
        """
        Проверка наличия следующей страницы

        :param html: HTML текущей страницы
        :return: True если есть следующая страница, False если нет
        """

        try:
            self.scrap_items_list_page()

            next_btn = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.XPATH, '//a[@class="styles-module-item-zINQ7 styles-module-item_arrow-hv3h0 styles-module-item_link-GS05K"][@aria-label="Следующая страница"]'))
            )

            url = next_btn.get_attribute('href')
            return url


        except Exception as e:
            print(f"Ошибка при поиске следующей страницы: {e}")
            return None


    def run(self):
        """Основной метод запуска парсера"""

        page = 1
        total_ads = 1
        try:
            while True:

                print(f"\nСтраница {page}:")

                main_df = pd.DataFrame(self.parse_items_list_page()).set_index(['item_url'], drop=False)

                for item_url in main_df['item_url']:
                    print(f"\nПарсинг объявления {total_ads}")
                    all_data = self.parse_item_page(item_url)

                    for col, value in all_data.items():
                        self.full_data.loc[item_url, col] = value

                    total_ads += 1



                self.current_items_list_page = self.next_list_items_page()
                page += 1
                if not self.current_items_list_page:
                    print("Следующая страница не найдена")
                    break

            # Сохраняем все данные в DataFrame и CSV
            self.save_to_dataframe()
            print("\nСтатистика собранных данных:")
            print(self.full_data.info())
            print("\nПример данных:")
            print(self.full_data.head())

            print(f"\nГотово! Всего собрано {total_ads} объявлений")

            return True

        except KeyboardInterrupt:
            print("\nПарсер остановлен пользователем")
            if not self.full_data.empty:
                return self.save_to_dataframe()
        except Exception as e:
            print(f"\nКритическая ошибка: {e}")
            if not self.full_data.empty:
                return self.save_to_dataframe()
        finally:
            self.driver.quit()

if __name__ == "__main__":
    # URL для коммерческой недвижимости в Казани
    url = "https://www.avito.ru/tatarstan/kommercheskaya_nedvizhimost/sdam-ASgBAgICAUSwCNRW?cd=1&f=ASgBAQECAkSwCNRW9BKk2gECQJ7DDSSI2TmK2TmI9BE0zIGLA8qBiwPIgYsDAUW2ExZ7ImZyb20iOm51bGwsInRvIjoxNTB9&p=3&s=104"

    parser = AvitoParser(url)
    parser.run()
