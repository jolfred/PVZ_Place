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
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup


class WildberriesPVZParser:
    def __init__(self, base_url, coordinates_list, output_dir="output_wb", headless=False):
        print(f"Инициализация парсера для {base_url}")
        self.base_url = base_url
        self.coordinates_list = coordinates_list
        self.output_dir = output_dir
        self.driver = self._init_driver(headless)
        self.parsed_pvz = set()
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





    def click_center_of_screen(self):
        """Клик в центр видимой области экрана"""
        try:
            # Получаем размеры окна браузера
            window_size = self.driver.get_window_size()
            center_x = window_size['width'] // 2
            center_y = window_size['height'] // 2

            # Выполняем клик через ActionChains
            actions = ActionChains(self.driver)
            actions.move_by_offset(center_x, center_y).click().perform()
            actions.reset_actions()  # Сбрасываем смещения после move_by_offset

            print("Клик в центр экрана выполнен")
            return True
        except Exception as e:
            print(f"Ошибка при клике в центр экрана: {e}")
            return False

    def parse_single_radius(self, html_content, radius):
        """
        Функция для парсинга HTML-кода и извлечения данных в виде словаря.

        :param html_content: str, HTML-код для парсинга
        :return: dict, словарь с заголовками и значениями
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        data = {'radius': radius}

        for item in soup.find_all('li', class_='ant-list-item'):
            title = item.find('strong').text
            value = item.find_all('span')[-1].text
            data[title] = value

        return data

    def pars_location_info(self):
        try:

            roll_location_info_el = WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.XPATH, '//*[@id="rc-tabs-0-panel-discovery"]/div[2]/div[4]/div[1]/div[1]'))
            )
            roll_location_info_el.click()
            html_content_300m = WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located(
                    (By.XPATH, '//*[@id="rc-tabs-0-panel-discovery"]/div[2]/div[4]/div[1]/div[2]/div/div/div[2]/div/div'))
            ).get_attribute('innerHTML')
            data_300m = self.parse_single_radius(html_content_300m, 300)

            location_info_el_600m = WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located(
                    (By.XPATH, '//*[@id="rc-tabs-0-panel-discovery"]/div[2]/div[4]/div[1]/div[2]/div/div/div[1]/div/label[2]/div'))
            )

            location_info_el_600m.click()

            WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located(
                    (By.XPATH, '//*[@id="rc-tabs-0-panel-discovery"]/div[2]/div[4]/div[1]/div[2]/div/div/div[2]/div/div/ul/li[1]/div/div[2]/span/strong'))
            )


            html_content_600m = self.driver.find_element(By.XPATH, '//*[@id="rc-tabs-0-panel-discovery"]/div[2]/div[4]/div[1]/div[2]/div/div/div[2]/div/div')
            print(html_content_600m.text)
            inner_html = self.driver.execute_script("return arguments[0].innerHTML;", html_content_600m)

            data_600m = self.parse_single_radius(html_content_600m, 600)

            return pd.DataFrame([data_300m, data_600m])

        except Exception as e:
            print("нет нихуя")
            return pd.DataFrame()


    def parse_coordinates(self, lat, lon):
        """Парсинг данных для конкретных координат"""
        url = f"{self.base_url}#17/{lat}/{lon}"

        if not self.load_page(url):
            return []

        self.click_center_of_screen()

        WebDriverWait(self.driver, 15).until(
            EC.presence_of_element_located((By.XPATH, '//div[@class="ant-flex css-160wgsb ant-flex-wrap-wrap '
                                                     'ant-flex-justify-space-between"]'))
        )
        wb_type = self.driver.find_element(By.XPATH, '//*[@id="rc-tabs-0-panel-discovery"]/div[2]/div[1]/div[1]/div')


        res = self.pars_location_info()

        if not res.empty:
            res.assign(
                point_info=wb_type.text,
                lat=lat,
                lon=lon
            )
        else:
            res = pd.DataFrame([{
                "point_info": wb_type.text,
                "lat": lat,
                "lon": lon
            }])
        print(res)
        return res

    def save_to_dataframe(self):
        """
        Сохраняет список ПВЗ в DataFrame и CSV
        """
        if self.full_data.empty:
            print("Нет данных для сохранения")
            return None

        # Сохраняем в CSV
        csv_path = os.path.join(self.output_dir, 'wb_pvz_data.csv')
        self.full_data.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"Данные сохранены в {csv_path}")

        return True

    def run(self):
        """Основной метод запуска парсера"""
        try:
            total_pvz = 0

            for i, (lat, lon) in enumerate(self.coordinates_list, 1):
                print(f"\nОбработка координат {i}/{len(self.coordinates_list)}: {lat}, {lon}")

                pvz_data = self.parse_coordinates(lat, lon)

                if not pvz_data.empty:
                    # Добавляем данные в DataFrame
                    self.full_data = pd.concat([self.full_data, pvz_data], ignore_index=True)


                    # Сохраняем промежуточные результаты
                    if i % 10 == 0:
                        self.save_to_dataframe()

                # Случайная задержка между запросами
                self._random_delay(3, 7)

            # Финальное сохранение
            self.save_to_dataframe()
            print("\nСтатистика собранных данных:")
            print(self.full_data.info())
            print("\nПример данных:")
            print(self.full_data.head())

            print(f"\nГотово! Всего собрано {total_pvz} ПВЗ")

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
    # Базовый URL карты Wildberries
    base_url = "https://pvz-stat-map.wildberries.ru/"

    # Пример списка координат (замените на свои)
    # ads_data = pd.read_csv('output/ads_data.csv')
    # coordinates_list = ads_data[['lat', 'lon']].values
    coordinates_list = [
        (55.800686, 48.967669)
    ]
    parser = WildberriesPVZParser(base_url, coordinates_list)
    parser.run()