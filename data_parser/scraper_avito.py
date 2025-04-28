import os
import time
import random
import xml.etree.ElementTree as ET
import csv
from urllib.parse import urljoin
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
        print(f"Initializing parser for {base_url}")
        self.base_url = base_url
        self.output_dir = output_dir
        self.driver = self._init_driver(headless)
        self.parsed_ads = set()
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {os.path.abspath(output_dir)}")

    def _init_driver(self, headless):
        """Initialize ChromeDriver with anti-detection settings"""
        chrome_options = Options()

        if headless:
            chrome_options.add_argument("--headless=new")

        # Anti-detection settings
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

        # Mask WebDriver
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
        """Random delay between actions"""
        delay = random.uniform(min_sec, max_sec)
        print(f"Waiting {delay:.2f} seconds...")
        time.sleep(delay)

    def load_page(self, url):
        """Load page with error handling"""
        print(f"Loading page: {url}")
        try:
            self.driver.get(url)
            self._random_delay(3, 7)
            return True
        except Exception as e:
            print(f"Error loading page: {e}")
            return False

    def _check_captcha(self):
        """Check for captcha presence"""
        try:
            return any([
                self.driver.find_element(By.CSS_SELECTOR, "div.captcha__container").is_displayed(),
                "captcha" in self.driver.page_source.lower(),
                "анти-робот" in self.driver.page_source.lower()
            ])
        except:
            return False

    def parse_page(self, url):
        """Parse page with ads"""
        try:
            if not self.load_page(url):
                return None

            # Check for captcha
            if self._check_captcha():
                print("Captcha detected! Please:")
                print("1. Solve captcha manually in browser")
                print("2. Change IP address")
                print("3. Wait 10-15 minutes")
                input("Press Enter after solving captcha...")
                self._random_delay(5, 10)

            # Wait for ads to load
            WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'div[data-marker="item"]')))

            # Scroll to load all ads
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            self._random_delay(1, 3)

            return self.driver.page_source
        except Exception as e:
            print(f"Error parsing page: {e}")
            return None

    def extract_ads(self, html):
        """Extract ads data from HTML"""
        print("Extracting ads...")
        soup = BeautifulSoup(html, 'html.parser')
        ads = []

        ad_cards = soup.select('div[data-marker="item"]')
        print(f"Found {len(ad_cards)} ad cards")

        for i, card in enumerate(ad_cards, 1):
            try:
                print(f"\nProcessing card #{i}")

                # Extract basic info
                title = card.select_one('[itemprop="name"]').text.strip()
                price = card.select_one('[itemprop="price"]')['content']
                address = card.select_one('[data-marker="item-address"]').text.strip()
                date = card.select_one('[data-marker="item-date"]').text.strip()
                link = urljoin(self.base_url, card.select_one('a[itemprop="url"]')['href'])

                # Skip if already parsed
                if link in self.parsed_ads:
                    continue
                self.parsed_ads.add(link)

                # Extract additional details
                area = self._extract_area(card)
                description = self._extract_description(card)

                ads.append({
                    'title': title,
                    'price': price,
                    'address': address,
                    'area': area,
                    'date': date,
                    'link': link,
                    'description': description
                })

            except Exception as e:
                print(f"Error in card #{i}: {e}")
                continue

        return ads

    def _extract_area(self, card):
        """Extract area in square meters"""
        try:
            params = card.select_one('[data-marker="item-specific-params"]').text
            for p in params.split(','):
                if 'м²' in p:
                    return p.split('м²')[0].strip()
            return "0"
        except:
            return "0"

    def _extract_description(self, card):
        """Extract ad description if available"""
        try:
            return card.select_one('[class*="description"]').text.strip()
        except:
            return ""

    def save_to_csv(self, ads, file_index):
        """Save ads to CSV file"""
        if not ads:
            print("No ads to save")
            return

        filename = os.path.join(self.output_dir, f"ads_{file_index}.csv")
        print(f"Saving {len(ads)} ads to {filename}")

        fieldnames = ['title', 'price', 'address', 'area', 'date', 'link', 'description']

        with open(filename, mode='w', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(ads)

    def run(self, max_ads=50, ads_per_file=20):
        """Main parser execution method"""
        print(f"\nStarting parser (max {max_ads} ads)")
        total_ads = 0
        page = 1

        try:
            while total_ads < max_ads:
                print(f"\nPage {page}:")
                url = f"{self.base_url}&p={page}"
                html = self.parse_page(url)

                if not html:
                    print("Failed to get data, stopping")
                    break

                ads = self.extract_ads(html)
                if not ads:
                    print("No ads found, stopping")
                    break

                self.save_to_csv(ads, page)
                total_ads += len(ads)

                # Check if we should continue
                if total_ads >= max_ads:
                    break

                page += 1
                self._random_delay(5, 10)

            print(f"\nDone! Collected {total_ads} ads total")
        except KeyboardInterrupt:
            print("\nParser stopped by user")
        except Exception as e:
            print(f"\nCritical error: {e}")
        finally:
            self.driver.quit()


if __name__ == "__main__":
    # Example URL for commercial real estate in Kazan
    url = "https://www.avito.ru/kazan/kommercheskaya_nedvizhimost/sdam-ASgBAgICAUSwCNRW?cd=1&f=ASgBAQECAkSwCNRW9BKk2gECQJ7DDTSI2TmG2TmK2TmI9BE0zIGLA8qBiwPIgYsDAkW2ExZ7ImZyb20iOm51bGwsInRvIjoxNTB9gqESHSLQvtGC0LTQtdC70YzQvdGL0Lkg0LLRhdC~0LQi&s=104"

    parser = AvitoParser(url)
    parser.run(max_ads=50)