from data_cleaning_pipeline import DataCleaner
import logging
from datetime import datetime
import os

# Пути к данным
DATA_DIR = os.path.join('data_cleaning', 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

# Создаем директории для данных, если они не существуют
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Создаем директорию для логов, если она не существует
os.makedirs('logs', exist_ok=True)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/rent_cleaning_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def main():
    # Создаем экземпляр очистки данных для аренды
    input_file = os.path.join(RAW_DATA_DIR, 'rent_data_raw.csv')
    output_file = os.path.join(PROCESSED_DATA_DIR, 'rent_data.csv')
    
    cleaner = DataCleaner(input_file, 'rent')
    
    # Выполняем очистку
    cleaned_df = cleaner.clean_data()
    
    # Сохраняем результаты
    cleaner.save_results(output_file)
    
    # Генерируем и выводим отчет
    report = cleaner.generate_report()
    
    logging.info("\n=== Отчет по очистке данных аренды ===")
    logging.info(f"Всего записей: {report['total_records']}")
    
    logging.info("\nРаспределение по городам:")
    for city, count in report['cities'].items():
        logging.info(f"{city}: {count}")
    
    logging.info("\nСтатистика по ценам аренды:")
    for stat, value in report['total_price_stats'].items():
        logging.info(f"{stat}: {value:,.2f}")
    
    logging.info("\nСтатистика по ценам аренды за м²:")
    for stat, value in report['price_per_meter_stats'].items():
        logging.info(f"{stat}: {value:.2f}")
    
    logging.info("\nСтатистика по площадям:")
    for stat, value in report['area_stats'].items():
        logging.info(f"{stat}: {value:.2f}")
    
    # Выводим информацию об экстремальных ценах
    cleaner.print_extreme_prices()

if __name__ == "__main__":
    main() 