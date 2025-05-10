import pandas as pd
import scrapers

ads_data = pd.concat(pd.read_csv('output/ads_data.csv'), pd.read_csv(
    'output_2/ads_data.csv'))

ads_data