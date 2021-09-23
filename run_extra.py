"""
@name: run_extra.py
@brief: run before running the main app. Will give you data for the app
"""

# import from files
from utilities.app_funcs import (etsy_data_scrapper,calc_review_classes, create_etsy_vocab, clean_amazon_data,create_model)

clean_amazon_data(r"D:\MyStuff\data_science_stuff\Clothing_Shoes_and_Jewelry.json")

create_model()

etsy_data_scrapper(True, "firefox", 1, 250)

calc_review_classes()

create_etsy_vocab()