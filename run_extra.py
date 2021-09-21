"""
@name: run_extra.py
@brief: run before running the main app. Will give you data for the app
"""

# import from files
from utilities.app_funcs import etsy_data_scrapper,calc_review_classes, create_etsy_vocab

#etsy_data_scrapper(True, "firefox")

calc_review_classes()

create_etsy_vocab()