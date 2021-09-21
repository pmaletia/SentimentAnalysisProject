"""
@name: main.py
@brief: starting point of the application
@author: Priyanka Maletia
"""

# file imports
from utilities.constants import PROJECT_NAME, STYLESHEETS, WORKING_DIR
from utilities.app_funcs import get_app_layout, get_review_result, unpickle_file, create_word_cloud

# import third party libraries
from dash import Dash, html
from dash.dependencies import Input, Output, State
import pandas as pd

# create dash app instance
app = Dash(__name__, external_stylesheets=STYLESHEETS)

# call back functions
@app.callback(
    Output('word-cloud-image', 'src'),
    [
        Input('word-cloud-image', 'id')
    ]
)
def make_image(b):
    """
    @name: make_wordcloud_image
    @brief: callback function that creates image for the wordcloud section
    @return: returns the wordcloud image to the wordcloud section
    """
    image = create_word_cloud()
    return image


@app.callback(
    Output("result-dropdown", "children"), 
    [
        Input('button-dropdown', 'n_clicks')
    ],
    [
        State('dropdown-reviews', 'value')
    ]
)
def dropdown_review(clicks, value):
    """
    @name: dropdown_review
    @brief: callback function that returns the positive or negative depending on the review selected in dropdown
    @return: returns the result to dropdown review section
    """
    print("Dropdown Data-Value")
    print("Data Type = ", str(type(value)))
    print("Value = ", str(value))
    print("Dropdown Data-Clicks")
    print("Data Type = ", str(type(clicks)))
    print("Value = ", str(clicks))
    if clicks>0:
        result = get_review_result(value)
        if (result == 0 ):
            return html.H2("Negative", id="negative")
        elif (result == 1 ):
            return html.H2("Positive", id="positive")
        else:
            return html.H2("Unknown", id="unknown")

@app.callback(
    Output("result-text", "children"), 
    [
        Input('text-review', 'value')
    ]
)
def text_review(value):
    """
    @name: text_review
    @brief: callback function that returns the positive or negative depending on the review written in textarea
    @return: returns the result to text review section
    """
    print("Text Data")
    print("Data Type = ", str(type(value)))
    print("Value = ", str(value))
    print()
    if value==None:
        return ""
    elif value=="":
        return ""
    else:
        result = get_review_result(value)
        if (result == 0):
            return html.H2("Negative", id="negative")
        elif (result == 1):
            return html.H2("Positive", id="positive")
        else:
            return html.H2("Unknown", id="unknown")


def main():
    """
    @name: main
    @brief: starts the dash app
    """
    app.title = PROJECT_NAME
    app.layout = get_app_layout()
    app.run_server(debug=False)
    

if __name__ == "__main__":
    main()      # calls the main function to start the app