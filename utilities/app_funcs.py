"""
@name: app_funcs.py
@brief: All the functions that we need for the app
@author: Priyanka Maletia
"""
# python file imports
from .constants import WORKING_DIR

# python third party library imports
from dash import html
from dash import dcc
import plotly.express as px
import pandas as pd
from wordcloud import WordCloud
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.firefox.options import Options as Options1
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# python inbuilt packages
import pickle
from os import getcwd, sep, mkdir
from time import sleep
import re
import base64
from glob import glob


# defining functions directly used in app at runtime
def load_model_vocab():
    """
    @name: load_model_vocab
    @brief: recreate the model and vocab objects and return it
    @return: model object, vocabulary object
    """
    # model recreate from pickle file
    file = open(WORKING_DIR + sep + "assets" + sep + "project_model.pkl", 'rb')
    model = pickle.load(file)
    
    # vocab file
    file_vocab = open(WORKING_DIR + sep + "assets" + sep + "project_vocab.pkl", 'rb')
    vocab = pickle.load(file_vocab)

    return model, vocab


def get_review_result(review: str):
    """
    @name: get_review_result
    @brief: Calculates if the result is positive(1) or negative(0)
    @param: review: review you wanna calculate the class(positive and negative) of.
    @return: 1 or 0
    """
    model, vocab = load_model_vocab()

    # vectorizer recreate
    vect = TfidfVectorizer(vocabulary=vocab)
    
    # change review
    review_transformed = vect.fit_transform([review])
        
    # guess by model
    guess = model.predict(review_transformed)
    
    return guess[0]

def create_pie_chart():
    """
    @name: create_pie_chart
    @brief: creates the figure to be fed to the dcc.graph
    @return: figure object
    """
    dataset = get_reviews_dataFrame(WORKING_DIR + sep + "assets" + sep + "etsy_reviews_predicted.csv")
    positive = dataset["class"][dataset["class"]==1].shape[0]
    negative = dataset["class"][dataset["class"]==0].shape[0]
    df = pd.DataFrame(columns=["categories", "values"])
    df["categories"] = ["Positive","Negative"]
    df["values"] = [positive, negative]
    fig = px.pie(df, values=df["values"].values, names=df["categories"].values, color='categories', color_discrete_map={'Positive':'lightgreen', "Negative":"pink"})
    return fig


def unpickle_file(filepath):
    """
    @name: unpickle_file
    @brief: unpickle any file and provides the original object
    @filepath: path to the pickle file
    @return: object of any kind
    """
    file = open(filepath, 'rb')
    return pickle.load(file)


def create_word_cloud():
    """
    @name: create_word_cloud
    @brief: create a word cloud image
    @return: png image location
    """
    
    """
    # just in case wc.generate_from_frequencies(data) doesn't work
    file = open(WORKING_DIR+"\\assets\\etsy_vocab.txt",'rb')
    text = str(file.read())
    file.close()
    wc = WordCloud(background_color='lightpink', width=480, height=360)
    print(type(text))
    wc.generate_from_text(text)
    """
    data = unpickle_file(WORKING_DIR + sep + "assets" + sep + "etsy_vocab.pkl")
    data = dict(data)
    wc = WordCloud(background_color='lightpink', width=480, height=360)
    wc.generate_from_frequencies(data)
    wc.to_file(WORKING_DIR + sep + "assets" + sep + "wordCloud.png")
   
    return "assets"+sep+"wordCloud.png"


#---------------------------------------------------------------------------------------------------------

# defining the app layout
def get_app_layout():
    """
    @name: get_app_layout
    @brief: creates a layout for our app
    @return: layout object
    """
    layout = html.Div(
            className="mainWrapper",
            children = [
                html.Img(src="assets"+sep+"4siWyr.webp", id="bg-image", className="bg-image"),
                html.Div(
                    className="content-wrapper",
                    children = [
                        html.Div(
                            className="row",
                            children = [
                                html.Div(
                                    className="column",
                                    children = [
                                        html.H1(children="Reviews Analysis for Etsy", id="pie-chart-head"),
                                        dcc.Graph(id="pie-chart", figure=create_pie_chart())
                                    ]
                                ),
                                html.Div(
                                    className="column",
                                    children = [
                                        html.H1(children="Word Cloud", id="word-count-head"),
                                        html.Img(id="word-cloud-image", src=None)
                                    ]
                                )
                            ]
                        ),
                        html.Div(
                            className="row",
                            children = [
                                html.Div(
                                    className="column",
                                    children = [
                                        html.H1(children="Dropdown Review Analysis", id="dropdown-review-head"),      
                                        dcc.Dropdown(
                                            id='dropdown-reviews',
                                            placeholder = 'Select a Review',
                                            options=[{'label': i[:100] + "...", 'value': i} for i in get_reviews_dataFrame(WORKING_DIR + sep + "assets" + sep + "etsy_reviews.csv").review],
                                            value = None,
                                            style = {'margin-bottom': '30px'}
            
                                        ),                                    
                                        html.Button("Check", className="button", id = 'button-dropdown', n_clicks=0),
                                        html.Div(id = 'result-dropdown'),
                                    ]
                                ),
                                html.Div(
                                    className="column",
                                    children = [
                                        html.H1(children="Analyse Review", id="text-review-head"),
                                        dcc.Textarea(id="text-review", className="text-review", placeholder="Type your review here"),
                                        html.Div(id ="result-text")
                                    ]
                                )
                            ]
                        )
                    ],
                )
            ]
        )
    return layout


# -------------------------------------------------------------------------------------------------------

# extra functions - to be run in run_extra.py before running the main apptp get some data
def etsy_data_scrapper(headless: bool, browserType: str, startPage: int, endPage: int):
    """
    @name: etsy_data_scrapper
    @brief: scrap reviews and create csv of scrapped reviews
    @param: headless: Give values True or False to tell if want to run the scrapping in headless 
    mode or not
    @param: browserType: give the type of browser you wanna run the scrapping in. Allowed values -> 'chrome',
    'firefox' (case insensitive)
    @param: startPage: give the page from where to start extracting data for the products
    @para: endPage: give the page from where to stop extracting data for the products
    """

    df = pd.DataFrame(columns=["review"])

    # loop to run it for each page having products. Mention the range of pages in the loop range
    for i in range(startPage, endPage+1):
        if browserType.lower()=="chrome":
            # set options for chrome browser
            options = Options()
            options.headless = headless
            # open the browser
            browser = webdriver.Chrome(executable_path=WORKING_DIR + sep + "assets" + sep + "chromedriver.exe", chrome_options=options)
        elif browserType.lower()=="firefox":
            # set options for firefox browser
            options = Options1()
            options.headless = headless
            # open the browser
            browser = webdriver.Firefox(executable_path=WORKING_DIR + sep + "assets" + sep + "geckodriver.exe", firefox_options=options)
        sleep(3)

        # navigate to url
        etsy_url = f"https://www.etsy.com/in-en/c/accessories?ref=pagination&explicit=1&page={i}"
        browser.get(etsy_url)
        print(f"start page: {i}")

        # get all products on page
        products = browser.find_elements_by_class_name("listing-link")
        
        # get reviews for each product
        for prod in products:
            try:
                prod.click()
                browser.switch_to.window(browser.window_handles[1])
            except Exception as e:
                print(e)
            sleep(3)

            try:
                review0 = browser.find_element_by_id("review-preview-toggle-0")
                df.loc[len(df.index)] = [review0.text.strip()]
            except Exception as e:
                print(e)
            try:
                review1 = browser.find_element_by_id("review-preview-toggle-1")
                df.loc[len(df.index)] = [review1.text.strip()]
            except Exception as e:
                print(e)
            try:
                review2 = browser.find_element_by_id("review-preview-toggle-2")
                df.loc[len(df.index)] = [review2.text.strip()]
            except Exception as e:
                print(e)
            try:
                review3 = browser.find_element_by_id("review-preview-toggle-3")
                df.loc[len(df.index)] = [review3.text.strip()]
            except Exception as e:
                print(e)
            
            sleep(3)

            # close current product page and move to parent page
            parent = browser.window_handles[0]
            chld = browser.window_handles[1]
            browser.switch_to.window(chld)
            browser.close()
            browser.switch_to.window(parent)
        
        print(f"Done page {i}")
        parent = browser.window_handles[0]
        browser.switch_to.window(parent)

        # closing browser after getting reviews for all products on the page
        for handle in browser.window_handles:
            browser.switch_to.window(handle)
            browser.close()
    
    # storing reviews in assets/etsy_reviews1.csv
    df.to_csv(WORKING_DIR + sep + "assets" + sep + "etsy_reviews1.csv", index=False)


def get_reviews_dataFrame(filepath: str):
    """
    @name: get_reviews_dataFrame
    @brief: creates the dataframe for the reviews provided and returns it
    @param: filepath: path to csv file
    @return: dataframe for reviews
    """
    df = pd.read_csv(filepath)
    return df

def calc_review_classes():
    """
    @name: get_reviews_dataFrame
    @brief: creates the dataframe for the reviews and predict positive, negative for them and save it
    """
    df = get_reviews_dataFrame(WORKING_DIR + sep + "assets" + sep + "etsy_reviews.csv")
    df["class"] = [get_review_result(review) for review in df["review"]]
    
    df.to_csv(WORKING_DIR + sep + "assets" + sep + "etsy_reviews_predicted.csv", index=False)


def create_etsy_vocab():
    """
    @name: create_etsy_vocab
    @brief: create etsy vocab and save it as pickle file
    """
    df = get_reviews_dataFrame(WORKING_DIR + sep + "assets" + sep + "etsy_reviews.csv")

    #data cleaning
    nltk.download('stopwords')
    corpus = []
    for i in range(df.shape[0]):
        if i == 0:  
            print(df.iloc[i,0])
        review = re.sub("[^a-zA-Z]", " ", df.iloc[i,0])
        review = review.lower()
        review = review.split()
        review = [word for word in review if word not in 
                        stopwords.words('english')]
        ps = PorterStemmer()
        review = [ps.stem(m) for m in review]
        review = " ".join(review)
        #print(review)
        corpus.append(review)

    # create text file with vocab
    data = " ".join(corpus)
    file = open(WORKING_DIR + sep + "assets" + sep + "etsy_vocab.txt", 'w')
    file.write(data)
    file.close()

    # create pickle file of vocab dict
    vect = TfidfVectorizer(min_df=5).fit(corpus)
    vocab = vect.vocabulary_
    file = open(WORKING_DIR + sep + "assets" + sep + "etsy_vocab.pkl", "wb")
    pickle.dump(vocab,file)


def clean_amazon_data(filepath: str):
    """
    @name: clean_amazon_data
    @brief: cleans amazon reviews data got from https://nijianmo.github.io/amazon/index.html
    @param: filepath: path to the untar data file.
    """
    # increase/decrease chunksize based on your computer's resources. Also untar the data fileafter downloading.
    df_reader = pd.read_json(filepath, lines=True, chunksize=1000000)

    mkdir(WORKING_DIR + sep + "assets" + sep + "csv_files")
    
    #for loop
    counter = 1
    for chunk in df_reader:
        # getting data of just three categories - overall, review, summary
        new_df = pd.DataFrame(chunk[['overall', 'reviewText', 'summary']])
        new_df1 = new_df[new_df['overall']==1].sample(4000)
        new_df2 = new_df[new_df['overall']==2].sample(4000)
        new_df4 = new_df[new_df['overall']==4].sample(4000)
        new_df5 = new_df[new_df['overall']==5].sample(4000)
        new_df3 = new_df[new_df['overall']==3].sample(8000)
        
        # concat all the data of different rating
        new_df_concat = pd.concat([new_df1, new_df2, new_df3, new_df4, new_df5], 
                                ignore_index=True)
        new_df_concat.to_csv(WORKING_DIR + sep + "assets" + sep + "csv_files/"+str(counter)+".csv", index=False)
        print(str(counter)+" iteration complete!")
        counter=counter+1

    # getting all the csv's made so far
    filenames = glob(WORKING_DIR + sep + "assets" + sep + "csv_files/*.csv")

    dataframes = []

    for f in filenames:
        dataframes.append(pd.read_csv(f))
        
    # creating final dataframe and csv with balanced reviews
    final_dataframe = pd.concat(dataframes, axis=0, ignore_index=True)
    final_dataframe.to_csv(WORKING_DIR + sep + "assets" + sep + "csv_files/balanced_reviews.csv", index=False)


def create_model():
    """
    @name: create_model
    @brief: create the model that will finally check if the review is negative or positive
    """
    dataset = pd.read_csv(WORKING_DIR + sep + "assets" + sep + "csv_files/balanced_reviews.csv")

    # EDA
    print(dataset.shape)
    print(dataset.columns)
    print(dataset.dtypes)

    # null value check 
    print(dataset.isnull().any(axis=0).value_counts())
    print(dataset["overall"].value_counts())

    # because text value can't be filled with anything, we drop all the null values
    dataset = dataset.dropna() 
    dataset["overall"].value_counts()

    # now because we want just good and bad, we remove the neutral data
    dataset = dataset[dataset["overall"] != 3]
    dataset["overall"].value_counts()

    # now we want to do binary classification but we have like 4 data points right
    # now. So we can change the original 1, 2, 3, 4 to 1 and 0, but it's better we
    # do a new column. this data persist and we get binary classification as well

    dataset["Positivity"] = np.where(dataset["overall"]>3,1,0)

    # features - reviewText
    # labels - Positivity
    features = dataset["reviewText"]
    labels = dataset["Positivity"]

    # now to train model, we need to work with text data and that needs NLP
    # train test split
    features_train, features_test, labels_train, labels_test = train_test_split(
                                features, labels, train_size=0.8, random_state=1)

    # vectorization through tfidf
    vect = TfidfVectorizer(min_df=5).fit(features_train)

    features_train_vectorized = vect.transform(features_train)

    # model creation
    model = LogisticRegression()
    model.fit(features_train_vectorized, labels_train)

    predictions = model.predict(vect.transform(features_test))

    # accuracy scores and confusion matrix
    accuracy_score(labels_test, predictions)

    #confusion_matrix(labels_test, predictions)

    vocab = vect.vocabulary_

    file = open(WORKING_DIR + sep + "assets" + sep + "project_model.pkl", "wb")
    pickle.dump(model,file)

    file = open(WORKING_DIR + sep + "assets" + sep + "project_vocab.pkl", "wb")
    pickle.dump(vocab,file)