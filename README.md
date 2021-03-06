# SentimentAnalysisProject

## Sentiment Analysis of reviews using Natural Language Processing Toolkit, Sklearn, Dash and WordCloud.

How to use this?
I have used python version 3.8 for this project.

1. Use 'pip install -r requirements.txt' to install all the requirements for your project. Alternatively, you can create a virtualenv. Consider reading
   "https://docs.python.org/3/library/venv.html".

2. Download the compatible chromedriver for your chrome browser or compatible geckodriver for your firefox browser and place it here in ‘assets’ folder.

3. Open file run_extra.py and update line no 9's path to the path where you extracted review data(from https://nijianmo.github.io/amazon/index.html) is stored. Open command prompt/terminal and move to the project directory and run command "python run_extra.py". This will extract the data from "https://www.etsy.com/" and you will
   have some data to show the results on ui.

4. Now run the command "python app.py" to run the dash app to show the results.

---

This UI has 4 parts:

a) The pie chart showing how many reviews that we extracted from "https://www.etsy.com/" are positive or negative.

b) The most frequently used words in the comment section.

c) A dropdown with randomly chosen 100 comments that we can click and check if that comment is positive or negative.

d) A textbox which in realtime will give us a positive or negative depending on what we write in the textbox.

---

The model used here was also created by me. The code for that is also available in app_funcs.py and the data was collected from https://nijianmo.github.io/amazon/index.html, where I used the data under "Clothes, shoes and jewellery" section from year 2018.
