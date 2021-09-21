# SentimentAnalysisProject
Sentiment Analysis of reviews using Natural Language Processing Toolkit, Sklearn, Dash and WordCloud.
------------------------------------------------------------------------------------------------------------

How to use this?
I have used python version 3.8 for this project.

1. Use 'pip install -r requirements.txt' to install all the requirements for your project. Alternatively, you can create a virtualenv. Consider reading 
"https://docs.python.org/3/library/venv.html".

2. Update the location of your where you kept this folder.To do so, open 'utilities/constants.py' file and update line 7 with 
WORKING_DIR = r"<Location to the project folder>"

3. Open command prompt/terminal and move to the project directory and run command "python run_extra.py". This will extract the data from "https://www.etsy.com/" and you will 
have some data to show the results on ui.

4. Now run the command "python app.py" to run the dash app to show the results.

------------------------------------------------------------------------------------------------------------------

This UI has 4 parts:
a) The pie chart showing how many reviews that we extracted from "https://www.etsy.com/" are positive or negative.
b) The most frequently used words in the comment section.
c) A dropdown with randomly chosen 100 comments that we can click and check if that comment is positive or negative.
d) A textbox which in realtime will give us a positive or negative depending on what we write in the textbox.
