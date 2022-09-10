from flask import Flask, json,redirect,render_template,flash,request,redirect,session,abort,jsonify
from flask.globals import request, session
from flask.helpers import url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash,check_password_hash

from flask_login import login_required,logout_user,login_user,login_manager,LoginManager,current_user
from models import Model
from depression_detection_tweets import DepressionDetection
from TweetModel import process_message
import os
#from flask_mail import Mail

from tracemalloc import stop
import streamlit as st
import numpy as np
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor


import json
import numpy as np
import pickle
import pandas as pd
import json
import plotly
import plotly.express as px
import plotly.graph_objs as go


from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

nltk.download('punkt')
nltk.download('stopwords')
sw=nltk.corpus.stopwords.words("english")



# model = pickle.load(open('stresslevel.pkl', 'rb'))
#creation of the Flask Application named as "app"
# mydatabase connection
local_server=True
app=Flask(__name__)


app = Flask(__name__,
            static_url_path='', 
            static_folder='static',
            template_folder='templates')

# app.config['SQLALCHEMY_DATABASE_URI']='mysql://root:@localhost/mental'

app.config['SQLALCHEMY_DATABASE_URI']='sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS']=False

db=SQLAlchemy(app)
app.secret_key="tandrima"
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(db.Model, UserMixin):
    id=db.Column(db.Integer,primary_key=True)
    usn=db.Column(db.String(20),unique=True)
    pas=db.Column(db.String(1000))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/signup')
@app.route('/signup',methods=['POST','GET'])
def signup():
    if request.method=="POST":
        usn=request.form.get('usn')
        pas=request.form.get('pas')
        
        # print(usn,pas)
        encpassword=generate_password_hash(pas)
        user=User.query.filter_by(usn=usn).first()
        if user:
            flash("UserID is already taken","warning")
            return render_template("usersignup.html")
            
        db.engine.execute(f"INSERT INTO `user` (`usn`,`pas`) VALUES ('{usn}','{encpassword}') ")
                
        # flash("SignUp Success Please Login","success")
        return render_template("userlogin.html")        

    return render_template("usersignup.html")

@app.route('/login',methods=['POST','GET'])
def login():
    if request.method=="POST":
        usn=request.form.get('usn')
        pas=request.form.get('pas')
        user=User.query.filter_by(usn=usn).first()
        if user and check_password_hash(user.pas,pas):
            login_user(user)
            # flash("Login Success","info")
            return redirect(url_for('home'))
        else:
            flash("Invalid Credentials","danger")
            return render_template("userlogin.html")


    return render_template("userlogin.html")

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash("Logout SuccessFul","warning")
    return redirect(url_for('login'))

@app.route('/findDepression')
@login_required
def findDepression():
    return render_template('depression.html')

@app.route('/board')
@login_required
def board():
    return render_template('board.html')

@app.route("/sentiment")
def sentiment():
    return render_template("sentiment.html")


@app.route("/predictSentiment", methods=["POST"])
def predictSentiment():
    message = request.form['form10']
    pm = process_message(message)
    result = DepressionDetection.classify(pm, 'bow') or DepressionDetection.classify(pm, 'tf-idf')
    return render_template("tweetresult.html",result=result)


@app.route('/predict', methods=["POST"])
def predict():
    q1 = int(request.form['a1'])
    q2 = int(request.form['a2'])
    q3 = int(request.form['a3'])
    q4 = int(request.form['a4'])
    q5 = int(request.form['a5'])
    q6 = int(request.form['a6'])
    q7 = int(request.form['a7'])
    q8 = int(request.form['a8'])
    q9 = int(request.form['a9'])
    q10 = int(request.form['a10'])

    values = [q1, q2, q3, q4, q5, q6, q7, q8, q9, q10]
    model = Model()
    classifier = model.svm_classifier()
    prediction = classifier.predict([values])
    if prediction[0] == 0:
            result = 'Your Depression test result : No Depression'
    if prediction[0] == 1:
            result = 'Your Depression test result : Mild Depression'
    if prediction[0] == 2:
            result = 'Your Depression test result : Moderate Depression'
    if prediction[0] == 3:
            result = 'Your Depression test result : Moderately severe Depression'
    if prediction[0] == 4:
            result = 'Your Depression test result : Severe Depression'
    return render_template("result.html", result=result)

app.secret_key = os.urandom(12)


@app.route('/music')
@login_required
def music():
    return render_template('music.html')

@app.route('/textAnalysis',methods=['POST','GET'])
@login_required
def text():
    if(request.method=='POST'):
        inp=request.form.get('inp')
        sid=SentimentIntensityAnalyzer()
        score=sid.polarity_scores(inp)
        if score["neg"]!=0:
            return render_template('text.html',message="Negative",color="red")
        else:
            return render_template('text.html',message="Positive",color="green")
    return render_template('text.html')


@app.route('/quizandgame')
@login_required
def quizandgame():
    return render_template('quizandgame.html')

@app.route('/simon')
@login_required
def simon():
    return render_template('simon.html')

@app.route('/moodMap')
@login_required
def moodMap():
    return render_template('mood.html')


@app.route('/memorygame')
@login_required
def memorygame():
    return render_template('memory.html')
    
@app.route('/exercises')
@login_required
def exercises():
    return render_template('exercises.html')


@app.route('/quiz')
@login_required
def quiz():
    return render_template('quiz.html')

@app.route('/game')
@login_required
def game():
    return render_template('game.html')

@app.route('/ticTacToe')
@login_required
def ticTacToe():
    return render_template('ticTacToe.html')

@app.route('/menja')
@login_required
def menja():
    return render_template('menja.html')


@app.route('/analysis',methods=['GET'])
@login_required
def analysis():
     #reading the dataset
    train_df = pd.read_csv('dreaddit-train.csv',encoding='ISO-8859-1')
    train_df.drop(['text', 'post_id' , 'sentence_range', 'id', 'social_timestamp'], axis=1, inplace=True)
    values = train_df['subreddit'].value_counts()
    labels = train_df['subreddit'].value_counts().index

    fig = px.pie(train_df, names=labels, values=values)
    fig.update_layout(title='Distribution of Subreddits')
    fig.update_traces(hovertemplate='%{label}: %{value}')
    #convert the plot to JSON using json.dumps() and the JSON encoder that comes with Plotly
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    train_df['label'].replace([0,1],['Not in Stress','In Stress'],inplace=True)
    fig2=px.histogram(train_df,
                 x="label",
                
                 title='Distribution of Stress Type',
                 color="label"
    )
    fig2.update_layout(bargap=0.1)
    #convert the plot to JSON using json.dumps() and the JSON encoder that comes with Plotly
    graphJSON2 = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
    
    fig3 = px.bar(train_df,
                 x='subreddit',
                 y='sentiment',
                 title='Car brand year resale ratio',
             color='subreddit')
    fig3.update_traces()
    graphJSON3 = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)
    fig4 = px.scatter(train_df,
                 x='subreddit',
                 y='social_karma',
                 title='Car brand price thousand ratio',
                 color="subreddit")
    fig4.update_traces()
    graphJSON4 = json.dumps(fig4, cls=plotly.utils.PlotlyJSONEncoder)
    
    fig5 = px.histogram(train_df,
                   x='confidence',
                   marginal='box',
                   title='Distribution of count reason of Mental Health issue',)
    fig5.update_layout(bargap=0.1)
    graphJSON5 = json.dumps(fig5, cls=plotly.utils.PlotlyJSONEncoder)
    
    fig6=px.histogram(train_df,
                 x="subreddit",
                
                 title='Distribution of Vehicle Type',color='subreddit')
    fig6.update_layout(bargap=0.1)
    graphJSON6 = json.dumps(fig6, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template('analysis.html', graphJSON=graphJSON,graphJSON2=graphJSON2,graphJSON3=graphJSON3,graphJSON4=graphJSON4,
                           graphJSON5=graphJSON5,graphJSON6=graphJSON6)
   

@app.route('/i')
@login_required
def i():
    return render_template('stress.html')

@app.route('/sudoku')
@login_required
def sudoku():
    return render_template('sudoku.html')

@app.route('/jokes&quotes')
@login_required
def jokes():
    return render_template('jokes.html')

@app.route('/faq')
def faq():
    return render_template('faq.html')

@app.route('/info')
def info():
    return render_template('info.html')

@app.route('/help')
def help():
    return render_template('help.html')

@app.route('/stress')
@login_required
def stress():
    return render_template('s.html')

# @app.route('/stressdetect',methods=['POST'])
# def stressdetect():
#     int_features = [int(x) for x in request.form.values()]
#     final_features = [np.array(int_features)]
#     prediction = model.predict(final_features)
#     #on basis of prediction displaying the desired output
#     if prediction=="Absence":
#         data="You are having Normal Stress!! Take Care of yourself"
#     elif prediction=="Presence":
#         data="You are having High Stress!! Consult a doctor and get the helpline number from our chatbot"
#     return render_template('stress.html', prediction_text3='Stress Level is: {}'.format(data))

#function to clean and transform the user input which is in raw format
def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text=y[:]
    y.clear()
    ps=PorterStemmer()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

#Stress Detection Prediction
tfidf3=TfidfVectorizer(stop_words=sw,max_features=20)
def transform3(txt1):
    txt2=tfidf3.fit_transform(txt1)
    return txt2.toarray()

df3=pd.read_csv("./Stress Detection.csv")
df3=df3.drop(["subreddit","post_id","sentence_range","syntax_fk_grade"],axis=1)
df3.columns=["Text","Sentiment","Stress Level"]
x=transform3(df3["Text"])
y=df3["Stress Level"].to_numpy()
x_train3,x_test3,y_train3,y_test3=train_test_split(x,y,test_size=0.1,random_state=0)
model3=DecisionTreeRegressor(max_leaf_nodes=2000)
model3.fit(x_train3,y_train3)

@app.route('/predictStress', methods=["POST","GET"])
@login_required
def predictStress():
    inp=request.form.get('inp')
    transformed_sent3=transform_text(inp)
    vector_sent3=tfidf3.transform([transformed_sent3])
    prediction3=model3.predict(vector_sent3)[0]
    
    if prediction3 >= 0:
            result = 'Your Stress test result : Stressful Text!!'
    elif prediction3<0:
            result = 'Your Stress test result : Not A Stressful Text!!'
    return render_template("result.html", result=result)

if __name__=="__main__":
    app.run(debug=True)