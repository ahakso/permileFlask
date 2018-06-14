from flask import render_template
from mysite import app
from mysite.a_Model import ModelIt
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import psycopg2
import pickle
from flask import request

# Python code to connect to Postgres
# You may need to modify this based on your OS, 
# as detailed in the postgres dev setup materials.
user = 'ahakso' #add your Postgres username here      
host = 'localhost'
dbname = 'birth_db'
db = create_engine('postgres://%s%s/%s'%(user,host,dbname))
con = None
con = psycopg2.connect(database = dbname, user = user)

with open('/Users/ahakso/Documents/gitDir/permileFlask/mysite/static/car_data.pkl','rb') as f:
    car_dict, car_data = pickle.load(f)

@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html",
       title = 'Home', user = { 'nickname': 'Miguel' },
       )
@app.route('/input')
def cesareans_input():
    return render_template("input.html", car_dict = car_dict)

@app.route('/output')
def cesareans_output():
  #pull the user make from input field and store it
  user_make = request.args.get('user_make')
  print(user_make)
  return render_template("output.html", births = user_make, the_result = user_make)
