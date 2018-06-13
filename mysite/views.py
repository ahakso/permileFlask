from flask import render_template
from mysite import app
from mysite.a_Model import ModelIt
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import psycopg2
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

@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html",
       title = 'Home', user = { 'nickname': 'Miguel' },
       )
@app.route('/input')
def cesareans_input():
    makes = pd.Series(['Acura','Audi','Chevrolet','Ford'])
    return render_template("input.html", makes = makes)

@app.route('/output')
def cesareans_output():
  #pull 'birth_month' from input field and store it
  patient = request.args.get('birth_month')
    #just select the Cesareans  from the birth dtabase for the month that the user inputs
  query = "SELECT index, attendant, birth_month FROM birth_data_table WHERE delivery_method='Cesarean' AND birth_month='%s'" % patient
  print(query)
  query_results=pd.read_sql_query(query,con)
  print(query_results)
  births = []
  for i in range(0,query_results.shape[0]):
      births.append(dict(index=query_results.iloc[i]['index'], attendant=query_results.iloc[i]['attendant'], birth_month=query_results.iloc[i]['birth_month']))
      the_result = ''
  the_result = ModelIt(patient,births)
  return render_template("output.html", births = births, the_result = the_result)
#@app.route('/output')
#def cesareans_output():
#    return render_template("output.html")
