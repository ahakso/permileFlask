from flask import render_template
from flask import jsonify
from flask import url_for
from mysite import app
from mysite.a_Model import ModelIt
import pandas as pd
import psycopg2
import pickle
import matplotlib.pyplot as plt
from flask import request
import pdb
import numpy as np
import base64
from io import BytesIO

with open('/Users/ahakso/Documents/gitDir/permileFlask/mysite/static/car_data.pkl','rb') as f:
    car_dict, car_data, _= pickle.load(f)

#staticfile_path = '/static'
#app.config['jsonfiles'] = staticfile_path

@app.route('/')
@app.route('/input', methods=['GET','POST'])
def cesareans_input():
    return render_template("input.html", car_dict = car_dict)

@app.route('/output', methods=['GET', 'POST'])

def permileOutput():
  def mysavefig():
    png_output = BytesIO()
    plt.savefig(png_output)
    png_output.seek(0)  # rewind to beginning of file
    figdata_png = base64.b64encode(png_output.getvalue()).decode('utf8')
    return figdata_png
  def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = pct*total
        #pdb.set_trace()
        return '{p:.1f}%\n({v:.0f} cents/mi)'.format(p=pct,v=val)
    return my_autopct


  #pull the user make from input field and store it
  user_make = request.form['select_make']
  user_model= request.form['select_model']
  user_year = request.form['select_year']
    
  # Pull relevant values from table
  depreciation = car_data.loc[(user_make, user_model,int(user_year)),'dollars_per_mile']
  fuel = 3.5/car_data.loc[(user_make, user_model,int(user_year)),'mpg']
  repair = car_data.loc[(user_make, user_model,int(user_year)),'repair']
  maintain = car_data.loc[(user_make, user_model,int(user_year)),'maintain']
  total = fuel+repair+maintain+depreciation
  print('gas: {}\ndepreciation: {}\nrepair: {}\nmaintain: {}\n'.format(fuel, depreciation, repair, maintain))

  # Make a pie plot
# fig, ax = plt.subplots(1)
# ax.set_aspect(1)
  piedata = np.array([fuel, depreciation, maintain, repair])/total
  pielbl = ['gas','depreciation','maintenance','repair']
  plt.pie(piedata,labels=pielbl,autopct=make_autopct(total*piedata))
  ax = plt.gca()
  ax.set_aspect(1)
  pie = mysavefig()

  return render_template("output.html", user_vehicle = (user_make, user_model, user_year),\
          total = total, depreciation = depreciation, fuel = fuel, repair = repair, maintain = maintain,\
          pie = pie)

@app.route('/get_models')
def get_models():
    make = request.args.get('make')
    if make:
        models = list(car_dict[make])
        data = [{"id": str(x), "name": str(x)} for x in models]
        
#       print(data)
    return jsonify(data)

@app.route('/get_years')
def get_years():
    model = request.args.get('model')
    make = request.args.get('make')
#   print('\n\n{} {}\n\n'.format(make, model))
#   print('make: {}'.format(make))
#   print('model: {}'.format(model))
    if model:
        years = list(car_dict[make][model])
        data = [{"id": str(x), "name": str(x)} for x in years]
        
        print(data)
    return jsonify(data)


#@app.route('/get_model_array')
#def read_txt_file():
  #read it in
#  make = read_csv()
# return car_dict[make]

