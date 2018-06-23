from flask import render_template
from sklearn import preprocessing
from flask import jsonify
from flask import url_for
from mysite import app
import pandas as pd
import pickle
import matplotlib.style
import matplotlib as mpl
import matplotlib.pyplot as plt
from flask import request
import pdb
import numpy as np
import base64
from io import BytesIO
import sys
import os 

if 'ubuntu' in os.getcwd():
    sys.path.append('/home/ubuntu/permileFlask/mysite')
    app_path = '/home/ubuntu/permileFlask'
else:
    sys.path.append('/Users/ahakso/Documents/gitDir/permileFlask/mysite')
    app_path = '/users/ahakso/Documents/gitDir/permileFlask'
from milemod import CustomDataFrame,  CustomSeries, nearest_neighbors, context_hist
import milemod
#print(CustomDataFrame(pd.DataFrame([0,1]))) This shows that the CustomDataFrame class is working

with open('{}/mysite/static/car_data.pkl'.format(app_path),'rb') as f:
    _ , car_data, _ = pickle.load(f)
with open('{}/mysite/static/combined_frame_dict.pkl'.format(app_path),'rb') as f:
    car_dict  = pickle.load(f)
with open('{}/mysite/static/combined_frame.pkl'.format(app_path),'rb') as f:
    combined_frame = pickle.load(f)


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
    plt.clf()
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
  piedata = np.array([fuel, depreciation, maintain, repair])/total
  pielbl = ['gas','depreciation','maintenance','repair']
  plt.pie(piedata,labels=pielbl,autopct=make_autopct(total*piedata))
  ax = plt.gca()
  ax.set_aspect(1)
  pie = mysavefig()

  # Make a context histogram
  neighbs, neighbs_all = nearest_neighbors(combined_frame, user_make, user_model, int(user_year), n_neighbors=20)
  ax = context_hist(neighbs, neighbs_all, user_make, user_model, int(user_year))
  histfig = mysavefig()

  return render_template("output.html", user_vehicle = (user_make, user_model, user_year),\
          total = total, depreciation = depreciation, fuel = fuel, repair = repair, maintain = maintain,\
          pie = pie, histfig = histfig)

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


