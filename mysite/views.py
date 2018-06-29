from flask import render_template
from sklearn import preprocessing
from flask import jsonify
from flask import url_for
from mysite import app
import pandas as pd
import pickle
import matplotlib.style
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from flask import request
import pdb
import numpy as np
import base64
from io import BytesIO
import sys
import os 
import requests

if 'ubuntu' in os.getcwd():
    sys.path.append('/home/ubuntu/permileFlask/mysite')
    app_path = '/home/ubuntu/permileFlask'
else:
    sys.path.append('/Users/ahakso/Documents/gitDir/permileFlask/mysite')
    app_path = '/users/ahakso/Documents/gitDir/permileFlask'
from milemod import CustomDataFrame,  CustomSeries, nearest_neighbors, context_hist, prep_gas, zip2price

#print(CustomDataFrame(pd.DataFrame([0,1]))) This shows that the CustomDataFrame class is working

with open('{}/mysite/static/combined_frame_and_dict_final.pkl'.format(app_path),'rb') as f:
    combined_frame, car_dict = pickle.load(f)
with open('{}/mysite/static/zipstate.pkl'.format(app_path),'rb') as f:
                zipstate = pickle.load(f)
# Get today's gas prices
gasprice = prep_gas()

@app.route('/')
@app.route('/input', methods=['GET','POST'])
def cesareans_input():
    return render_template("input.html", car_dict = car_dict)

@app.route('/output', methods=['GET', 'POST'])
def permileOutput():
  def mysavefig():
    png_output = BytesIO()
#   plt.tight_layout()
    plt.savefig(png_output,transparent=False)
    png_output.seek(0)  # rewind to beginning of file
    figdata_png = base64.b64encode(png_output.getvalue()).decode('utf8')
    plt.clf()
    return figdata_png

  def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = pct*total
        return '{p:.1f}%\n({v:.0f} \xa2/mi)'.format(p=pct,v=val)
    return my_autopct


  #pull the user make from input field and store it
  user_make = request.form['select_make']
  user_model= request.form['select_model']
  user_year = request.form['select_year']
  user_zip = request.form['user_zip']
  if len(user_zip) == 0:
      user_zip = 999999
  else:
    user_zip = float(user_zip)
  user_gas = zip2price(zipstate, gasprice, user_zip)
  monthly_miles = float(request.form.get('monthly_miles'))
  with open('{}/mysite/static/combined_frame_and_dict_final.pkl'.format(app_path),'rb') as f:
      combined_frame, car_dict = pickle.load(f)

    
  # Pull relevant values from table
  depreciation = combined_frame.loc[(user_make, user_model,int(user_year)),'dollars_per_mile']
  fuel = user_gas/combined_frame.loc[(user_make, user_model,int(user_year)),'mpg']
  repair = combined_frame.loc[(user_make, user_model,int(user_year)),'repair']
  maintain = combined_frame.loc[(user_make, user_model,int(user_year)),'maintain']
  total = fuel+repair+maintain+depreciation
  #print('gas: {}\ndepreciation: {}\nrepair: {}\nmaintain: {}\n'.format(fuel, depreciation, repair, maintain))

  # This column is needed for the neighbors algorithm
  combined_frame = combined_frame.assign(total = combined_frame.dollars_per_mile+3.5/combined_frame.mpg+combined_frame.repair+combined_frame.maintain)
  
  # Make a pie plot
  mpl.style.use('seaborn')
  cost_list = [fuel, depreciation, maintain, repair]
  piedata = np.array(cost_list)/total
  pielbl = ['Fuel','Depreciation','Maintenance','Repair']
  pielbl = ['{}:\n{:0.2f} \xa2/mile'.format(pielbl[x],100*cost_list[x]) for x in range(4)]
  fig = plt.figure()
  fig.patch.set_alpha(0.8)
  ax = fig.add_axes((0.25,0.25,.5,.5))
  ax.patch.set_alpha(0.8)
  patches, texts = ax.pie(piedata,labels=pielbl, explode=(0.03,0.03,0.03,0.03), labeldistance=1.4)
  #patches, texts, autotexts = ax.pie(piedata,labels=pielbl,autopct=make_autopct(total*piedata))
  [x.set_fontsize(24) for x in texts]
# [x.set_fontsize(14) for x in autotexts]
  ax = plt.gca()
  ax.set_aspect(1)
  pie = mysavefig()

  # Make a context histogram
  neighbs_min, neighbs_max,  neighbs_all = nearest_neighbors(combined_frame, user_make, user_model, int(user_year), n_neighbors=20)
  ax, context_models, context_costs = context_hist(neighbs_min, neighbs_max, neighbs_all, user_make, user_model, int(user_year))
  target_make = context_models[0].split()[1].capitalize()
  target_model = context_models[0].split()[2]
  histfig = mysavefig()
  return render_template("output.html", user_vehicle = (user_make, user_model, user_year),monthly_miles=monthly_miles,\
          total = total, depreciation = depreciation, fuel = fuel, repair = repair, maintain = maintain,\
          pie = pie, histfig = histfig, context_models = context_models, context_costs = context_costs, target=(target_make,target_model))

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


