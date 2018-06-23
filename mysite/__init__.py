from flask import Flask
app = Flask(__name__)

from mysite import views
#from mysite/milemod import CustomDataFrame, CustomSeries
import os
import sys
if ubuntu in os.getcwd():
    sys.path.append('/home/ubuntu/flaskapp/mysite')
