from flask import Flask
app = Flask(__name__, instance_path='/staticfiles')
from mysite import views
