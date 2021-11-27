import backend
from flask import Flask, redirect, url_for, render_template, request

application = Flask(__name__, static_url_path = '/static', static_folder = 'static')

@application.route('/')
def hello():
  return render_template('search.html')

@application.route('/result', methods = ['POST'])
def main():
  backend.main(request)
  return render_template('out.html')

if __name__ == '__main__':
  application.run()
