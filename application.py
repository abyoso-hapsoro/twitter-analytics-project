import backend
from flask import Flask, redirect, url_for, render_template, request
from flask_ngrok import run_with_ngrok

app = Flask(__name__, static_url_path = '/static', static_folder = 'static')
run_with_ngrok(app)

@app.route('/')
def hello():
  return render_template('search.html')

@app.route('/result', methods = ['POST'])
def main():
  backend.main(request)
  return render_template('out.html')

app.run()