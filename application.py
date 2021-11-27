import backend
from flask import Flask, redirect, url_for, render_template, request
from flask_sockets import Sockets

application = Flask(__name__, static_url_path = '/static', static_folder = 'static')
sockets     = Sockets(application)

@sockets.route('/echo')
def echo_socket(ws):
  while not ws.closed:
    message = ws.receive()
    ws.send(message)

@application.route('/')
def hello():
  return render_template('search.html')

@application.route('/result', methods = ['POST'])
def main():
  backend.main(request)
  return render_template('out.html')

if __name__ == '__main__':
  from gevent import pywsgi
  from geventwebsocket.handler import WebSocketHandler
  server = pywsgi.WSGIServer(('', 5000), application, handler_class = WebSocketHandler)
  server.serve_forever()
