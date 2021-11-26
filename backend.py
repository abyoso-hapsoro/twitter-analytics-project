import os, base64
from PIL import Image
from io import BytesIO

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from plotly.offline import download_plotlyjs, init_notebook_mode, plot
import plotly.graph_objs as go
import plotly.figure_factory as ff
init_notebook_mode(connected = False)

from wordcloud import WordCloud

import networkx as nx
from networkx.readwrite import json_graph

import tweepy

import re
from string import punctuation
from collections import Counter
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import joblib

def main(request):
  keyword, tweetcount = str(request.form['Keyword']), int(request.form['TweetCount'])
  
  # using borrowed API from the workshop
  consumer_key        = 'kNm3NCo0JCbvaxMQmues1cVTa' # os.getenv('consumer_key')
  consumer_secret     = 'KMT1J4XH2Xz48qVnWGMMWNYefab6UJaEXvu32rEjixknlhDsvD' # os.getenv('consumer_secret')
  access_token        = '1435202232304082944-il5JLZKufbFzoDmWGiirQFyux1HB21' # os.getenv('access_token')
  access_secret       = 'mk6Jvqeki8SZpzIoqJjtcRYzto7vU3IPVXsmiadZcCOXU' # os.getenv('access_secret')

  auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
  auth.set_access_token(access_token, access_secret)
  api  = tweepy.API(auth)

  tweets = tweepy.Cursor(api.search_tweets, q = keyword, tweet_mode = 'extended', lang = 'id').items(tweetcount)
  retweet, user_name, retweet_count, created_at, message, user_id = [], [], [], [], [], []
  count  = 0
  for tweet in tweets:
    count += 1
    if hasattr(tweet, 'retweeted_status'):
      retweet.append(tweet.retweeted_status.user.screen_name)
      user_name.append(tweet.user.screen_name)
      retweet_count.append(tweet.retweet_count)
      created_at.append(tweet.created_at)
      message.append(tweet.retweeted_status.full_text)
      user_id.append(tweet.user.id)
    else:
      retweet.append(None)
      user_name.append(tweet.user.screen_name)
      retweet_count.append(tweet.retweet_count)
      created_at.append(tweet.created_at)
      message.append(tweet.full_text)
      user_id.append(tweet.user.id)
  
  data  = pd.DataFrame({
      'author'       : retweet,
      'username'     : user_name,
      'retweet_count': retweet_count,
      'created_at'   : created_at,
      'tweets'       : message
  })
  data  = data.dropna(subset = ['tweets'])
  sna_data = data.copy()
  
  # WordCloud
  def clean_tweet(string):
    string  = str(string).lower()
    string  = re.sub('[\s]+', ' ', string)
    string  = re.sub(r'\&\w*;', '', string)
    string  = re.sub('@[^\s]+', '', string)
    string  = re.sub(r'\$\w*', '', string)
    string  = re.sub(r'https?:\/\/.*\/\w*', '', string)
    string  = re.sub(r'#\w*', '', string)
    string  = re.sub(r'[' + punctuation.replace('@', '') + ']+', ' ', string)
    string  = re.sub(r'\b\w{1,2}\b', '', string)
    string  = ''.join(c for c in string if c <= '\uFFFF')
    string  = string.lstrip(' ') 
    string  = string.strip('\'"')
    
    stpword = StopWordRemoverFactory().create_stop_word_remover()
    string  = stpword.remove(string)

    stemmer = StemmerFactory().create_stemmer()
    string  = stemmer.stem(string)

    return string

  data['tweets'] = data['tweets'].apply(clean_tweet)
  data  = data.drop_duplicates('tweets')

  def tokenize_helper(string, lst = ['']):
    punct_less = [char for char in list(str(string)) if char not in punctuation]
    punct_less = ''.join(punct_less)
    return [word for word in punct_less.lower().split() if word.lower() not in lst]
  
  data['tokens'] = data['tweets'].apply(tokenize_helper)
  data  = data[['tweets','tokens']]
  
  def plot_wordcloud(wordfreq, figsize = (700/100, 350/100), cloudsize = (800, 450)):
    wordcloud = WordCloud(width = cloudsize[0],
                          height = cloudsize[1],
                          max_words = 500,
                          max_font_size = 100,
                          relative_scaling = 0.5,
                          colormap = 'gist_rainbow',
                          normalize_plurals = True).generate_from_frequencies(wordfreq)
    fig, tmpfile = plt.figure(figsize = figsize), BytesIO()
    
    plt.imshow(wordcloud, interpolation = 'bilinear', figure = fig)
    plt.axis('off')

    fig.set_facecolor('None')
    fig.savefig(tmpfile, facecolor = fig.get_facecolor(), edgecolor = 'None', format = 'png')

    return base64.b64encode(tmpfile.getvalue()).decode('utf-8')

  all_words = []
  for line in data['tokens']: 
    all_words.extend(line)  
  
  wordfreq  = Counter(all_words)
  wc_deploy = plot_wordcloud(wordfreq)

  # Sentiment Analysis
  pred_data = data.copy()
  
  Data = pd.read_csv('data_cleaned.csv')
  Data = Data.dropna()
  Data = Data.rename({'Tweet': 'tweets'}, axis = 1)
  Data = pd.concat([Data['tweets'], data['tweets']], ignore_index = True)

  bow_transformer   = CountVectorizer().fit(Data)
  bow_tweets        = bow_transformer.transform(Data)
  tfidf_transformer = TfidfTransformer().fit(bow_tweets)
  tfidf_tweets      = tfidf_transformer.transform(bow_tweets)

  model     = joblib.load('TwitterSentiments.pkl')
  pred_data['predictions'] = model.predict(tfidf_tweets[-len(data):,:17577])
  
  try:
    negatives = pred_data['predictions'].value_counts()[-1]
  except:
    negatives = 0
  
  try:
    neutrals  = pred_data['predictions'].value_counts()[0]
  except:
    neutrals  = 0
  
  try:
    positives = pred_data['predictions'].value_counts()[1]
  except:
    positives = 0

  values, labels = [negatives, neutrals, positives], ['Negative', 'Neutral', 'Positive']
  sentiment_pie  = {'data' : [{
    'type'      : 'pie', 
    'name'      : 'Indonesian Tweets Sentiment Analysis',  
    'values'    : values,
    'labels'    : labels,
    'direction' : 'clockwise',
    'marker'    : {'colors': ['red', 'lightgray', 'blue']}
    }], 'layout': {'title' : '', 'width': 350, 'height': 350,
                   'paper_bgcolor': 'rgba(0,0,0,0)', 'plot_bgcolor': 'rgba(0,0,0,0)'}
  }
  sa_deploy      = plot(sentiment_pie, config = {'displayModeBar': False}, show_link = False, include_plotlyjs = False, output_type = 'div')

  # Social Network Analysis
  sna_data = sna_data.dropna()
  
  network  = nx.from_pandas_edgelist(sna_data, source = 'author', target = 'username')
  G        = nx.convert_node_labels_to_integers(network, first_label = 0, ordering = 'default', label_attribute = None)
  pos      = nx.fruchterman_reingold_layout(G)
  poslabs  = nx.fruchterman_reingold_layout(network)

  Xe, Ye  = [], []
  for e in G.edges():
    Xe.extend([pos[e[0]][0], pos[e[1]][0], None])
    Ye.extend([pos[e[0]][1], pos[e[1]][1], None])
  trace_edges = {
    'type'  : 'scatter', 'mode': 'lines',
    'x'     : Xe, 'y': Ye,
    'line'  : {'width': 1, 'color': 'rgb(25, 25, 25)'},
    'hoverinfo': 'none'
  }

  Xn, Yn = [pos[k][0] for k in range(len(pos))], [pos[k][1] for k in range(len(pos))]
  trace_nodes = {
    'type'  : 'scatter', 'mode': 'markers',
    'x'     : Xn, 'y': Yn,
    'marker': {
      'showscale': True, 'size': 5, 'color' : [], 'colorscale': 'Rainbow', 'reversescale': True,
      'colorbar' : {'thickness': 15, 'title': 'Node Connections', 'xanchor': 'left', 'titleside': 'right'}
    },
    'text'  : list(poslabs) + list(' : '),
    'hoverinfo': 'text'
  }
  
  for node, adjacencies in enumerate(G.adjacency()):
    trace_nodes['marker']['color'] += tuple([len(adjacencies[1])])
  
  axis   = {'showline': False, 'zeroline': False, 'showgrid': False, 'showticklabels': False, 'title': ''}
  layout = {
      'font'  : {'family': 'Cambria'},
      'width' : 1085, 'height': 1085, 'autosize': False, 'showlegend': False,
      'xaxis' : axis, 'yaxis' : axis,
      'margin': {'l': 40, 'r': 40, 'b': 15, 't': 15, 'pad': 0},
      'hovermode': 'closest',
      'paper_bgcolor': 'rgba(0, 0, 0, 0)', 'plot_bgcolor': 'rgba(0, 0, 0, 0)'
  }
  
  social_network = {'data': [trace_edges, trace_nodes], 'layout': layout}
  sna_deploy     = plot(social_network, config = {'displayModeBar': False}, show_link = False, include_plotlyjs = False, output_type = 'div')

  # Data Sample Table
  data_sample  = sna_data.sample(min(30, len(sna_data)), random_state = 123)
  sample_table = ff.create_table(data_sample, height_constant = 25)
  table_deploy = plot(sample_table, config = {'displayModeBar': False}, show_link = False, include_plotlyjs = False, output_type = 'div')

  # Produce HTML with the process-acquired variables
  html_string = '''
<!DOCTYPE html>
<html>

<head>
  <style type='text/css'>
    .body{min-height: 100vh}
    .header-emoji{font-size: 50px}
    .header-title{font-size: 38px}
    .header-description{font-size: 16px}
  </style>
  <meta charset='utf-8'>
  <meta name='viewport' content='height=device-height, width=device-width, initial-scale=1.0'>
  <title>Twitter Analytics Dashboard: Understand Your Tweet Query!</title>
  <script src='https://cdn.plot.ly/plotly-latest.min.js'></script>
  <link rel='icon' type='image/x-png' href='https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/120/twitter/282/bird_1f426.png'>
  <link href='https://fonts.googleapis.com/css2?family=Cambria:wght@400;700&amp;display=swap' rel='stylesheet'>
  <link rel='stylesheet' href='https://cdnjs.cloudflare.com/ajax/libs/twitter-bootstrap/4.1.0/css/bootstrap.min.css'>
</head>

<body style='background-color:#add8e6;margin:0'>
  <main class='entry-point'>
    <div>
      <div class='header' style='background-color:#2cb037;color:#ffffff;text-shadow:2px 2px #222222;padding-bottom:35px'>
        <p class='header-emoji' style='text-align:center;margin:0;padding-top:25px'>🐦</p>
        <h1 class='header-title' style='text-align:center'>Twitter Analytics Dashboard</h1>
        <p class='header-description' style='text-align:center'>
          Here is the interactive Social Network Analysis, Sentiment Analysis, and WordCloud for your query
        </p>
      </div>
    <div display='flex' justify-content='center'>
      <div class='container'>
        <div class='row'>
          <div class='col-md-12' style='text-align:center;padding:25px 0 25px 0'>
            <h2 style='text-align:center'>Social Network Analysis</h2> 
            ''' + sna_deploy + '''
          </div>
          <div class='col-md-8' style='text-align:center'>
            <h2 style='text-align:center'>WordCloud</h2>
            ''' + '<img src=\'data:image/png;base64,{}\'>'.format(wc_deploy) + '''
          </div>
          <div class='col-md-4' style='text-align:center'>
            <h2 style='text-align:center'>Sentiment Analysis</h2> 
            ''' + sa_deploy + '''
          </div>
          <div class='col-md-12' style='text-align:center;padding-top:25px'>
            <h2 style='text-align:center'>Tweet Samples</h2>
            ''' + table_deploy + '''
          </div>
        </div>
      </div>
    </div>
  </main>
  <script src='https://cdnjs.cloudflare.com/ajax/libs/jquery/3.2.1/jquery.min.js'></script>
  <script src='https://cdnjs.cloudflare.com/ajax/libs/twitter-bootstrap/4.1.0/js/bootstrap.bundle.min.js'></script>
  <script src="{{url_for('static',filename='assets/js/script.min.js')}}"></script>
</body>

</html>
'''
  
  with open('templates/out.html', 'w') as html_file:
    html_file.write(html_string)
