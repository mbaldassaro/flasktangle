from flask import Flask, render_template, url_for, request, make_response, flash, send_file, Response
from flask_bootstrap import Bootstrap
import pandas as pd
import numpy as np
import requests
#import os
import io
import json
import csv
from urllib.parse import urlsplit
import networkx as nx
#from rq import Queue, get_current_job
#from rq.job import Job
#from worker import conn
import time
import community as community_louvain
import plotly
import plotly.express as px
import plotly.graph_objects as go
import re
import string
import nltk
import nltk.corpus
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
import re
import string

app = Flask(__name__)
Bootstrap(app)
#app.secret_key = os.urandom(24)
#q = Queue(connection=conn)
def get_links(token, link, platforms='facebook', count=1000):
    api_url_base = "https://api.crowdtangle.com/links?token="
    link_pre = '&link='
    count_pre = '&count='
    plat_pre = '&platforms='
    api_url = format(f'{api_url_base}{token}{link_pre}{link}{plat_pre}{platforms}{count_pre}{count}')
    response = requests.get(api_url)
    if response.status_code == 200:
        return json.loads(response.content.decode('utf-8'))
    else:
        return None

def prep_batch(data, type='pages', minsize=0, listname='null'):
    df = pd.DataFrame.from_dict(data['result']['posts'])
    df = pd.concat([df.drop(['account'], axis=1), df['account'].apply(pd.Series)], axis=1)
    df = df.groupby(['name', 'url', 'accountType']).size().to_frame().reset_index().sort_values(by=0, ascending=False)
    if type == 'pages':
        df1 = df.loc[((df['accountType'] == 'facebook_page') & (df[0] > minsize))]
    else:
        df1 = df.loc[((df['accountType'] == 'facebook_group') & (df[0] > minsize))]
    df1['List'] = listname
    global df2
    df2 = df1.rename(columns={"url": "Page or Account URL"}).reset_index(drop=True)
    df2 = df2[['Page or Account URL', 'List']].reset_index(drop=True)
    #return df1[['Page or Account URL', 'List']]

def dupe_links(data):
    data_links = data[data['Link'].notnull()]
    data_links = data_links[data_links.duplicated(['Link'], keep=False)]
    data_links = pd.DataFrame(data_links[['Post Created', 'Name', 'Link']]).sort_values(by=['Link', 'Post Created'], ascending=False)
    #temp_data_links = pd.DataFrame(np.sort(data_links[['Name','Link']], axis=1)).reset_index(drop=True)
    #data_links2 = temp_data_links[~temp_data_links.duplicated()]
    #temp_df2 = temp_df2.rename(columns={0: 'target', 1: 'source'}).reset_index(drop=True)
    return data_links

def dupe_messages(data):
    data_messages = data[data['Message'].notnull()]
    data_messages = data_messages[data_messages.duplicated(['Message'], keep=False)]
    data_messages = pd.DataFrame(data_messages[['Post Created', 'Name', 'Message']]).sort_values(by=['Message', 'Post Created'], ascending=False)
    return data_messages

def prep_sna(data):
    data_sub = data[data['Link'].notnull()]
    data_sub = data_sub[data_sub['Link'].str.contains(r"facebook.com")]
    data_sub = data_sub[~data_sub['Link'].str.contains(r"photo.php")]
    data_sub['Link'] = data_sub['Link'].replace(to_replace=r"/groups", value='', regex=True)
    data_sub['Link'] = data_sub['Link'].replace(to_replace=r"/photos", value='', regex=True)
    data_sub['Link'] = data_sub['Link'].replace(to_replace=r"\?", value='', regex=True)
    data_sub['Link'] = data_sub['Link'].replace(to_replace=r"fbid=", value='', regex=True)
    data_sub['Link'] = data_sub['Link'].replace(to_replace=r"\&.*", value='/', regex=True)
    data_sub['protocol2'], data_sub['domain2'], data_sub['path2'], data_sub['query2'], data_sub['fragment2'] = zip(*[urlsplit(i) for i in data_sub['Link']])
    data_sub['target'] = data_sub['Name']
    data_sub['source'] = data_sub['path2'].str.extract(r'/\s*([^\/]*)\s*\/', expand=False)
    data_index = data_sub[data_sub['source'] == data_sub['User Name']].index
    data_sub.drop(data_index, inplace=True)
    data_index2 = data_sub[data_sub['source'] == data_sub['Facebook Id']].index
    data_sub.drop(data_index2, inplace=True)
    return data_sub

def prep_sna_links(data):
    data_sub = data[data['Link'].notnull()]
    #data_sub = data_sub[data_sub.duplicated(['Link'], keep=False)]
    data_sub['target'] = data_sub['Name']
    data_sub = data_sub.rename(columns={'Link': "source"})
    return data_sub

def prep_sna_messages(data):
    data_sub = data[data['Message'].notnull()]
    #data_sub = data_sub[data_sub.duplicated(['Message'], keep=False)]
    data_sub['target'] = data_sub['Name']
    data_sub = data_sub.rename(columns={'Message': "source"})
    return data_sub

def pairwise_corr(data):
    data_sub_network = data[['source', 'target']]
    data_sub_counts = pd.DataFrame(data_sub_network.groupby(['target']).size().sort_values(ascending=False).reset_index())
    data_sub_counts = data_sub_counts.rename(columns={0: 'n'}).reset_index(drop=True)
    data_sub_network_counts = pd.merge(data_sub_network, data_sub_counts)
    vector_matrix = pd.get_dummies(data_sub_network['source']).T.dot(pd.get_dummies(data_sub_network['target'])).clip(0, 1)
    pairwise_cov_matrix = vector_matrix.cov()
    pairwise_cor = pairwise_cov_matrix.corr(method="pearson", min_periods=1)
    pages = list(pairwise_cov_matrix.columns)
    pairwise_cor_matrix = pd.DataFrame(pairwise_cor, columns = pages, index = pages)
    temp_mat = pairwise_cor_matrix[pairwise_cor_matrix.index.isin(data['Name'])]
    temp_df = pd.DataFrame(temp_mat.T.unstack().reset_index(name='correlation').sort_values('correlation', ascending=False))
    temp_df = temp_df.rename(columns={"level_0": 'target', "level_1": 'source'})
    #temp_df = temp_df.sort_values(['target','correlation'], ascending=False).reset_index(drop=True)
    temp_df = temp_df[temp_df['target'] != temp_df['source']]
    temp_df1 = temp_df[['target', 'source']].reset_index(drop=True)
    temp_df1 = pd.DataFrame(np.sort(temp_df1[['target','source']], axis=1)).reset_index(drop=True)
    temp_df2 = temp_df1[~temp_df1.duplicated()]
    temp_df2 = temp_df2.rename(columns={0: 'target', 1: 'source'}).reset_index(drop=True)
    temp_df3 = pd.merge(temp_df2, temp_df)
    global temp_df4
    temp_df4 = temp_df3[temp_df3['correlation'] > 0.5]
    return temp_df4

def prep_sna_domain(data):
    data_sub = data[data['Link'].notnull()]
    data_sub = data_sub[~data_sub['Link'].str.contains(r"facebook.com")]
    data_sub['protocol2'], data_sub['domain2'], data_sub['path2'], data_sub['query2'], data_sub['fragment2'] = zip(*[urlsplit(i) for i in data_sub['Link']])
    data_sub['target'] = data_sub['Facebook Id']
    data_sub['source'] = data_sub['domain2']
    data_sub = data_sub[pd.notnull(data_sub['source'])].reset_index(drop=True)
    return data_sub

def sna_weight(data):
    pairs = data[['Name','target','source']]
    pairs = pairs[pd.notnull(pairs['source'])]
    pairs_index = pairs[pairs['target'] == pairs['source']].index
    pairs.drop(pairs_index, inplace=True)
    pairs_weight = pd.DataFrame(pairs.groupby(['Name', 'target', 'source']).size().reset_index())
    pairs_weight = pairs_weight.rename(columns={0: 'weight'}).reset_index(drop=True)
    global pairs_weight_all
    pairs_weight_all = pairs_weight.sort_values(by='weight', ascending=False).reset_index(drop=True)
    return pairs_weight_all

def prep_batch_from_posts(data, minsize=0, listname='null'):
    df = data
    #remove everything starting w/permalink and after
    df['URL'] = df['URL'].replace(to_replace=r"/permalink.*", value='/', regex=True)
    df = df.groupby(['Name', 'URL']).size().to_frame().reset_index().sort_values(by=0, ascending=False)
    df['List'] = listname
    df = df.rename(columns={"URL": "Page or Account URL"}).reset_index(drop=True)
    df = df.drop_duplicates()
    return df[['Page or Account URL', 'List']]

def socialnet_links(data2):
    prepsna = prep_sna_links(data2)
    pairwise_correlation = pairwise_corr(prepsna)
    G = nx.from_pandas_edgelist(pairwise_correlation, source='source', target='target', edge_attr=True)
    return G

def socialnet_messages(data2):
    prepsna = prep_sna_messages(data2)
    pairwise_correlation = pairwise_corr(prepsna)
    G = nx.from_pandas_edgelist(pairwise_correlation, source='source', target='target', edge_attr=True)
    return G
    #return G

def socialnet_pages(data1):
    prepsna = prep_sna(data1)
    pairwise_correlation = pairwise_corr(prepsna)
    G = nx.from_pandas_edgelist(pairwise_correlation, source='source', target='target', edge_attr=True)
    return G
    #return G

def communities_getter(G):
    partition = community_louvain.best_partition(G)
    return partition

def clean_text(data):
    stop_words = set(stopwords.words('english'))
    data = data.lower() #this may trigger a  warning...
    data = ' '.join([word for word in data.split(' ') if word not in stop_words])
    #data = ' '.join([word for word in data.split(' ') if word not in stopwords])
    data = data.encode('ascii', 'ignore').decode()
    data = re.sub(r'https*\S+', ' ', data)
    data = re.sub(r'@\S+', ' ', data)
    data = re.sub(r'#\S+', ' ', data)
    data = re.sub(r'\'\w+', '', data)
    data = re.sub('[%s]' % re.escape(string.punctuation), ' ', data)
    data = re.sub(r'\w*\d+\w*', '', data)
    data = re.sub(r'\s{2,}', ' ', data)
    data = ' '.join([word for word in data.split(' ') if word not in stop_words])
    #data = ' '.join([word for word in data.split(' ') if word not in stopwords])
    return data

def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

def topic_network(data2):
    data_topics = data2[data2['Message'].notnull()]
    data_topics['text'] = data_topics['Message'].apply(clean_text)
    data_topics = data_topics[data_topics['text'].notnull()]
    data_topics['text'].reset_index(drop=True)
    common_words = get_top_n_words(data_topics['text'], 20)
    common_words_df = pd.DataFrame(common_words)
    return common_words_df

def bigram_network(data2):
    data_topics = data2[data2['Message'].notnull()]
    data_topics['text'] = data_topics['Message'].apply(clean_text)
    data_topics = data_topics[data_topics['text'].notnull()]
    data_topics['text'].reset_index(drop=True)
    bigrams = [(x, i.split()[j + 1]) for i in data_topics['text']
       for j, x in enumerate(i.split()) if j < len(i.split()) - 1]
    frequency_dist_bigrams = FreqDist(bigrams)
    common_bigrams_df = pd.DataFrame(frequency_dist_bigrams.most_common(20))
    return common_bigrams_df

def word_matrix(data):
    vectorizer = CountVectorizer(analyzer='word',
                              token_pattern=r'\b[a-zA-Z]{3,}\b',
                              ngram_range=(1, 1),
                              min_df=10)
    count_vectorized = vectorizer.fit_transform(data['text'])
    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    vectorized = tfidf_transformer.fit_transform(count_vectorized)
    vector_matrix = pd.DataFrame(vectorized.toarray(),
             index=['message '+str(i)
                    for i in range(1, 1+len(data['text']))],
             columns=vectorizer.get_feature_names())

def word_pairwise_corr(vector_matrix):
    pairwise_cov_matrix = vector_matrix.cov()
    pairwise_cor = np.corrcoef(pairwise_cov_matrix)
    words = list(pairwise_cov_matrix.columns)
    pairwise_cor_matrix = pd.DataFrame(pairwise_cor, columns = words, index = words)

def corr_network(matrix, topics, words):
    temp_mat = matrix[matrix.index.isin(topics)]
    temp_df = pd.DataFrame(temp_mat.T.unstack().reset_index(name='correlation').sort_values('correlation', ascending=False))
    temp_df = temp_df.rename(columns={"level_0": 'topic', "level_1": 'word'})
    temp_df = temp_df.sort_values(['topic','correlation'], ascending=False).reset_index(drop=True)
    temp_df = temp_df[temp_df['topic'] != temp_df['word']]
    return temp_df.groupby(['topic'],as_index=False).apply(lambda x: x.nlargest(words, 'correlation'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/snowball', methods=['POST'])
def snowball():
    if request.method == 'POST':
        token = request.form['token']
        getlink = request.form['getlink']
        global call
        call = get_links(token, link=getlink)
        data = pd.DataFrame.from_dict(call['result']['posts'])
        data = data.rename(columns={"subscriberCount": "initialSubscriberCount", "platform": "initialPlatform"})
        data = pd.concat([data.drop(['account'], axis=1), data['account'].apply(pd.Series)], axis=1)
        data = pd.concat([data.drop(['statistics'], axis=1), data['statistics'].apply(pd.Series)], axis=1)
        data = pd.concat([data.drop(['actual'], axis=1), data['actual'].apply(pd.Series)], axis=1)
        #shares = data.groupby(['name', 'accountType']).size().to_frame().reset_index().rename(columns={0: 'shares'}).sort_values(by='shares', ascending=False)
        data = data.replace(np.nan,0)
        data['date'] = pd.to_datetime(data.date)
        data = data.sort_values(by='date')
        data['Reach'] = data['subscriberCount'].cumsum()
        data['Likes'] = data['likeCount'].cumsum()
        data['Shares'] = data['shareCount'].cumsum()
        data['Comments'] = data['commentCount'].cumsum()
        #plot time-series
        fig = px.scatter(data, x = 'date', y='Reach', size='subscriberCount', title="Reach", hover_name="name", hover_data=["Likes", "Shares", "Comments"])#color='platform') )
        #fig.add_trace(go.Scatter(x=df['date'], y=df['reach']))
        fig.update_xaxes(tickangle=45, tickfont=dict(size=8))
        fig.update_yaxes(tickfont=dict(size=8))
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        data_subset = data[['name', 'date', 'accountType', 'subscriberCount', 'likeCount', 'shareCount', 'commentCount']]
        columns = list(data_subset.columns.values)
        values = list(data_subset.values)
    return render_template('results.html', columns = columns, values = values, graphJSON = graphJSON)

@app.route('/export', methods=['POST'])
def export():
    if request.method == 'POST':
        type = request.form['options']
        listname = request.form['listname']
        batch = prep_batch(data=call, type=type, minsize=0, listname=listname)
        resp = make_response(df2.to_csv(index=False))
        resp.headers["Content-Disposition"] = "attachment; filename=" + listname + "_" + type + ".csv"
        resp.headers["Content-Type"] = "text/csv"
        return resp

@app.route('/communities')
def communities():
    return render_template('community.html')

@app.route("/import", methods=['POST'])
def mport():
      # get the uploaded file
      uploaded_file = pd.read_csv(request.files['file'])
      data = uploaded_file
      data.columns.values[0] = "Name"
      data['Likes at Posting'] = data['Likes at Posting'].replace(np.nan,0)
      data['Followers at Posting'] = data['Followers at Posting'].replace(np.nan,0)
      data['Subscribers'] = pd.to_numeric(data['Likes at Posting'] + data['Followers at Posting'])
      data['Subscribers'] = data['Subscribers'].replace(np.nan,0)
      data['Subscribers'] = data['Subscribers'].astype(int)
      global data1
      data1 = data[["Name", "User Name", "Facebook Id", "Subscribers", "URL", "Link"]]
      data1 = pd.DataFrame(data1)
      global data2
      data2 = data[["Name", "Subscribers", "Post Created", "Message", "URL", "Link", "Description"]]
      data2 = pd.DataFrame(data2)
      columns = list(data2.columns.values)
      values = list(data2.values)
      return render_template('analyze.html', columns = columns, values = values)

@app.route("/userinfluence", methods=['POST'])
def userinfluence():
    prepsna = prep_sna(data1)
    snaweight = sna_weight(prepsna)
    weight = snaweight.groupby('source').sum().sort_values(by='weight', ascending=False).reset_index().head(50)
    weight = weight[['source', 'weight']]
    columns = list(weight.columns.values)
    values = list(weight.values)
    return render_template('userinfluence.html', columns = columns, values = values)

@app.route("/domaininfluence", methods=['POST'])
def domaininfluence():
    prepsna = prep_sna_domain(data1)
    snaweight = sna_weight(prepsna)
    weight = snaweight.groupby('source').sum().sort_values(by='weight', ascending=False).reset_index().head(50)
    weight = weight[['source', 'weight']]
    columns = list(weight.columns.values)
    values = list(weight.values)
    return render_template('domaininfluence.html', columns = columns, values = values)

@app.route("/snagraph", methods=['POST'])
def snagraph():
    G = socialnet_pages(data1)
    pos = nx.fruchterman_reingold_layout(G, k=1/G.order())
    for n, p in pos.items():
        G.nodes[n]['pos'] = p
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    #color_line = []
    #for corr in G.edges:
    #    line = G.edges[corr]['correlation']
    #    color_line.append(line)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color="#888"),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        #mode='markers',
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            showscale=True,
        # colorscale options
        #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
        #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
        #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='Hot',
            reversescale=True,
            color=[],
            size = 10,
            #size=node_size,
            colorbar=dict(
                thickness=15,
                title='Connections to Node (More Connections = More Centrality)',
                xanchor='left',
                titleside='right'
                ),
                line_width=2))

    node_name = []
    for i in enumerate(G.nodes()):
        node_name.append(i[1])
    node_trace.text = node_name

#node_correlation = []
#for node, nbrsdict in G.adj.items():
#    for i in nbrsdict.values():
#        for j in i.values():
#            node_correlation.append(j)

#node_trace.marker.color = node_correlation

    node_adjacencies = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
    #node_text.append('# of connections: '+str(len(adjacencies[1])))
    node_trace.marker.color = node_adjacencies

    fig = go.Figure(data=[edge_trace, node_trace],
            layout=go.Layout(
                title='',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Graph Correlation = Greater Than 0.5",
                    showarrow=True,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
    columns = list(temp_df4.columns.values)
    values = list(temp_df4.values)
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('graph.html', graphJSON = graphJSON, columns = columns, values=values)

@app.route("/linkgraph", methods=['POST'])
def linkgraph():
    G = socialnet_links(data2)
    pos = nx.fruchterman_reingold_layout(G, k=1/G.order())
    for n, p in pos.items():
        G.nodes[n]['pos'] = p
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    #color_line = []
    #for corr in G.edges:
    #    line = G.edges[corr]['correlation']
    #    color_line.append(line)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color="#888"),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        #mode='markers',
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            showscale=True,
        # colorscale options
        #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
        #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
        #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='Hot',
            reversescale=True,
            color=[],
            size = 10,
            #size=node_size,
            colorbar=dict(
                thickness=15,
                title='Connections to Node (More Connections = More Centrality)',
                xanchor='left',
                titleside='right'
                ),
                line_width=2))

    node_name = []
    for i in enumerate(G.nodes()):
        node_name.append(i[1])
    node_trace.text = node_name

#node_correlation = []
#for node, nbrsdict in G.adj.items():
#    for i in nbrsdict.values():
#        for j in i.values():
#            node_correlation.append(j)

#node_trace.marker.color = node_correlation

    node_adjacencies = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
    #node_text.append('# of connections: '+str(len(adjacencies[1])))
    node_trace.marker.color = node_adjacencies

    fig = go.Figure(data=[edge_trace, node_trace],
            layout=go.Layout(
                title='',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Graph Correlation = Greater Than 0.5",
                    showarrow=True,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
    columns = list(temp_df4.columns.values)
    values = list(temp_df4.values)
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('linkgraph.html', graphJSON = graphJSON, columns = columns, values=values)

@app.route("/messagegraph", methods=['POST'])
def messagegraph():
    G = socialnet_messages(data2)
    pos = nx.fruchterman_reingold_layout(G, k=1/G.order())
    for n, p in pos.items():
        G.nodes[n]['pos'] = p
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    #color_line = []
    #for corr in G.edges:
    #    line = G.edges[corr]['correlation']
    #    color_line.append(line)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color="#888"),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        #mode='markers',
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            showscale=True,
        # colorscale options
        #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
        #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
        #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='Hot',
            reversescale=True,
            color=[],
            size = 10,
            #size=node_size,
            colorbar=dict(
                thickness=15,
                title='Connections to Node (More Connections = More Centrality)',
                xanchor='left',
                titleside='right'
                ),
                line_width=2))

    node_name = []
    for i in enumerate(G.nodes()):
        node_name.append(i[1])
    node_trace.text = node_name

#node_correlation = []
#for node, nbrsdict in G.adj.items():
#    for i in nbrsdict.values():
#        for j in i.values():
#            node_correlation.append(j)

#node_trace.marker.color = node_correlation

    node_adjacencies = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
    #node_text.append('# of connections: '+str(len(adjacencies[1])))
    node_trace.marker.color = node_adjacencies

    fig = go.Figure(data=[edge_trace, node_trace],
            layout=go.Layout(
                title='',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Graph Correlation = Greater Than 0.5",
                    showarrow=True,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
    columns = list(temp_df4.columns.values)
    values = list(temp_df4.values)
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('messagegraph.html', graphJSON = graphJSON, columns = columns, values=values)


@app.route("/sna", methods=['POST'])
def sna():
    global G2
    G2 = socialnet_pages(data1)
    return render_template('communitysearching.html')

@app.route("/communitydetected", methods=['GET', 'POST'])
def communitydetected():
    partition = communities_getter(G2)
    global communities_detected
    communities_detected = pd.DataFrame(partition.keys(), partition.values()).reset_index(level=[0])
    communities_detected.columns = ['id', 'Name']
    comms = communities_detected['id'].nunique()
    #else: comms = "0"
    return render_template('communitydetected.html', comms=comms)

@app.route("/communityselect", methods=['GET', 'POST'])
def communityselect():
    if request.method == 'POST':
        comid = request.form['comid']
        temp = communities_detected.loc[communities_detected['id'] == int(comid)]
        global temp_posts
        temp_posts = data1.loc[data1['Name'].isin(temp['Name'])]
        temp_groups = temp_posts.groupby(['Name']).size().to_frame().reset_index().sort_values(by=0, ascending=False)
        temp_groups = temp_groups.rename(columns={0: 'posts'}).reset_index(drop=True)
        temp_groups = pd.DataFrame(temp_groups)
        temp_groups = temp_groups[['Name', 'posts']]
        #comms = len(communities_detected)
        columns = list(temp_groups.columns.values)
        values = list(temp_groups.values)
        return render_template('communityselect.html', columns = columns, values = values, comid = comid)

@app.route('/exportcom', methods=['POST'])
def exportcom():
    if request.method == 'POST':
        #type = request.form['options']
        listname = request.form['listname']
        export_groups = data1[data1['Name'].isin(temp_posts['Name'])]
        batch = prep_batch_from_posts(data=export_groups, minsize=0, listname=listname)
        resp = make_response(batch.to_csv(index=False))
        resp.headers["Content-Disposition"] = "attachment; filename=" + listname + ".csv"
        resp.headers["Content-Type"] = "text/csv"
        return resp

@app.route('/dupelinks', methods=['POST'])
def dupelinks():
    dflinks = dupe_links(data2)
    columns = list(dflinks.columns.values)
    values = list(dflinks.values)
    return render_template('dupelinks.html', columns = columns, values = values)

@app.route('/dupemessages', methods=['POST'])
def dupemessages():
    dfmessages = dupe_messages(data2)
    columns = list(dfmessages.columns.values)
    values = list(dfmessages.values)
    return render_template('dupelinks.html', columns = columns, values = values)


@app.route("/topics", methods=['POST'])
def topics():
    df = topic_network(data2)
    matrix = word_matrix(df)
    word_pairwise_matrix = word_pairwise_corr(matrix)
    pairwise_cor_network = corr_network(pairwise_cor_matrix, common_words_df[0], 10)
    fig = px.bar(df, x = 0, y = 1)#color='platform') )
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    df2 = bigram_network(data2)
    fig2 = px.bar(df2, x = 0, y = 1)
    graphJSON2 = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('topics.html', graphJSON = graphJSON, graphJSON2 = graphJSON2)

if __name__ == '__main__':
    app.run(debug=True)
