from flask import Flask, render_template, url_for, request, make_response, flash, send_file
from flask_bootstrap import Bootstrap
import pandas as pd
import numpy as np
import requests
import os
import json
import csv
from urllib.parse import urlsplit
import networkx as nx
from networkx.algorithms import community
from matplotlib.figure import Figure
from io import BytesIO
import base64
from rq import Queue
from rq.job import Job
from worker import conn
import time

app = Flask(__name__)
Bootstrap(app)
#app.secret_key = os.urandom(24)
q = Queue(connection=conn)

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

def prep_sna_group(data):
    #remove rows where there is no link  value
    data_sub = data[data['Link'] != 0]
    #remove rows where link contains 'photo.php' - there is no user id handle in those links
    data_sub = data_sub[~data_sub['Link'].str.contains(r"photo.php")]
    #find + remove '/groups' in link
    data_sub['Link'] = data_sub['Link'].replace(to_replace=r"/groups", value='', regex=True)
    #find + remove '/photos' in link
    data_sub['Link'] = data_sub['Link'].replace(to_replace=r"/photos", value='', regex=True)
    #find + remove "?" in link
    data_sub['Link'] = data_sub['Link'].replace(to_replace=r"\?", value='', regex=True)
    #find + remove "fbid=" in link
    data_sub['Link'] = data_sub['Link'].replace(to_replace=r"fbid=", value='', regex=True)
    #find + remove '&" + everything afterwards in link
    data_sub['Link'] = data_sub['Link'].replace(to_replace=r"\&.*", value='/', regex=True)
    #parse cleaned up link url to separate components into different columns
    data_sub['protocol2'], data_sub['domain2'], data_sub['path2'], data_sub['query2'], data_sub['fragment2'] = zip(*[urlsplit(i) for i in data_sub['Link']])
    #keep only rows where domain is facebook.com
    data_sub = data_sub[data_sub['domain2'] == 'www.facebook.com']
    #change Facebook Id value to 'target' (this is id of the group where post was shared)
    data_sub['target'] = data_sub['Facebook Id']
    #change path2 value to 'source' (this is the id of the user that targeted the group)
    data_sub['source'] = data_sub['path2'].str.extract(r'/\s*([^\/]*)\s*\/', expand=False)
    #add column 'group' that is the same as 'Group Name'. in some functions, space btwn Group and Name causes probs
    #data_sub['group'] = data_sub['name']
    #remove any rows where 'source' is null -- probably caused by  find + remove error...to be examined
    data_sub = data_sub[pd.notnull(data_sub['source'])].reset_index(drop=True)
    #return the final cleaned table
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
    #if type == 'pages':
    #    df1 = df.loc[((df['accountType'] == 'facebook_page') & (df[0] > minsize))]
    #else: #need to fix the else to set to 'groups' as an option -- not a big deal right now
    #    df1 = df.loc[((df['accountType'] == 'facebook_group') & (df[0] > minsize))]
    df['List'] = listname
    df = df.rename(columns={"URL": "Page or Account URL"}).reset_index(drop=True)
    df = df.drop_duplicates()
    return df[['Page or Account URL', 'List']]

def communities_getter(data1):
    prepsna = prep_sna_group(data1)
    snaweight = sna_weight(prepsna)
    G = nx.from_pandas_edgelist(snaweight, source='source', target='Name', edge_attr=True)
    communities_generator = community.girvan_newman(G)
    top_level_communities = next(communities_generator)
    next_level_communities = next(communities_generator)
    global communities_detected
    communities_detected = sorted(map(sorted, next_level_communities))
    return communities_detected

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
        data = pd.concat([data.drop(['account'], axis=1), data['account'].apply(pd.Series)], axis=1)
        shares = data.groupby(['name', 'accountType']).size().to_frame().reset_index().rename(columns={0: 'shares'}).sort_values(by='shares', ascending=False)
        columns = list(shares.columns.values)
        values = list(shares.values)
    return render_template('results.html', columns = columns, values = values)

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
      data = data.replace(np.nan,0)
      data['Subscribers'] = pd.to_numeric(data['Likes at Posting'] + data['Followers at Posting'])
      data = data[["Name", "Facebook Id", "Subscribers", "URL", "Link"]]
      global data1
      data1 = pd.DataFrame(data)
      columns = list(data1.columns.values)
      values = list(data1.values)
      return render_template('analyze.html', columns = columns, values = values)

@app.route("/userinfluence", methods=['POST'])
def userinfluence():
    prepsna = prep_sna_group(data1)
    snaweight = sna_weight(prepsna)
    weight = snaweight.groupby('source').sum().sort_values(by='weight', ascending=False).reset_index().head(50)
    weight = weight[['source', 'weight']]
    columns = list(weight.columns.values)
    values = list(weight.values)
    return render_template('userinfluence.html', columns = columns, values = values)

@app.route("/sna", methods=['POST'])
def sna():
    return render_template('graph.html')

@app.route("/snagraph", methods=['POST'])
def snagraph():
    prepsna = prep_sna_group(data1)
    snaweight = sna_weight(prepsna)
    weight_net = snaweight[snaweight['source'].isin(snaweight['source'].value_counts()[snaweight['source'].value_counts() > 1].index)]
    weight_net = snaweight[snaweight['weight'] > 1]
    groups = list(weight_net['Name'].unique())
    shared = list(weight_net['source'].unique())
    node_labels = dict(zip(groups, groups))
    shared_labels = dict(zip(shared, shared))
    G = nx.from_pandas_edgelist(weight_net, source='source', target='Name', edge_attr=True)
    layout = nx.spring_layout(G, k=10/G.order())
    dc = nx.degree_centrality(G)
    hi_degree_targets = [k for k, v in dc.items() if k in groups and v > 0.03]
    node_labels_top = dict(zip(hi_degree_targets, hi_degree_targets))
    #color based on degree size leveraging nx spectrum
    node_color = [20000.0 * G.degree(v) for v in G]
    #same with size
    node_size =  [v * 10000 for v in dc.values()]
    #set plotting area
    fig = Figure(figsize=(25,25))
    #plot nodes & edges
    nx.draw_networkx_nodes(G, layout, node_size=node_size, node_color=node_color)
    nx.draw_networkx_edges(G, layout, width=1, alpha=0.5, edge_color='gray', style='dashed')
    #add labels for just the high degree targets
    nx.draw_networkx_labels(G, layout, labels=node_labels_top, font_size=15)
    img = BytesIO()
    fig.savefig(img)
    #img.seek(0)
    #data = base64.b64encode(img.getbuffer()).decode("ascii")
    return send_file(img, mimetype='image/png')

@app.route("/communitydetected", methods=['GET', 'POST'])
def communitydetected():
    from app import communities_getter
    global job
    job = q.enqueue_call(func=communities_getter, args=(data1,), result_ttl=5000)
    while not job.is_finished:
        job.get_status()
        time.sleep(5)
        #print(job.get_status())
    result = Job.fetch(job.id, connection=conn)
    if result.is_finished:
        comms = len(result.result)
    else: comms = "0"
    return render_template('communitydetected.html', comms=comms)

@app.route("/communityselect", methods=['GET', 'POST'])
def communityselect():
    result = Job.fetch(job.id, connection=conn)
    if result.is_finished:
        communities_detected = result.result
    else:
        print('boo')
    #the above all works, but need to add 'loading page' that redirects when finished
    if request.method == 'POST':
        comid = request.form['comid']
        temp = pd.DataFrame(communities_detected[int(comid)], columns=['Name'])
        global temp_posts
        temp_posts = data1.loc[data1['Name'].isin(temp['Name'])]
        temp_groups = temp_posts.groupby(['Name']).size().to_frame().reset_index().sort_values(by=0, ascending=False)
        temp_groups = temp_groups.rename(columns={0: 'posts'}).reset_index(drop=True)
        temp_groups = pd.DataFrame(temp_groups)
        temp_groups = temp_groups[['Name', 'posts']]
        #comms = len(communities_detected)
        columns = list(temp_groups.columns.values)
        values = list(temp_groups.values)
        return render_template('communityselect.html', columns = columns, values = values, comid = comid, comms=comms)

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

if __name__ == '__main__':
    app.run(debug=True)
