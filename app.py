from flask import Flask, render_template, url_for, request, make_response, flash
from flask_bootstrap import Bootstrap
import pandas as pd
import numpy as np
import requests
import os
import json
import csv
from urllib.parse import urlsplit

app = Flask(__name__)
Bootstrap(app)
# Upload folder
UPLOAD_FOLDER = 'static/files'
app.config['UPLOAD_FOLDER'] =  UPLOAD_FOLDER

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

@app.route('/community')
def community():
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

@app.route("/analyze", methods=['POST'])
def analyze():
    prepsna = prep_sna_group(data1)
    pairs = prepsna[['Name','target','source']]
    pairs = pairs[pd.notnull(pairs['source'])]
    pairs_index = pairs[pairs['target'] == pairs['source']].index
    pairs.drop(pairs_index, inplace=True)
    pairs_weight = pd.DataFrame(pairs.groupby(['Name', 'target', 'source']).size().reset_index())
    pairs_weight = pairs_weight.rename(columns={0: 'weight'}).reset_index(drop=True)
    pairs_weight = pairs_weight.sort_values(by='weight', ascending=False).reset_index(drop=True)
    weight = pairs_weight.groupby('source').sum().sort_values(by='weight', ascending=False).reset_index().head(50)
    weight = weight[['source', 'weight']]
    columns = list(weight.columns.values)
    values = list(weight.values)
    return render_template('graph.html', columns = columns, values = values)

if __name__ == '__main__':
    app.run(debug=True)
