from flask import Flask, render_template, url_for, request, make_response
import csv
from flask_bootstrap import Bootstrap
import pandas as pd
import requests
import json
from io import StringIO
import csv

app = Flask(__name__)
Bootstrap(app)

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/linkcheck', methods=['POST'])
def linkcheck():
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
    #return render_template('results.html', output = shares)
    return render_template('results.html', columns = columns, values = values)

@app.route('/export', methods=['POST'])
def export():
    if request.method == 'POST':
        type = request.form['options']
        listname = request.form['listname']
        batch = prep_batch(data=call, type=type, minsize=0, listname=listname)
        #return render_template('results.html', output = shares)
        resp = make_response(df2.to_csv(index=False))
        resp.headers["Content-Disposition"] = "attachment; filename=" + listname + "_" + type + ".csv"
        resp.headers["Content-Type"] = "text/csv"
        return resp

if __name__ == '__main__':
    app.run(debug=True)
