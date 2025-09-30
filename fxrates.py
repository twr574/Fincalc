""" A series of functions related to FX rates."""

import urllib.request as ureq
from urllib.error import HTTPError, URLError
from bs4 import BeautifulSoup as bs
import datetime
import json
import pandas as pd
import numpy as np
import warnings

opener = ureq.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]


def fxreadtable(rate_url,table_id=0):
    "Obtain the table of rates"
    rate_page = opener.open(rate_url)
    html_bytes = rate_page.read()
    rate_html = bs(html_bytes, 'html.parser')
    table = rate_html.body.find_all('table')[table_id]
    if table.tbody.find_all('tr') == []:
        raise ValueError('Empty Data Frame')
    return table


def fxcrawl(curr):
    "Crawl sites for FX rates tables associated with the input currency."
    lst = []
    fxsites = open('fx\\fxsites.json','r')
    fxsitesdata = json.load(fxsites)
    for site in fxsitesdata:
        try:
            rate_page = opener.open(eval(site['url']))
            html_bytes = rate_page.read()
            rate_html = bs(html_bytes, 'html.parser')
            table = rate_html.body.find_all('table')[eval(site['table_id'])]
            # For each row (currency), create a dict. for the FX rate.
            for row in table.tbody.find_all('tr'):
                columns = row.find_all('td')
                currency = eval(site['currency_id'])
                name = eval(site['currency_name'])
                validdate = eval(site['valid_date'])
                bid = eval(eval(site['bid_price']))
                ask = eval(eval(site['ask_price']))
                source = site['site_id']
                curr_present = next((curr for curr in lst if curr['Currency'] == currency),0)
                if bid + ask > 0:
                    if curr_present == 0:
                        lst.append({'Currency':currency,
                                    'Name':name,
                                    'Valid Date':validdate,
                                    'Bid':bid,
                                    'Ask':ask,
                                    'Source':source})
                    elif datetime.datetime.strptime(curr_present['Valid Date'],'%d/%m/%Y') < datetime.datetime.strptime(validdate,'%d/%m/%Y') and bid + ask > 0:
                        lst.remove(curr_present)
                        lst.append({'Currency':currency,
                                    'Name':name,
                                    'Valid Date':validdate,
                                    'Bid':bid,
                                    'Ask':ask,
                                    'Source':source})
        except (HTTPError,IndexError,URLError):
            pass
        # Create the dataframe out of the list of dicts.
    if lst == []:
        raise ValueError(f'Could not retrieve any valid currency data for {curr.upper()}.')
    df = pd.DataFrame(lst)
    fxsites.close()
    return df


def fxbid(base,curr2):
    "Bid price to buy a unit of base currency using currency 2."
    try:
        table = fxcrawl(curr2)
        table.to_csv(f'fx\\fx_{curr2.upper()}.csv')
    except URLError:
        try:
            table = pd.read_csv(f'fx\\fx_{curr2.upper()}.csv')
            table.to_csv(f'fx\\fx_{curr2.upper()}.csv')
        except FileNotFoundError:
            raise ValueError(f'Could not retrieve any valid currency data for {curr2.upper()}.')
    row = table.loc[table.Currency == base.upper()]
    row = row.reset_index(drop=True)
    return [float(row.Bid[0]), row['Valid Date'][0]]


def fxask(base,curr2):
    "Ask price to buy a unit of base currency using currency 2."
    try:
        table = fxcrawl(curr2)
        table.to_csv(f'fx\\fx_{curr2.upper()}.csv')
    except URLError:
        try:
            table = pd.read_csv(f'fx\\fx_{curr2.upper()}.csv')
        except FileNotFoundError:
            raise ValueError(f'Could not retrieve any valid currency data for {curr2.upper()}.')
    row = table.loc[table.Currency == base.upper()]
    row = row.reset_index(drop=True)
    return [float(row.Ask[0]), row['Valid Date'][0]]


def fxrate(base,curr2):
    "Calculates the value of a unit of base currency using currency 2."
    try:
        table = fxcrawl(curr2)
    except (URLError,ValueError):
        try:
            table = pd.read_csv(f'fx\\fx_{curr2.upper()}.csv')
        except FileNotFoundError:
            raise ValueError(f'Could not retrieve any valid currency data for {curr2.upper()}.')
    row = table.loc[table.Currency == base.upper()]
    row = row.reset_index(drop=True)
    return [(float(row.Bid[0]) + float(row.Ask[0]))*0.5, row['Valid Date'][0],
            float(row.Bid[0]), float(row.Ask[0])]


def fxscrape():
    "Scrapes FX data and saves it to .csv files for later use."
    for curr in ['USD','EUR','JPY','GBP','CNY','AUD','CAD','CHF','HKD','SGD',
                 'SEK','INR','BRL','PLN','ILS','TRY']:
        try:
            table = fxcrawl(curr)
            table.to_csv(f'fx\\fx_{curr.upper()}.csv')
        except (ValueError,HTTPError):
            warnings.warn(f'Could not retrieve any valid currency data for {curr.upper()}.')


def transcost(price,local,curr,fee=0.0275):
    "Calculates the cost, in a local currency, of a transaction in another currency."
    [rate,date,lo,hi] = fxrate(curr,local)
    cost = np.round((1+fee)*price*np.array([rate,lo,hi]),2)
    return cost
