""" A series of functions related to FX rates."""

import urllib.request as ureq
from urllib.error import URLError
from bs4 import BeautifulSoup as bs
import pandas as pd
import numpy as np

opener = ureq.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]


def fxcrawl(curr):
    "Crawl ferates.com for the Mastercard FX rates associated with the input currency."
    # Obtain the table of rates
    rate_url = f'https://ferates.com/mastercard/{curr.lower()}'
    rate_page = opener.open(rate_url)
    html_bytes = rate_page.read()
    rate_html = bs(html_bytes, 'html.parser')
    table = rate_html.body.table
    lst = []
    # For each row (currency), create a dict. for the FX rate
    for row in table.tbody.find_all('tr'):
        columns = row.find_all('td')
        currency = columns[0].text.strip()
        name = columns[1].text.strip()
        validdate = columns[2].text.strip()
        bid = columns[3].text.strip().split('\n')[0]
        ask = columns[4].text.strip().split('\n')[0]
        lst.append({'Currency':currency,
                    'Name':name,
                    'Valid Date':validdate,
                    'Bid':bid,
                    'Ask':ask})
        # Create the dataframe out of the list of dicts.
        df = pd.DataFrame(lst)
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
            raise ValueError('Could not retrieve any valid currency data.')
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
            raise ValueError('Could not retrieve any valid currency data.')
    row = table.loc[table.Currency == base.upper()]
    row = row.reset_index(drop=True)
    return [float(row.Ask[0]), row['Valid Date'][0]]


def fxrate(base,curr2):
    "Calculates the value of a unit of base currency using currency 2."
    try:
        table = fxcrawl(curr2)
    except URLError:
        try:
            table = pd.read_csv(f'fx\\fx_{curr2.upper()}.csv')
        except FileNotFoundError:
            raise ValueError('Could not retrieve any valid currency data.')
    row = table.loc[table.Currency == base.upper()]
    row = row.reset_index(drop=True)
    return [(float(row.Bid[0]) + float(row.Ask[0]))*0.5, row['Valid Date'][0],
            float(row.Bid[0]), float(row.Ask[0])]


def fxscrape():
    "Scrapes FX data and saves it to .csv files for later use."
    for curr in ['USD','EUR','JPY','GBP','CNY','AUD','CAD','CHF','HKD','SGD',
                 'SEK','INR','BRL','PLN','ILS','TRY']:
        table = fxcrawl(curr)
        table.to_csv(f'fx\\fx_{curr.upper()}.csv')


def transcost(price,local,curr,fee=0.0275):
    "Calculates the cost, in a local currency, of a transaction in another currency."
    [rate,date,lo,hi] = fxrate(curr,local)
    cost = np.round((1+fee)*price*np.array([rate,lo,hi]),2)
    return cost
