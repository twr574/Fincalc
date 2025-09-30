# Demonstration code for fincalc

import time
from tabulate import tabulate

import fxrates

from fxcorrelation import Currency
from fxcorrelation import FxOption
from fxcorrelation import FxRateTable
from fxcorrelation import HistCorr
from fxcorrelation import ImpVolTable
from fxcorrelation import fx_createportfolio


from duration import Bond
from intrates import VasicekRate

from options import Asset
from options import EuropeanOption
from options import EuropeanBarrierOption
from options import BinaryOption
from options import AsianEuropeanOption

# %%

print('This is a demonstration of some of the features in this collection of modules.')

# %%

print('\n\n1. Live Exchange Rate Information')
time.sleep(1)

curr = 'GBP'
trans_curr = 'USD'

rates = fxrates.fxrate(curr,trans_curr)
print(f'\nThe current exchange rate for GBP/{trans_curr} is {rates[0]}.')

price = 100
fee = 0.03

cost = fxrates.transcost(price,curr,trans_curr,fee)[0]

print(f'\nAssuming a transaction fee of {fee*100}%,'
      f'\nit will cost {cost} {curr} to purchase an item worth {price} {trans_curr}.')

time.sleep(1)

print('\nScraping the internet for FX currency data...')

fxrates.fxscrape()

print('\nCompleted!')

time.sleep(3)

# %%

print('\n\n2. Cross-Market Correlation Positions')
time.sleep(1)

print('\nCreating some currency data:')

a = Currency('Currency A','A',0.03)
b = Currency('Currency B','B',0.07)
c = Currency('Currency C','C',0.06)

print(tabulate([[a.name,a.code,a.int_rate],
               [b.name,b.code,b.int_rate],
               [c.name,c.code,c.int_rate]],
               headers=['Currency','Code','Interest Rate'],tablefmt='pretty'))

time.sleep(1)

fx = FxRateTable({'A_B': 2.5, 'A_C':1.8})

tab = []
for x in fx.table:
    tab.append([x,round(fx.table[x],4)])
print(tabulate(tab,headers=['Currency','Exchange Rate'],tablefmt='pretty'))

time.sleep(1)

hist_corr = HistCorr({'A_B,A_C':-0.6})
# imp_vol[BA] means buy 1*B with A
imp_vol = ImpVolTable({'B_A bid':0.125, 'B_A ask':0.126,
                       'A_C bid':0.115, 'A_C ask':0.116,
                       'B_C bid':0.135, 'B_C ask':0.136},
                      expiry=0.25)

tab = []
for x in imp_vol.table:
    tab.append([x,round(imp_vol.table[x],4)])
print(tabulate(tab,headers=[f'FX Option\nExpiry:{imp_vol.expiry}','Implied Volatility'],tablefmt='pretty'))

time.sleep(1)

portfolio = fx_createportfolio(a,b,c,fx,hist_corr,imp_vol)

tab = []
for x in portfolio:
    if type(x) is FxOption:
        tab.append([f'Call on {x.curr1.code}/{x.curr2.code}',round(portfolio[x],4)])
    else:
        tab.append([x,round(portfolio[x],4)])
print(tabulate(tab,headers=['Item','Position'],tablefmt='pretty'))

time.sleep(3)

# %%
print('\n\n3. Interest Rate Hedging')
time.sleep(1)

bond = Bond('bond',10)
bond1 = Bond('bond1',5)
bond2 = Bond('bond2',20)
print(f'\n\nHedging a bond of maturity {bond.maturity} using bonds of maturities {bond1.maturity} and {bond.maturity}.')

const_rate = 0.08
print(f'Using the conventional duration hedge, with interest rate {const_rate}.')
[q1,q2,b1,b2,m1] = bond._hedge_dur_std(bond1,bond2,const_rate)
print(f'Buy {"%.5f" % b1} of Bond 1 and {"%.5f" % b2} of Bond 2 (present values).\n')

disturbance = 0.01
error = bond._acid_dur_std(bond1,bond2,q1,q2,const_rate,disturbance)
print(f'If interest rate changes by {disturbance}, the hedging error is {"%.5f" % error}.\n\n')

time.sleep(1)

vas_rate = VasicekRate('Vasicek',0.08,0.1,0.2,0.05)
print(f'Using the Vasicek duration hedge, with initial interest rate {vas_rate.r0}.')
[q1,q2,b1,b2,m1] = bond._hedge_dur_vasicek(bond1,bond2,vas_rate)
print(f'Buy {"%.5f" % b1} of Bond 1 and {"%.5f" % b2} of Bond 2 (present values).\n')

disturbance = 0.01
error = bond._acid_dur_vasicek(bond1,bond2,q1,q2,vas_rate,0.01)
print(f'If interest rate changes by {disturbance}, the hedging error is {"%.5f" % error}.\n\n')

time.sleep(3)

# %%
print('\n\n4. Option  Valuation')
time.sleep(1)

strikep = 100
expr = 1
s0 = 100

print(f"\n\nConstructing a series of Options on a single share of 'ACME Corp' currently worth {s0}.\n")
acme = Asset('ACME Corp.',s0,0.1)

# %%
rt_eu = EuropeanOption('EVC on ACME',acme,strike=strikep,option_type='call',expiry=expr,style='european')
print(f'Evaluating the value of a European Call with Expiry {expr}, Strike Price {strikep}.\n')
t = time.time()
v0 = rt_eu.black_scholes(int_rate=0.2,volatility=0.2)
t0 = time.time() - t
timesteps = 1000
print(f'Using a Binomial lattice over {timesteps} time steps...')
t = time.time()
v1 = rt_eu.binomial_value(timesteps,int_rate=0.2,volatility=0.2,coeffs='JR')
t1 = time.time() - t
print(f'Using a Trinomial lattice over {timesteps} time steps...')
t = time.time()
v2 = rt_eu.trinomial_value(timesteps,int_rate=0.2,volatility=0.2,coeffs='JR')
t2 = time.time() - t
paths = 10000000
pathlength = 10
print(f'Performing Monte Carlo using {paths} paths over {pathlength} time steps...')
t = time.time()
v3 = rt_eu.mc_value(paths,pathlength,int_rate=0.2,volatility=0.2)
t3 = time.time() - t

print(tabulate([['Black Scholes'   ,"%.8f" % round(v0,8),0                           ,"%.8f" % round(t0,8)],
               ['Binomial Lattice' ,"%.8f" % round(v1,8),"%.8f" % round(abs(v1-v0),8),"%.8f" % round(t1,8)],
               ['Trinomial Lattice',"%.8f" % round(v2,8),"%.8f" % round(abs(v2-v0),8),"%.8f" % round(t2,8)],
               ['Monte Carlo'      ,"%.8f" % round(v3,8),"%.8f" % round(abs(v3-v0),8),"%.8f" % round(t3,8)]],
               headers=['European Call','Value','Error','Time Elapsed (s)'],tablefmt='pretty'))

time.sleep(3)

# %%
cashreturn = 100
rt_bin = BinaryOption('C.O.N. on ACME',acme,strike=strikep,option_type='cash-or-nothing call',expiry=expr,style='binary',cash=cashreturn)
print(f'\nEvaluating the value of a Cash or Nothing Call with Cash {cashreturn}, Expiry {expr}, Strike Price {strikep}.\n')
t = time.time()
u0 = rt_bin.black_scholes(int_rate=0.2,volatility=0.2)
t0 = time.time() - t
timesteps = 1000
print(f'Using a Binomial lattice over {timesteps} time steps...')
t = time.time()
u1 = rt_bin.binomial_value(timesteps,int_rate=0.2,volatility=0.2,coeffs='JR')
t1 = time.time() - t
print(f'Using a Trinomial lattice over {timesteps} time steps...')
t = time.time()
u2 = rt_bin.trinomial_value(timesteps,int_rate=0.2,volatility=0.2,coeffs='JR')
t2 = time.time() - t
paths = 100000
pathlength = 10
print(f'Performing Monte Carlo using {paths} paths over {pathlength} time steps...')
t = time.time()
u3 = rt_bin.mc_value(paths,timesteps,int_rate=0.2,volatility=0.2)
t3 = time.time() - t

print(tabulate([['Black Scholes'   ,"%.8f" % round(u0,8),0                           ,"%.8f" % round(t0,8)],
               ['Binomial Lattice' ,"%.8f" % round(u1,8),"%.8f" % round(abs(u1-u0),8),"%.8f" % round(t1,8)],
               ['Trinomial Lattice',"%.8f" % round(u2,8),"%.8f" % round(abs(u2-u0),8),"%.8f" % round(t2,8)],
               ['Monte Carlo'      ,"%.8f" % round(u3,8),"%.8f" % round(abs(u3-u0),8),"%.8f" % round(t3,8)]],
               headers=['Cash-or-Nothing Call','Value','Error','Time Elapsed (s)'],tablefmt='pretty'))

time.sleep(3)

# %%
barr = 70
rt_bar = EuropeanBarrierOption('EVC on ACME',acme,strike=strikep,option_type='call',expiry=expr,style='european',barrier_type='down and out',barrier=barr)
print(f'\nEvaluating the value of a Down and Out Barrier Call with Barrier {barr}, Expiry {expr}, Strike Price {strikep}.\n')

print(f'Using a Binomial lattice over {timesteps} time steps...')
t = time.time()
x1 = rt_bar.binomial_value(timesteps,int_rate=0.2,volatility=0.2,coeffs='JR')
t1 = time.time() - t
timesteps = 1000
print(f'Using a Trinomial lattice over {timesteps} time steps...')
t = time.time()
x2 = rt_bar.trinomial_value(timesteps,int_rate=0.2,volatility=0.2,coeffs='JR')
t2 = time.time() - t
paths = 100000
pathlength = 10
print(f'Performing Monte Carlo using {paths} paths over {pathlength} time steps...')
t = time.time()
x3 = rt_bar.mc_value(paths,timesteps,int_rate=0.2,volatility=0.2)
t3 = time.time() - t


timesteps_ref = 10000
print(f'Computing approximate reference solution using a trinomial lattice with {timesteps_ref} time steps...')
t = time.time()
x0 = rt_bar.trinomial_value(timesteps_ref,int_rate=0.2,volatility=0.2,coeffs='JR')
t0 = time.time() - t

print(tabulate([['Reference Solution',"%.8f" % round(x0,8),0                           ,"%.8f" % round(t0,8)],
               ['Binomial Lattice'   ,"%.8f" % round(x1,8),"%.8f" % round(abs(x1-x0),8),"%.8f" % round(t1,8)],
               ['Trinomial Lattice'  ,"%.8f" % round(x2,8),"%.8f" % round(abs(x2-x0),8),"%.8f" % round(t2,8)],
               ['Monte Carlo'        ,"%.8f" % round(x3,8),"%.8f" % round(abs(x3-x0),8),"%.8f" % round(t3,8)]],
               headers=['Barrier Call ','Value','Error (Approx.)','Time Elapsed (s)'],tablefmt='pretty'))

time.sleep(3)

# %%
rt_as = AsianEuropeanOption('Asian on ACME',acme,strike=strikep,option_type='average strike call',expiry=expr,style='asian',avg_mthd='arth')
print(f'\nEvaluating the value of an Asian (Average Strike) Call with arithmetic averaging, Strike Price {strikep}.\n')

paths = 10000
pathlength = 10
print(f'Performing Monte Carlo using {paths} paths over {pathlength} time steps...')
t = time.time()
w1 = rt_as.mc_value(paths,timesteps,int_rate=0.2,volatility=0.2)
t1 = time.time() - t
paths = 300000
pathlength = 10
print(f'Calculating approximate reference solution using {paths} paths over {pathlength} time steps...')
t = time.time()
w0 = rt_as.mc_value(paths,timesteps,int_rate=0.2,volatility=0.2)
t0 = time.time() - t

print(tabulate([['Reference Solution',"%.8f" % round(w0,8),0                           ,"%.8f" % round(t0,8)],
               [f'N = {paths}'       ,"%.8f" % round(w1,8),"%.8f" % round(abs(w1-w0),8),"%.8f" % round(t1,8)]],
               headers=['Asian Call','Value','Error (Approx.)','Time Elapsed (s)'],tablefmt='pretty'))

time.sleep(3)
