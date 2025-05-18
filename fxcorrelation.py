"""
Correlation position calculator. Uses inputted forex data to determine whether
it is appropriate to take on a cross-market position.
"""

import math
import numpy as np
import scipy as sp


class Currency():
    """
    Defines a currency for use with FX options. Also defines interest rates
    associated with the locale for the given currency.
    """

    def __init__(self,name,code,int_rate):
        self.name = name
        self.code = code
        self.int_rate = int_rate


class FxOption():
    "Purchase one unit of currency 2 (foreign) with currency 1 (base)."

    def __init__(self,curr1,curr2,strike,option_type='call',expiry=1):
        self.curr1 = curr1
        self.curr2 = curr2
        self.strike = strike
        self.option_type = option_type
        self.expiry = expiry

    def fx_blackscholes(self,s0,sig):
        """
        Black Scholes formula for a cross-market option to purchase a
        foreign currency. Also calculates delta and vega.
        """
        rd = self.curr1.int_rate
        rf = self.curr2.int_rate
        expiry = self.expiry
        value_x = self.strike*math.exp(-rd*expiry)
        value_s = s0*math.exp(-rf*expiry)
        sig_rt = sig*np.sqrt(expiry)
        #
        d1 = np.log(value_s/value_x)/sig_rt + 0.5*sig_rt
        d2 = d1 - sig_rt
        #
        if self.option_type == 'call':
            v = value_s * sp.stats.norm.cdf(d1,0,1) - value_x * sp.stats.norm.cdf(d2,0,1)
            delta = value_s/s0 * sp.stats.norm.cdf(d1,0,1)
            vega = value_s * sp.stats.norm.pdf(d1,0,1) * np.sqrt(expiry)
        elif self.option_type == 'put':
            v = value_x * sp.stats.norm.cdf(-d2,0,1) - value_s * sp.stats.norm.cdf(-d1,0,1)
            delta = - value_s/s0 * sp.stats.norm.cdf(-d1,0,1)
            vega = value_s * sp.stats.norm.pdf(d1,0,1) * np.sqrt(expiry)
        return v, delta, vega

    def fx_blackscholes2(self,fx,imp_vol,position):
        "Shorthand version of fx_blackscholes for correlation position calculations."
        s0 = fx.table[f'{self.curr2.code}_{self.curr1.code}']
        sig = imp_vol.table[f'{self.curr2.code}_{self.curr1.code} {position}']
        return self.fx_blackscholes(s0,sig)

    def fx_crossvega(self,imp_vol,hist_corr,hist_vol,opt2,key,fx,vega,pos):
        """
        Calculates the cross-vega dcall(CB)/dvol(*) where * = CA, AB,
        where a correlation position is taken on in the AB,AC market.
        """
        if pos == 1:
            position = 'ask'
            switch = 1
        else:
            position = 'bid'
            switch = 0
        vol_opt2 = imp_vol.table[f'{opt2.curr2.code}_{opt2.curr1.code} {position}']
        opt3_set = [self.curr1.code,self.curr2.code,opt2.curr1.code,opt2.curr2.code]
        opt3_set = [i for i in opt3_set if opt3_set.count(i) == 1]
        vol_opt3 = imp_vol.table[f'{opt3_set[0]}_{opt3_set[1]} {position}']
        corr = hist_corr.table[key]
        s0 = fx.table[f'{self.curr2.code}_{self.curr1.code}']
        vega = self.fx_blackscholes(s0,hist_vol[switch])[2]
        deriv = (vol_opt2 - corr*vol_opt3)/hist_vol[switch]
        crossvega = deriv * vega

        return crossvega


class FxRateTable():
    "Dictionary containing FX rates."

    def __init__(self,table):
        self.table = self._fxrate_closure(table)

    def _fxrate_closure(self,rates):
        """
        Closes a dictionary of interest rates (i.e. creates B_A where A_B exists,
        and creates A_C where A_B and B_C exist. Here, A_B = x means that 1*A = x*B.
        """
        recip = {}
        for rate in rates:
            recip.update({'_'.join((rate.split('_')[1],rate.split('_')[0])):1/rates[rate]})
        rates.update(recip)
        recip = {}
        for rate in rates:
            for rate2 in rates:
                if rate.split('_')[1] == rate2.split('_')[0] and '_'.join((rate.split('_')[0],rate2.split('_')[1])) not in rates and rate.split('_')[0] != rate2.split('_')[1]:
                    recip.update({'_'.join((rate.split('_')[0],rate2.split('_')[1])):rates[rate]*rates[rate2]})
        rates.update(recip)
        return rates


class HistCorr():
    "Dictionary containing historical correlation data."

    def __init__(self,table):
        self.table = self._histcorr_closure(table)

    def _histcorr_closure(self,corrs):
        "Closes a dictionary of correlation data (i.e. corr(Y,X) = corr(X,Y).)"
        recip = {}
        for corr in corrs:
            recip.update({f"{corr.split(',')[1]},{corr.split(',')[0]}":corrs[corr]})
        corrs.update(recip)
        return corrs


class ImpVolTable():
    "Dictionary containing implied volatilities."

    def __init__(self,table,expiry):
        self.table = self._imp_vol_closure(table)
        self.expiry = expiry

    def _imp_vol_closure(self,vols):
        "Closes an implied volatility dictionary (i.e. creates B_A where A_B exists)."
        recip = {}
        for vol in vols:
            recip.update({f"{vol.split(' ')[0].split('_')[1]}_{vol.split(' ')[0].split('_')[0]} {vol.split(' ')[1]}":vols[vol]})
        vols.update(recip)
        return vols

    def _corr(self,pair1,pair2):
        """
        Derives implied correlation on a cross rate, using implied volatility
        for two given currency pairs (must be given in the form C_A, A_B).
        """
        # Find correct volatilities
        u_bid = self.table[f'{pair1} bid']
        v_bid = self.table[f'{pair2} bid']
        u_ask = self.table[f'{pair1} ask']
        v_ask = self.table[f'{pair2} ask']
        w_bid = self.table[f"{pair1.split('_')[0]}_{pair2.split('_')[1]} bid"]
        w_ask = self.table[f"{pair1.split('_')[0]}_{pair2.split('_')[1]} ask"]
        # Derive implied correlations
        imp_corr_bid = (u_bid**2 + v_bid**2 - w_ask**2)/(2*u_bid*v_bid)
        imp_corr_ask = (u_ask**2 + v_ask**2 - w_bid**2)/(2*u_ask*v_ask)
        return [imp_corr_bid, imp_corr_ask]

    def _vol_hist(self,hist_corr,key):
        "Calculates the implied volatility according to historic correlation"
        bid = []
        ask = []
        for xmart in key.split(','):
            bid.append(self.table[f'{xmart} bid'])
            ask.append(self.table[f'{xmart} ask'])
        corr = hist_corr.table[key]
        vol_bid = np.sqrt(bid[0]**2 + bid[1]**2 - 2*corr*bid[0]*bid[1])
        vol_ask = np.sqrt(ask[0]**2 + ask[1]**2 - 2*corr*ask[0]*ask[1])
        return [vol_bid,vol_ask]


def fx_base(asset,base,origin,fx):
    "Changes an asset to a different currency base according to a fx table"
    return asset * fx.table[f'{origin}_{base}']


def fx_setposition(corr_bid_imp,corr_ask_imp,hist_corr,key,threshold=0.15):
    """
    Determines whether to take on a correlation position based on whether
    the market data and 'historically accurate' data differ by more than
    a certain threshold.
    """
    if abs(corr_bid_imp - hist_corr.table[key]) < threshold or abs(corr_ask_imp - hist_corr.table[key]) < threshold:
        return 0
    else:
        return np.sign(0.5*(corr_bid_imp + corr_ask_imp) - hist_corr.table[key])


def fx_deltahedge(qty,fx,imp_vol,pos):
    "Delta-hedges a forex portfolio by hedging each option individually."
    dhedged_qty = {}
    for opt in qty:
        curr1 = opt.curr1.code
        curr2 = opt.curr2.code
        dhedged_qty.update({curr1:0, curr2:0})
    # Loop over options in portfolio.
    for opt in qty:
        # domestic = 1, foreign = 2
        curr1 = opt.curr1.code
        curr2 = opt.curr2.code
        s0 = fx.table[f'{curr2}_{curr1}']
        if np.sign(qty[opt]) == 1:
            position = 'ask'
        else:
            position = 'bid'
        sig = imp_vol.table[f'{opt.curr2.code}_{opt.curr1.code} {position}']
        # Delta hedge the option.
        [value,delta,vega] = opt.fx_blackscholes(s0,sig)
        delta2 = value - delta*s0
        v1 = -delta2*qty[opt]
        v2 = -delta*qty[opt]
        # If all options are hedged, the portfolio is.
        dhedged_qty[opt.curr1.code] += v1
        dhedged_qty[opt.curr2.code] += v2
    # Update portfolio.
    qty.update(dhedged_qty)
    return qty


def fx_createportfolio(a,b,c,fx,hist_corr,imp_vol):
    """
    The main cross-market portfolio creation function.
    Admits currencies a,b,c as well as fx rates, current market data in the
    form of implied volatilities, and an 'historically accurate' correlation,
    which is believed to be true, but is not (yet) reflected in the markets.
    """

    imp_corr = imp_vol._corr(f'{c.code}_{a.code}',f'{a.code}_{b.code}')
    hist_vol = imp_vol._vol_hist(hist_corr,f'{a.code}_{b.code},{a.code}_{c.code}')
    # Determines whether to take on a position
    pos = fx_setposition(imp_corr[0],imp_corr[1],hist_corr,f'{a.code}_{b.code},{a.code}_{c.code}')
    if pos != 0:
        # If a position is taken, define cross-market options
        call_bc = FxOption(c,b,fx.table[f'{b.code}_{c.code}'],expiry=imp_vol.expiry)
        call_ac = FxOption(c,a,fx.table[f'{a.code}_{c.code}'],expiry=imp_vol.expiry)
        call_ba = FxOption(a,b,fx.table[f'{b.code}_{a.code}'],expiry=imp_vol.expiry)
        # Work out value, related quantities of the relevant calls
        [v_bc, del_bc, vega_bc] = call_bc.fx_blackscholes2(fx,imp_vol,'bid')
        [v_ac, del_ac, vega_ac] = call_ac.fx_blackscholes2(fx,imp_vol,'bid')
        [v_ba, del_ba, vega_ba] = call_ba.fx_blackscholes2(fx,imp_vol,'bid')
        # Convert vegas to base C
        vega_ba = fx_base(vega_ba,f'{c.code}',f'{a.code}',fx)
        # Calculates cross-vegas for the purposes of vega-hedging
        crossvega_ac = call_bc.fx_crossvega(imp_vol,hist_corr,hist_vol,call_ac,f'{a.code}_{b.code},{a.code}_{c.code}',fx,vega_bc,pos)
        crossvega_ba = call_bc.fx_crossvega(imp_vol,hist_corr,hist_vol,call_ba,f'{a.code}_{b.code},{a.code}_{c.code}',fx,vega_bc,pos)
        # Define vega-hedged portfolio
        portfolio = {call_bc: pos,
                     call_ac: -pos*crossvega_ac/vega_ac,
                     call_ba: -pos*crossvega_ba/vega_ba}
        # Delta-hedge the vega-hedged portfolio
        portfolio = fx_deltahedge(portfolio,fx,imp_vol,pos)
    else:
        # No position taken
        portfolio = {}
        print('No position taken.')
    return portfolio
