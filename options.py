"""
Contains some simple calculators for financial derivatives.
"""

import math
import os
import numpy as np
import scipy as sp

os.system("")  # enables ansi escape characters in terminal
COLOR = {"HEADER": "\033[95m","BLUE": "\033[94m","GREEN": "\033[92m","RED": "\033[91m","ENDC": "\033[0m",}


class Asset():
    "Defines an asset class."

    def __init__(self,name,asset_value,div_yield):
        self.name = name
        self.asset_value = asset_value
        self.div_yield = div_yield

    def _mc_sim_path(self,no_paths,path_length,expiry,int_rate,sigma):
        "Simulates paths for the value of an asset."
        s0 = self.asset_value
        div = self.div_yield
        dt = expiry/path_length
        rtdt = np.sqrt(dt)

        # Create Wiener Processes
        w = np.zeros((no_paths,path_length+1))
        rndm_pts = np.random.normal(size=(no_paths,path_length),scale=rtdt)
        for i in range(path_length):
            w[:,i+1] = w[:,i] + rndm_pts[:,i]

        # Evolve Asset using exact GBM (not Euler!)
        s = np.zeros((no_paths,path_length+1))
        s[:,0] = s0
        for i in range(path_length):
            s[:,i+1] = s[:,i]*np.exp(((int_rate - div - sigma**2 /2)*dt + sigma*(w[:,i+1]-w[:,i])))
        return s


class Option():
    """
    Defines the class of options.
    The underlying assset, strike price, option type (e.g. put/call) and expiry time are
    attributes which are defined for all options.
    """

    def __init__(self,name,asset,*,strike,option_type,expiry):
        self.name = name
        self.asset = asset
        self.strike = strike
        self.option_type = option_type
        self.expiry = expiry

    def payoff(self,asset_value,strike,option_type,*,asset_avg=None):
        "Defines the payoff functions for a variety of options."

        match option_type:
            case 'call':
                return max(asset_value - strike,0)
            case 'put':
                return max(strike - asset_value,0)
            case 'cash-or-nothing call':
                return self.cash*(asset_value - strike >= 0)
            case 'cash-or-nothing put':
                return self.cash*(strike - asset_value >= 0)
            case 'asset-or-nothing call':
                return asset_value*(asset_value - strike >= 0)
            case 'asset-or-nothing put':
                return asset_value*(strike - asset_value >= 0)
            case 'average strike call':
                return max(asset_value - asset_avg,0)
            case 'average strike put':
                return max(asset_avg - asset_value,0)
            case 'average rate call':
                return max(asset_avg - strike,0)
            case 'average rate put':
                return max(strike - asset_avg,0)


class LatticeValuedOption(Option):
    """
    Options which can be valued using lattice methods,
    including European, American and Binary options.
    """

    def __init__(self,name,asset,*,strike,option_type,expiry,style):
        self.name = name
        self.asset = asset
        self.strike = strike
        self.option_type = option_type
        self.expiry = expiry
        self.style = style
        #
        if style not in ['european','american','binary']:
            raise ValueError('Invalid option style for lattice method (must choose e.g., european, american).')
        if style in ['european','american'] and option_type not in ['call', 'put']:
            raise ValueError('Incompatible option style/type.')

    def binomial_value(self,time_steps,u=None,d=None,p=None,*,int_rate,volatility,coeffs=None):
        "Calculates the value of a Lattice-valued option using a binomial lattice."
        T = self.expiry
        r = int_rate
        sig = volatility
        div = self.asset.div_yield
        dt = T/time_steps
        # Determine coeffs to match expectation, variance
        if coeffs is None:
            if None in [u,d,p]:
                raise ValueError('Must specify lattice coefficients u,d,p.')
        elif coeffs == "CRR":  # Cox, Rubenstein, Ross, ud = 1
            a = (math.exp(-(r-div)*dt) + math.exp(((r-div)+sig**2)*dt))/2
            u = a + math.sqrt(a**2 - 1)
            d = 1/u
            p = (math.exp((r-div)*dt)-d)/(u-d)
        elif coeffs == "JR":  # Jarrow-Rudd, p = 1/2
            a = math.sqrt(math.exp(sig**2*dt)-1)
            u = math.exp((r-div)*dt)*(1+a)
            d = math.exp((r-div)*dt)*(1-a)
            p = 1/2
        if u<=1 or d<=0 or d>=1 or p<=0 or p>=1:
            raise ValueError("Invalid coefficient ranges. In some cases, this may be due to not enough time steps.")
        # Evolve asset value and initalise payoff
        s = list()
        v = list()
        s.append([self.asset.asset_value])
        v.append([0])
        for i in range(1,time_steps+1):
            s.append([s[0][0]* u**j * d**(i-j) for j in range(i+1)])
            v.append([0 for j in range(i+1)])
        # Evaluate nodal payoffs
        erdt = math.exp(-r*dt)
        v[-1] = [self.payoff(x,self.strike,self.option_type) for x in s[-1]]
        #
        return self._bin_value(time_steps,v,p,s,erdt)

    def trinomial_value(self,time_steps,u=None,m=None,d=None,pu=None,pm=2/3,pd=None,*,int_rate,volatility,coeffs=None):
        "Calculates the value of a Lattice-valued option using a trinomial lattice."
        T = self.expiry
        r = int_rate
        sig = volatility
        div = self.asset.div_yield
        dt = T/time_steps
        # Determine coeffs to match expectation, variance
        if coeffs is None:
            if None in [u,m,d,pu,pm]:
                raise ValueError('Must specify lattice coefficients u,m,d,pu')
        elif coeffs == "CRR":  # CRR-like coeffs, m = 1
            a = math.exp(sig*math.sqrt(dt))
            b = math.sqrt(dt/(12*sig**2))*(r-div-sig**2/2)
            [u, m, d] = [a, 1, 1/a]
            [pu, pm, pd] = [1/6 + b, 2/3, 1/6 - b]
        elif coeffs == 'JR':  # JR-like coeffs, pu = pd = 1/6
            a = math.exp((r-div-sig**2/2)*dt)
            b = math.exp(sig*math.sqrt(3*dt))
            [u, m, d] = [a*b, a, a/b]
            [pu,pm, pd] = [1/6, 2/3, 1/6]
        if u<=1 or u<=m or m<=d or d>=1 or d<=0 or not [pu,pm,pd]>[0,0,0] or not [pu,pm,pd]<[1,1,1] or round(pu+pm+pd,15) != 1:
            raise ValueError("Invalid coefficient ranges. In some cases, this may be due to not enough time steps.")
        # Evolve asset value
        s = list()
        v = list()
        s.append([self.asset.asset_value])
        v.append([0])
        for i in range(1,time_steps+1):
            s.append([s[0][0]* m**j * d**(i-j) for j in range(i+1)] + [s[0][0] * u**j * m**(i-j) for j in range(1,i+1)])
            v.append([0 for j in range(i+1)])
        # Evaluate nodal payoffs
        erdt = math.exp(-r*dt)
        v[-1] = [self.payoff(x,self.strike,self.option_type) for x in s[-1]]
        #
        return self._tri_value(time_steps,v,pu,pm,pd,s,erdt)


class MCValuedOption(Option):
    """
    Options which can be valued using Monte Carlo methods,
    including European and Asian (i.e. averaging) options.
    """

    def __init__(self,no_paths,path_length,*,int_rate=None,volatility=None):
        self.no_paths = no_paths
        self.path_length = path_length
        self.int_rate = int_rate
        self.volatility = volatility

    def mc_value(self,no_paths,path_length,int_rate,volatility):
        "Calculates the value of an option using a simple Monte Carlo method."
        erdt = math.exp(-int_rate*self.expiry)
        sim_values = self._mc_simulation(no_paths,path_length,int_rate,volatility)
        v_mc = erdt * np.sum(sim_values)/no_paths
        return v_mc


class EuropeanOption(LatticeValuedOption,MCValuedOption):
    """
    Vanilla options which have a payoff dependent on the difference between
    the asset value and the strike on expiry. Easy to value via a variety
    of methods, and have an exact closed-form solution, which is callable
    using the 'black_scholes' method.
    """

    def __init__(self,name,asset,*,strike,option_type,expiry,style):
        if option_type not in ['call','put']:
            raise ValueError('Invalid option type (must specify call/put).')
        else:
            self.name = name
            self.asset = asset
            self.strike = strike
            self.option_type = option_type
            self.expiry = expiry
            self.style = style

    def _bin_value(self,time_steps,v,p,s,erdt):
        for i in range(time_steps-1,-1,-1):
            v[i] = [erdt*(p*v[i+1][j+1] + (1-p)*v[i+1][j]) for j in range(0,i+1)]
        return v[0][0]

    def _tri_value(self,time_steps,v,pu,pm,pd,s,erdt):
        for i in range(time_steps-1,-1,-1):
            v[i] = [erdt*(pu*v[i+1][j+2] + pm*v[i+1][j+1] + pd*v[i+1][j]) for j in range(2*i+1)]
        return v[0][0]

    def _mc_simulation(self,no_paths,path_length,int_rate,volatility):
        s = self.asset._mc_sim_path(no_paths,path_length,self.expiry,int_rate,volatility)
        v = np.zeros(no_paths)
        for i in range(no_paths):
            v[i] = self.payoff(s[i,-1],self.strike,self.option_type)
        return v

    def black_scholes(self,int_rate,volatility):
        s = self.asset.asset_value
        strike = self.strike
        expiry = self.expiry
        div = self.asset.div_yield
        #
        d1 = (np.log(s/strike) + (int_rate - div + volatility**2 / 2)*expiry)/(volatility*np.sqrt(expiry))
        d2 = d1 - volatility*np.sqrt(expiry)
        if self.option_type == 'call':
            v = s*math.exp(-div*expiry)*sp.stats.norm.cdf(d1,0,1) - strike*math.exp(-int_rate*expiry)*sp.stats.norm.cdf(d2,0,1)
        elif self.option_type == 'put':
            v = strike*math.exp(-int_rate*expiry)*sp.stats.norm.cdf(-d2,0,1) - s*math.exp(-div*expiry)*sp.stats.norm.cdf(-d1,0,1)
        else:
            ValueError('Invalid option type (must specify call/put).')
        return v


class AmericanOption(LatticeValuedOption):
    """
    Class of American options, which are similar to European options, but
    permit early exercise. Early exercise boundaries make MC-based methods
    prohibitive for this option.
    """

    def __init__(self,name,asset,*,strike,option_type,expiry,style):
        if option_type not in ['call','put']:
            raise ValueError('Invalid option type (must specify call/put).')
        else:
            self.name = name
            self.asset = asset
            self.strike = strike
            self.option_type = option_type
            self.expiry = expiry
            self.style = style

    def _bin_value(self,time_steps,v,p,s,erdt):
        for i in range(time_steps-1,-1,-1):
            v[i] = [max(erdt*(p*v[i+1][j+1] + (1-p)*v[i+1][j]),self.payoff(s[i][j],self.strike,self.option_type)) for j in range(0,i+1)]
        return v[0][0]

    def _tri_value(self,time_steps,v,pu,pm,pd,s,erdt):
        for i in range(time_steps-1,-1,-1):
            v[i] = [max(erdt*(pu*v[i+1][j+2] + pm*v[i+1][j+1] + pd*v[i+1][j]),self.payoff(s[i][j],self.strike,self.option_type)) for j in range(2*i+1)]
        return v[0][0]


class BinaryOption(LatticeValuedOption,MCValuedOption):
    """
    Class of Binary options, which offer either cash/asset or nothing on expiry.
    Risky exotic option, but simple to value. Has an exact closed-form
    solution which is callable using the 'black_scholes' method.
    """

    def __init__(self,name,asset,*,strike,option_type,expiry,style,cash=None):
        self.name = name
        self.asset = asset
        self.strike = strike
        self.option_type = option_type
        self.expiry = expiry
        self.style = style
        self.cash = cash
        if option_type not in ['cash-or-nothing call', 'cash-or-nothing put','asset-or-nothing call', 'asset-or-nothing put']:
            raise ValueError('Incompatible option style/type.')

    def _bin_value(self,time_steps,v,p,s,erdt):
        for i in range(time_steps-1,-1,-1):
            v[i] = [erdt*(p*v[i+1][j+1] + (1-p)*v[i+1][j]) for j in range(0,i+1)]
        return v[0][0]

    def _tri_value(self,time_steps,v,pu,pm,pd,s,erdt):
        for i in range(time_steps-1,-1,-1):
            v[i] = [erdt*(pu*v[i+1][j+2] + pm*v[i+1][j+1] + pd*v[i+1][j]) for j in range(2*i+1)]
        return v[0][0]

    def _mc_simulation(self,no_paths,path_length,int_rate,volatility):
        s = self.asset._mc_sim_path(no_paths,path_length,self.expiry,int_rate,volatility)
        v = np.zeros(no_paths)
        for i in range(no_paths):
            v[i] = self.payoff(s[i,-1],self.strike,self.option_type)
        return v

    def black_scholes(self,int_rate,volatility):
        s = self.asset.asset_value
        div = self.asset.div_yield
        strike = self.strike
        expiry = self.expiry
        #
        d1 = (np.log(s/strike) + (int_rate - div + volatility**2 / 2)*expiry)/(volatility*np.sqrt(expiry))
        d2 = d1 - volatility*np.sqrt(expiry)
        if self.option_type == 'cash-or-nothing call':
            v = self.cash*math.exp(-int_rate*expiry)*sp.stats.norm.cdf(d2,0,1)
        elif self.option_type == 'cash-or-nothing put':
            v = self.cash*math.exp(-int_rate*expiry)*sp.stats.norm.cdf(-d2,0,1)
        elif self.option_type == 'asset-or-nothing call':
            v = s*math.exp(-div*expiry)*sp.stats.norm.cdf(d1,0,1)
        else:
            v = s*math.exp(-div*expiry)*sp.stats.norm.cdf(-d1,0,1)
        return v


class EuropeanBarrierOption(LatticeValuedOption,MCValuedOption):
    """
    Class of Barrier options which permit early exercise.
    The 'Out' barriers are easily calculated, whereas 'In' barriers
    are calculated using In-Out Parity results, and therefore take longer to
    calculate.
    """

    def __init__(self,name,asset,*,strike,option_type,expiry,style,barrier_type=None,barrier=None):
        self.name = name
        self.asset = asset
        self.strike = strike
        self.option_type = option_type
        self.expiry = expiry
        self.style = style
        self.barrier_type = barrier_type
        self.barrier = barrier
        #
        if barrier_type is not None:
            if barrier_type in ['up and in','up and out'] and barrier < asset.asset_value:
                raise ValueError("Can't have 'up' barrier when the asset is already above the barrier threshold.")
            elif barrier_type in ['down and in','down and out'] and barrier > asset.asset_value:
                raise ValueError("Can't have 'down' barrier when the asset is already below the barrier threshold.")
            elif barrier_type not in ['up and in','up and out','down and in','down and out']:
                raise ValueError("Invalid barrier type (must specify e.g. down and out).")

    def _bin_value(self,time_steps,v,p,s,erdt):
        b = list()
        vb = list()
        b.append([1])
        vb.append([0])
        if self.barrier_type in ['up and in','up and out']:
            for i in range(1,time_steps+1):
                b.append([1*(s[i][j] <= self.barrier) for j in range(i+1)])
                vb.append([0 for j in range(i+1)])
        else:
            for i in range(1,time_steps+1):
                b.append([1*(s[i][j] >= self.barrier) for j in range(i+1)])
                vb.append([0 for j in range(i+1)])
        vb[-1] = [self.payoff(x,self.strike,self.option_type) for x in s[-1]]
        #
        if self.barrier_type in ['up and out','down and out']:
            for i in range(time_steps-1,-1,-1):
                vb[i] = [erdt*(p*b[i+1][j+1]*vb[i+1][j+1] + (1-p)*b[i+1][j]*vb[i+1][j]) for j in range(i+1)]
            return vb[0][0]
        else:  # use in-out parity for 'in' options
            for i in range(time_steps-1,-1,-1):
                v[i] = [erdt*(p*v[i+1][j+1] + (1-p)*v[i+1][j]) for j in range(i+1)]
                vb[i] = [erdt*(p*b[i+1][j+1]*vb[i+1][j+1] + (1-p)*b[i+1][j]*vb[i+1][j]) for j in range(i+1)]
            v0 = v[0][0] - vb[0][0]
            return v0

    def _tri_value(self,time_steps,v,pu,pm,pd,s,erdt):
        b = list()
        vb = list()
        b.append([1])
        vb.append([0])
        if self.barrier_type in ['up and in','up and out']:
            for i in range(1,time_steps+1):
                b.append([1*(s[i][j] <= self.barrier) for j in range(2*i+1)])
                vb.append([0 for j in range(2*i+1)])
        else:
            for i in range(1,time_steps+1):
                b.append([1*(s[i][j] >= self.barrier) for j in range(2*i+1)])
                vb.append([0 for j in range(i+1)] + [0 for j in range(2*i+1)])
        vb[-1] = [self.payoff(x,self.strike,self.option_type) for x in s[-1]]
        #
        if self.barrier_type in ['up and out','down and out']:
            for i in range(time_steps-1,-1,-1):
                vb[i] = [erdt*(pu*b[i+1][j+2]*vb[i+1][j+2] + pm*b[i+1][j+1]*vb[i+1][j+1] + pd*b[i+1][j]*vb[i+1][j]) for j in range(2*i+1)]
            return vb[0][0]
        else:  # use in-out parity for 'in' options
            for i in range(time_steps-1,-1,-1):
                v[i] = [erdt*(pu*v[i+1][j+2] + pm*v[i+1][j+1] + pd*v[i+1][j]) for j in range(2*i+1)]
                vb[i] = [erdt*(pu*b[i+1][j+2]*vb[i+1][j+2] + pm*b[i+1][j+1]*vb[i+1][j+1] + pd*b[i+1][j]*vb[i+1][j]) for j in range(2*i+1)]
            v0 = v[0][0] - vb[0][0]
            return v0

    def _mc_simulation(self,no_paths,path_length,int_rate,volatility):
        s = self.asset._mc_sim_path(no_paths,path_length,self.expiry,int_rate,volatility)
        # Take averages for each sample path
        if self.barrier_type == 'down_and_out':
            i = np.min(s,1) < self.barrier
        if self.barrier_type == 'down and in':
            i = np.min(s,1) >= self.barrier
        if self.barrier_type == 'up and out':
            i = np.max(s,1) > self.barrier
        if self.barrier_type == 'up and in':
            i = np.max(s,1) <= self.barrier
        # Calculate payoffs
        v = np.zeros(no_paths)
        for i in range(no_paths):
            v[i] = self.payoff(s[i,-1],self.strike,self.option_type)
        return v


class AmericanBarrierOption(LatticeValuedOption):
    """
    Class of Barrier options which permit early exercise.
    NOTE: As with the European Barrier Options, 'Out' barriers are easily calculated,
    whereas 'In' barriers require In-Out Parity to be calculated.
    This does exist for options with early exercise, but is more complicated
    than the naive calculation presented here, which only provides a lower bound on the value.
    """

    def _bin_value(self,time_steps,v,p,s,erdt):
        b = list()
        vb = list()
        b.append([1])
        vb.append([0])
        if self.barrier_type in ['up and in','up and out']:
            for i in range(1,time_steps+1):
                b.append([1*(s[i][j] <= self.barrier) for j in range(i+1)])
                vb.append([0 for j in range(i+1)])
        else:
            for i in range(1,time_steps+1):
                b.append([1*(s[i][j] >= self.barrier) for j in range(i+1)])
                vb.append([0 for j in range(i+1)])
        vb[-1] = [self.payoff(x,self.strike,self.option_type) for x in s[-1]]
        #
        if self.barrier_type in ['up and out','down and out']:
            for i in range(time_steps-1,-1,-1):
                vb[i] = [max(erdt*(p*b[i+1][j+1]*vb[i+1][j+1] + (1-p)*b[i+1][j]*vb[i+1][j]),
                             self.payoff(s[i][j],self.strike,self.option_type)) for j in range(i+1)]
            return vb[0][0]
        else:  # use in-out parity for 'in' options
            print("\033[1;31;1mWARNING: \033[0;0;1mNot properly implemented - see help(AmericanBarrierOption).\n")
            for i in range(time_steps-1,-1,-1):
                v[i] = [max(erdt*(p*v[i+1][j+1] + (1-p)*v[i+1][j]),self.payoff(s[i][j],self.strike,self.option_type)) for j in range(i+1)]
                vb[i] = [max(erdt*(p*b[i+1][j+1]*vb[i+1][j+1] + (1-p)*b[i+1][j]*vb[i+1][j]),self.payoff(s[i][j],self.strike,self.option_type)) for j in range(i+1)]
            v0 = v[0][0] - vb[0][0]
            return v0

    def _tri_value(self,time_steps,v,pu,pm,pd,s,erdt):
        b = list()
        vb = list()
        b.append([1])
        vb.append([0])
        if self.barrier_type in ['up and in','up and out']:
            for i in range(1,time_steps+1):
                b.append([1*(s[i][j] <= self.barrier) for j in range(2*i+1)])
                vb.append([0 for j in range(2*i+1)])
        else:
            for i in range(1,time_steps+1):
                b.append([1*(s[i][j] >= self.barrier) for j in range(2*i+1)])
                vb.append([0 for j in range(i+1)] + [0 for j in range(2*i+1)])
        vb[-1] = [self.payoff(x,self.strike,self.option_type) for x in s[-1]]
        #
        if self.barrier_type in ['up and out','down and out']:
            for i in range(time_steps-1,-1,-1):
                vb[i] = [max(erdt*(pu*b[i+1][j+2]*vb[i+1][j+2] + pm*b[i+1][j+1]*vb[i+1][j+1] + pd*b[i+1][j]*vb[i+1][j]),self.payoff(s[i][j],self.strike,self.option_type)) for j in range(2*i+1)]
            return vb
        else:  # use in-out parity for 'in' options
            print("\033[1;31;1mWARNING: \033[0;0;1mNot properly implemented - see help(AmericanBarrierOption).\n")
            for i in range(time_steps-1,-1,-1):
                v[i] = [max(erdt*(pu*v[i+1][j+2] + pm*v[i+1][j+1] + pd*v[i+1][j]),self.payoff(s[i][j],self.strike,self.option_type)) for j in range(2*i+1)]
                vb[i] = [max(erdt*(pu*b[i+1][j+2]*v[i+1][j+2] + pm*b[i+1][j+1]*v[i+1][j+1] + pd*b[i+1][j]*v[i+1][j]),self.payoff(s[i][j],self.strike,self.option_type)) for j in range(2*i+1)]
            v0 = v[0][0] - vb[0][0]
            return v0


class AsianEuropeanOption(MCValuedOption):
    """
    Class of options which yield payoffs which are dependent on averages.
    The implementation here allows for average rate/strike puts/calls using
    (ar)i(th)metic and (geom)etric averaging.
    """

    def __init__(self,name,asset,*,strike,option_type,expiry,style,avg_mthd):
        self.name = name
        self.asset = asset
        self.strike = strike
        self.option_type = option_type
        self.expiry = expiry
        self.style = style
        self.avg_mthd = avg_mthd
        if option_type not in ['average rate call','average strike call','average rate put','average strike put']:
            raise ValueError('Incompatible option style/type.')
        if avg_mthd not in ['arth', 'geom']:
            raise ValueError('Invalid averaging method (arth/geom).')

    def _mc_simulation(self,no_paths,path_length,int_rate,volatility):
        s = self.asset._mc_sim_path(no_paths,path_length,self.expiry,int_rate,volatility)
        # Take averages for each sample path
        if self.avg_mthd == 'arth':
            avg_s = sum(s.transpose()[:])/(path_length+1)
        else:  # geom
            avg_s = math.prod(s.transpose()[:])**(1/(path_length+1))
        # Calculate payoffs
        v = np.zeros(no_paths)
        for i in range(no_paths):
            v[i] = self.payoff(s[i,-1],self.strike,self.option_type,asset_avg=avg_s[i])

        return v
