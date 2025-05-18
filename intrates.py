"""
Describes some simple interest rate models. Allows for the plotting of sample paths.
"""

import math
import numpy as np
import matplotlib.pyplot as plot


class InterestRate():
    """
    Defines the class of interest rates.
    The initial rate, r0, must always be defined.
    Other variables defined depend on the specific model.
    """

    def __init__(self,name,r0):
        self.name = name
        self.r0 = r0

    def _rate_sim_path(self,path_length,expiry):
        "Simulates an interest rate path until expiry using a specified path length."
        r0 = self.r0
        volatility = self.volatility

        dt = expiry/path_length
        rtdt = np.sqrt(dt)
        w = np.zeros(path_length+1)
        if volatility is not None:
            # Create Wiener Process
            rndm_pts = np.random.normal(size=path_length,scale=rtdt)
            for i in range(path_length):
                w[i+1] = w[i] + rndm_pts[i]
        # Evolve interest rate
        int_rate = self._evolve_rate(path_length,w,dt,r0)
        return int_rate

    def _plot_rate_path(self,rate_path):
        "Plots a simulated rate path."
        plot.figure()
        x = np.array(range(len(rate_path)))
        y = rate_path
        plot.plot(x,y)
        plot.show()


class ConstantRate(InterestRate):
    "Defines a constant interest rate, r = r0."

    def __init__(self,name,r0):
        self.name = name
        self.r0 = r0

    def _evolve_rate(self,path_length,w,dt):
        r = np.zeros(path_length+1)
        mu = self.r0
        r = r + mu
        return r


class VasicekRate(InterestRate):
    """
    Defines a Vasicek short-rate model process:
        dr(t) = alph(mu - r(t))dt + sig*dz(t).
    """

    def __init__(self,name,r0,mean,reversion_speed,volatility):
        self.name = name
        self.r0 = r0
        self.mean = mean
        self.reversion_speed = reversion_speed
        self.volatility = volatility

    def _evolve_rate(self,path_length,w,dt):
        r = np.zeros(path_length+1)
        mu = self.mean
        a = self.reversion_speed
        sig = self.volatility
        r[0] = self.r0
        for i in range(path_length):
            r[i+1] = r[i] + a*(mu - r[i])*dt + sig*(w[i+1]-w[i])
        return r

    def _spot_rate(self,t):
        """
        Defines the spot rate for the Vasicek process.
        Theoretically derived using well-known expectation, variance and bond pricing.
        """
        r0 = self.r0
        mu = self.mean
        alph = self.reversion_speed
        sig = self.volatility

        erat = (1 - math.exp(-alph*t)) / (alph*t)
        r_fin = mu - (sig/alph)**2 / 2

        r = r_fin + (r0 - r_fin)*erat + ((t*sig**2)/(4*alph))*erat**2
        return r

    def _duration(self,t):
        """
        Defines the duration of a given Vasicek process.
        """
        alph = self.reversion_speed
        d = (1 - math.exp(-alph*t))/alph
        return d


class VasicekRiskAdjusted(VasicekRate):
    """
    Defines a risk-adjusted Vasicek process:
        dr(t) = alph(mu - r(t) - l*sig)dt + sig*dz(t).
    Here, l represents the price of risk, defining how much additional return is required
    per-unit of standard deviation carried.
    Equivalent to the process:
        dr(t) = alpha(mu' - r(t))dt + *dz(t), mu' = mu - l*sig/alph.
    """

    def __init__(self,name,r0,mean,reversion_speed,volatility,risk_factor):
        self.name = name
        self.r0 = r0
        self.mean = mean - risk_factor*volatility/reversion_speed
        self.reversion_speed = reversion_speed
        self.volatility = volatility


class CIRRate(InterestRate):
    """
    Defines a CIR short-rate model process:
        dr(t) = alph(mu - r(t))dt + sig*(rt)^0.5*dz(t).
    """

    def __init__(self,name,r0,mean,reversion_speed,volatility):
        if 2*reversion_speed*mean <= volatility:
            raise ValueError('Feller condition not satisfied; lower volatility.')
        else:
            self.name = name
            self.r0 = r0
            self.mean = mean
            self.reversion_speed = reversion_speed
            self.volatility = volatility

    def _evolve_rate(self,path_length,w,dt):
        r = np.zeros(path_length+1)
        mu = self.mean
        a = self.reversion_speed
        sig = self.volatility
        r[0] = self.r0
        try:
            for i in range(path_length):
                r[i+1] = r[i] + a*(mu - r[i])*dt + math.sqrt(r[i])*sig*(w[i+1]-w[i])
            return r
        except ValueError:
            for i in range(path_length):
                if r[i] >= 0:
                    r[i+1] = r[i] + a*(mu - r[i])*dt + math.sqrt(r[i])*sig*(w[i+1]-w[i])
                else:
                    r[i+1] = 0
            return r
