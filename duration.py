"""
Duration hedging calculator.
Uses bonds to hedge against interest rate risk.
"""

import math
import numpy as np
from intrates import VasicekRate


class Bond():
    "Defines a class of bonds."

    def __init__(self,name,maturity):
        self.name = name
        self.maturity = maturity

    def _hedge_dur_std(self,bond1,bond2,int_rate):
        "The conventional duration hedge."

        t = self.maturity
        t1 = bond1.maturity
        t2 = bond2.maturity

        v = math.exp(-int_rate*t)
        v1 = math.exp(-int_rate*t1)
        v2 = math.exp(-int_rate*t2)

        h = np.array([[1,1],[t1,t2]])
        c = np.array([[1],[t]])
        [[w1],[w2]] = np.linalg.solve(h,c)
        [q1,q2] = [w1*v/v1,w2*v/v2]
        [b1,b2] = [q1*v1,q2*v2]
        miss = v - b1 - b2

        return [q1,q2,b1,b2,miss]

    def _acid_dur_std(self,bond1,bond2,q1,q2,int_rate,magnitude):
        """
        Tests a hedge against a change 'magnitude' in the interest rate,
        for the standard duration hedge.
        """

        new_rate = int_rate + magnitude

        t = self.maturity
        t1 = bond1.maturity
        t2 = bond2.maturity

        v = math.exp(-new_rate*t)
        v1 = math.exp(-new_rate*t1)
        v2 = math.exp(-new_rate*t2)

        hedge_err = v - q1*v1 - q2*v2

        return hedge_err

    def _hedge_dur_vasicek(self,bond1,bond2,int_rate):
        "The Vasicek duration hedge."

        t = self.maturity
        t1 = bond1.maturity
        t2 = bond2.maturity

        r = int_rate._spot_rate(t)
        r1 = int_rate._spot_rate(t1)
        r2 = int_rate._spot_rate(t2)

        v = math.exp(-r*t)
        v1 = math.exp(-r1*t1)
        v2 = math.exp(-r2*t2)

        d = int_rate._duration(t)
        d1 = int_rate._duration(t1)
        d2 = int_rate._duration(t2)

        h = np.array([[1,1],[d1,d2]])
        c = np.array([[1],[d]])

        [[w1],[w2]] = np.linalg.solve(h,c)
        [q1,q2] = [w1*v/v1,w2*v/v2]
        [b1,b2] = [q1*v1,q2*v2]
        miss = v - b1 - b2

        return [q1,q2,b1,b2,miss]

    def _acid_dur_vasicek(self,bond1,bond2,q1,q2,vas_rate,magnitude):
        """
        Tests a hedge against a change 'magnitude' in the interest rate,
        for the Vasicek duration hedge.
        """

        int_rate = VasicekRate('Vasicek',vas_rate.r0 + magnitude,0.1,0.2,0.05)

        t = self.maturity
        t1 = bond1.maturity
        t2 = bond2.maturity

        r = int_rate._spot_rate(t)
        r1 = int_rate._spot_rate(t1)
        r2 = int_rate._spot_rate(t2)

        v = math.exp(-r*t)
        v1 = math.exp(-r1*t1)
        v2 = math.exp(-r2*t2)

        hedge_err = v - q1*v1 - q2*v2

        return hedge_err

    def _hedge_conv_std(self,bond1,bond2,bond3,int_rate):
        "An interest rate hedge using convexity. Requires 3 bonds."

        t = self.maturity
        t1 = bond1.maturity
        t2 = bond2.maturity
        t3 = bond3.maturity

        v = math.exp(-int_rate*t)
        v1 = math.exp(-int_rate*t1)
        v2 = math.exp(-int_rate*t2)
        v3 = math.exp(-int_rate*t2)

        h = np.array([[1,1,1],[t1,t2,t3],[t1**2,t2**2,t3**2]])
        c = np.array([[1],[t],[t**2]])
        [[w1],[w2],[w3]] = np.linalg.solve(h,c)
        [q1,q2,q3] = [w1*v/v1,w2*v/v2,w3*v/v3]
        [b1,b2,b3] = [q1*v1,q2*v2,q3*v3]
        miss = v - b1 - b2 - b3

        return [q1,q2,q3,b1,b2,b3,miss]

    def _acid_conv_std(self,bond1,bond2,bond3,q1,q2,q3,int_rate,magnitude):
        """
        Tests a hedge against a change 'magnitude' in the interest rate,
        for the standard convexity hedge.
        """

        new_rate = int_rate + magnitude

        t = self.maturity
        t1 = bond1.maturity
        t2 = bond2.maturity
        t3 = bond3.maturity

        v = math.exp(-new_rate*t)
        v1 = math.exp(-new_rate*t1)
        v2 = math.exp(-new_rate*t2)
        v3 = math.exp(-new_rate*t3)

        hedge_err = v - q1*v1 - q2*v2 - q3*v3

        return hedge_err
