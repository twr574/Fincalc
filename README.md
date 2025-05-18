
# Fincalc

A Python-based collection of finance-related modules.

Download the repo and open demo<area>.py for a demonstration of some of the features included.

## Features

- Exchange rate information utility `fxrates`, which can be used to crawl exchange rate data to calculate the cost of transactions between currencies.

- Simple FX option trade calculator `fxcorrelation`, which, given the required input data, determines whether to take on a correlation position in a market (and determines the appropriately hedged portfolio).

- Interest rate hedging calculator, `duration`, which uses bonds and duration-based calculations to hedge against interest rate risk.

- Option valuation calculator `options`, which values a number of common option styles using Monte Carlo and Lattice methods, including European, American, and some exotics.
  
## Requirements

- Modules which are listed in in Python's standard library: <https://docs.python.org/3/library/index.html>.

- Beautiful Soup: <https://pypi.org/project/beautifulsoup4/>.

- Matplotlib: <https://pypi.org/project/matplotlib/>.

- NumPy: <https://pypi.org/project/numpy/>.

- Pandas: <https://pypi.org/project/pandas/>.

- SciPy: <https://pypi.org/project/scipy/>.
  
- Tabulate: <https://pypi.org/project/tabulate/>.

## Future Work

This project was originally written as an independent coding exercise in order to work on my knowledge of financial mathematics, object-oriented programming and simple tasks such as saving/importing CSVs and scraping data from webpages.

If time permits, some additions may eventually be made to this work. Some ideas include:

- Use of `intrates` for modelling calculations related to savings/ISAs. This should be fairly simple to do, and gives this module more of a practical application than it currently has.

- Allowing for non-constant interest rate models in option valuations, providing an explicit link between the `options` and `intrates` modules.

- The valuation of American-style barrier options. The European case is already dealt with, and the 'out' barriers for American-style options are already computed correctly. In-out barrier parity does exist for American options, but the early exercise possibility makes the maths a little more complicated.

- Optimisations for the option-based calculations, in general. Possible improvements include control variates and low-discrepancy sequences for Monte Carlo (the latter leading to Quasi-Monte Carlo), as well as pruning for lattice-based methods, for example.

- More general interest rate hedging calculators in `duration`. Where there is no explicit formula for duration, this can be constructed numerically using difference methods, if appropriate Bond price and interest rate data can be derived.

- An explicit link between `fxrates` and `fxcorrelation`. Using actual market data is an obvious next step for calculations performed in the latter.
