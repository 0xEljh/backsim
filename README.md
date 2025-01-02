# backsim

(Truly) minimal and flexible backtesting for Python

This is a lightweight, event-driven backtesting framework written in pure Python. It focuses on **core backtesting logic**—time iteration, order execution, and portfolio accounting—while giving users the freedom to handle their own data ingestion, indicator computation, and plotting.

## Why backsim?

Backsim is inspired by [pmorissette's bt](https://github.com/pmorissette/bt) and takes its flexibility one step further. In exchange for rich features, many backtesters are too opinionated about data formating, plotting, and even strategy structure. Backsim is designed for advanced users who want to build upon a solid foundation without being constrained by the framework.

Backsim hence aims to:

- **Stay minimal**: Provide backtesting essentials with a modular design. It will attempt to thread the needle between offering sufficient out of the box functionality and yet keeping features optional/modular and extensible.
- **Stay flexible**: Use simple data interfaces so you can bring your own data, transformations, and charting tools.
- **Offer clarity**: Keep the code readable and well-documented, making it easy to extend or optimize.
- **Be strategy-first**: Allow the user to "drop in" their strategy logic without having to make data transforms to fit the framework.

### Features

- **Event-Driven or Vector-Friendly**: Drive the simulation via an event loop but allow vectorized pre-computation for signals.
- **Modular Portfolio/Broker**: Pluggable models for fills, slippage, and commissions.
- **Multiple Frequencies**: Supports daily bars, intraday data, and event-based (e.g. fundamental) triggers.
- **Extensible Data Ingestion**: Minimal *DataSource* classes (CSV, pandas DataFrame, etc.) let you feed data your way.
- **Optional Plots**: We'd prefer to hand you rich results data and let you plot it your way, but we'll offer some sanity checks and out-of-the-box plots.

## Installation

backsim is on PyPI (planned), so you can install it with pip:

```bash
pip install backsim
```

## License

Backsim is released under the [MIT License](/LICENSE).
