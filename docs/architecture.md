# backsim architecture & design choices

## Overview

backsim is built around an **event-driven** backtesting loop, with a minimal set of Pythonic abstractions. The goal is to:

- Ensure **clarity** (simple pythonic classes, minimal code)
- Maintain **flexibility** (extensible data ingestion, custom broker/portfolio)
- Keep **performance** in mind without sacrificing simplicity (allow vectorized pre-computation)

## Core Components

### Event Engine (Clock)

- Advances simulation time by a predefined tick/step-size.
- Compiles or coordinates the time-aligned data from one or more DataSources into an AssetUniverse.
- Feeds data, in chronological order, to each Strategy and coordinates chunking if necessary.

### DataSource

- Responsible for providing time-ordered data (candles, ticks, or custom events).
- An abstract class that can be subclassed for CSV, SQL, Pandas, etc.
- Allows batched data retrieval/loading for ML or vectorized operations.
- Must implement some form of query or slice interface (e.g. get_data_for_time(t), or get_data_range(start_t, end_t)).

### Data/Asset Universe

- A compiled, time-aligned dataset that merges/joins data from multiple DataSources.
- Potentially stored in Pandas, Polars, NumPy arrays, or a custom in-memory format.
- Exposes a consistent API (e.g., get_data_slice(t, requirements)) that returns data in a standardized shape.
- Manages chunking or memory constraints if data is very large.

### Portfolio

Central record keeper. It tracks:

- Current positions: (symbol -> quantity, cost basis, unrealized PnL).
- Cash balance or buying power.
- A list of open orders that are not yet filled.

It receives fill events from the Broker.

### Strategy

Generates orders based on market data + Portfolio state.

Inputs:

- Data slice of AssetUniverse at a given time.
- Portfolio state (current positions, open orders, cash balance).

Outputs:

- A list of orders

Strategy can be both stateful or stateless. It will also come with some in-built sanity checks and time-stepping logic.

Defines a data_slicer or similar method indicating how it requests data from the AssetUniverse (e.g., last NN bars, specific columns, etc.).

### Broker

Decides how orders are filled in the simulation

- Iterates over Portfolio's open orders
- Simulates fills based on market data (has reference to AssetUniverse) so it can model slippage, spread, etc.

Calls back into Portfolio.fill_order(...) as fills occur.

### Logger

Creates, stores, and aggregates logs for the simulation

- Logs can be used for debugging, performance analysis, or visualization.

## Pseudocode

Here is how engine works in pseudocode:

```python

for t in timeline:
    orders = []

    # 1) create new orders from strategies
    for strategy in strategies:
      # get data slice for strategy
      data_slice = asset_universe.get_data_slice(strategy.data_slicer, t)
      orders.extend(strategy.generate_orders(data_slice, portfolio, t))

    # 2) Portfolio receives new orders; these are not seen by the broker till the next time step
    portfolio.add_orders(orders)

    # 3) Broker attempts fills on existing orders; broker may be configured so fills cannot happen on the same time step
    broker.process_fills(portfolio, t)

    # 4) Log or snapshot
    logger.store_snapshot(t, portfolio, broker, strategy)
```

Here is how backsim is used, in pseudocode:

```python
from backsim import BackSim, CSVDataSource, Broker, Strategy, Portfolio

# 1) Initialize data source
data_source = CSVDataSource("path/to/data.csv")

# 2) Build AssetUniverse from the data source
#    (e.g. load data into memory or keep references for chunked loading)
asset_universe = AssetUniverse([data_source])

# 3) Create Broker
broker = Broker(
  fill_time="next_tick",     # or "immediate"
  slippage_model=BackSim.FixedSlippage(0.01)
)

# 4) Create Portfolio
portfolio = Portfolio(cash=100000.0)

# 5) Define a custom Strategy
class MyStrategy(Strategy):
    def generate_orders(self, data_slice, portfolio, t):
        # your strategy logic here
        # e.g. if data_slice['AAPL'].close > data_slice['AAPL'].open: buy AAPL
        return orders
    
    def data_slicer(self, t):
        # define the data shape you need
        # e.g. last 20 bars for symbols: AAPL, MSFT
        # something like:
        return {
            "symbols": ["AAPL", "MSFT"],
            "fields": ["open", "high", "low", "close", "volume"],
            "lookback": 20
        }

# 6) Initialize backsim (Event Engine) & run
engine = BackSim(
    tick_size="1D",
    asset_universe=asset_universe,
    portfolio=portfolio,
    broker=broker,
    strategies=[MyStrategy()]
)
engine.run(start_date='2020-01-01', end_date='2020-12-31')


