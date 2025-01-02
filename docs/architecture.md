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

### AssetUniverse

Stores and manages only price/volume (OHLCV or ticks) in a standardized frequency.
Typically constructed by aggregating and aligning price data from one or more raw data sources (e.g. CSV, SQL).
Provides a consistent API for retrieving slices of price data at the framework’s main time-step.
For example:

```python
asset_universe.get_price_slice(t, symbols=["AAPL", "BTC"], fields=["open", "close"])
```

- Manages chunking or memory constraints if data is very large.

### DataSource

- Encapsulates supporting data (e.g., fundamentals, macro data, alternative data) that may be at various time frequencies.
- An abstract class that can be subclassed for CSV, SQL, Pandas, etc.
- Allows batched data retrieval/loading for ML or vectorized operations.
- Must implement some form of query or slice interface (e.g. get_data_for_time(t) or get_data_range(start, end)).

Note: The user can combine or “slice” DataSource outputs and AssetUniverse price data within their data_slicer function (see Strategy’s data-slicing mechanism below).

### Portfolio

Central record keeper. It tracks:

- Current positions: (symbol -> quantity, cost basis, unrealized PnL).
- Cash balance or buying power.
- A list of open orders that are not yet filled.

It receives fill events from the Broker.

### Strategy

Generates orders based on market data + Portfolio state.

Inputs:

- Price/volume data from the AssetUniverse.
- Supporting data from relevant DataSources (potentially at different frequencies).
- Portfolio state (current positions, open orders, cash balance).

Outputs:

- A list of orders

Strategy can be both stateful or stateless. It will also come with some in-built sanity checks and time-stepping logic.

The Strategy needs to define a `data_slicer` method that determines how and what data is pulled at each time step:

- For example, how many bars to look back, which symbols to include, which fields are needed, or how to align a lower-frequency DataSource (e.g., daily fundamentals) with higher-frequency price data (e.g., 5-min bars).

```python
def data_slicer(self, t):
    # Price data: last 20 bars, 4-hour frequency
    price_slice_req = {
        "symbols": ["AAPL", "MSFT"],
        "fields": ["open", "high", "low", "close", "volume"],
        "lookback": 20,
        "frequency": "4h"
    }
    
    # Supporting data: daily fundamentals from a separate DataSource
    # alternatively, instead of a dict, you can return a function that fetches the data
    fundamentals_req = {
        "data_source_name": "fundamentals_csv",
        "frequency": "1d",
        "lookback": 1,
        "align_method": "forward_fill",  # optional
        "align_data": True  # optional
    }
    
    return (price_slice_req, fundamentals_req)
```

At each simulation step:

- The framework will first retrieve the price data slice from AssetUniverse.
- The user (or framework) retrieves any supporting data from the specified DataSource(s) via custom logic that handles frequency alignment (e.g. forward fill, last known value).
- The Strategy logic then combines both data sets to produce trading signals and orders.

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

    # 1) generate new orders from strategies
    for strategy in strategies:
        # 1a) retrieve price data slice from AssetUniverse at time t
        price_slice = asset_universe.get_data_slice(strategy.data_slicer(t).price_slice_req, t)
        
        # 1b) retrieve supporting data from other DataSources
        aux_slice = load_supporting_data(strategy.data_slicer(t).fundamentals_req, t)
        
        # 1c) combine both slices to produce orders
        orders.extend(strategy.generate_orders(
            price_slice, 
            aux_slice, 
            portfolio, 
            t
        ))

    # 2) Portfolio records the new orders (they are not seen by the Broker until the next step, if so configured)
    portfolio.add_orders(orders)

    # 3) Broker attempts fills on existing orders at time t
    broker.process_fills(portfolio, t)

    # 4) Log or snapshot state
    logger.store_snapshot(t, portfolio, broker, strategies)

```

Here is how backsim is used, in pseudocode:

```python
from backsim import BackSim, CSVDataSource, Broker, Strategy, Portfolio, AssetUniverse

# 1) Initialize data sources
price_data_source = CSVDataSource("path/to/price_data.csv")
fundamentals_data_source = CSVDataSource("path/to/fundamentals.csv")

# 2) Build the AssetUniverse for standardized frequency price data
#    (internally, you might specify daily, hourly, etc.)
asset_universe = AssetUniverse([price_data_source], frequency="1h")

# 3) Create Broker with chosen fill policy/slippage
broker = Broker(
    fill_time="next_tick", 
    slippage_model=BackSim.FixedSlippage(0.01)
)

# 4) Create Portfolio
portfolio = Portfolio(cash=100_000.0)

# 5) Define a custom Strategy
class MyStrategy(Strategy):
    def generate_orders(self, price_slice, fundamentals_slice, portfolio, t):
        # your strategy logic here
        # e.g. if 'AAPL' close price > open price: buy AAPL
        orders = []
        aapl_close = price_slice["AAPL"]["close"][-1]
        aapl_open = price_slice["AAPL"]["open"][-1]
        
        if aapl_close > aapl_open:
            orders.append({"symbol": "AAPL", "quantity": 10, "side": "BUY"})
        
        # Possibly use fundamentals_slice for additional signals
        # ...
        return orders

    def data_slicer(self, t):
        # define how to slice price data and any supporting data
        return {
            "price_slice_req": {
                "symbols": ["AAPL", "MSFT"],
                "fields": ["open", "high", "low", "close", "volume"],
                "lookback": 20,
                "frequency": "4h"
            },
            "fundamentals_req": {
                "data_source_name": "fundamentals_csv",
                "frequency": "1d",
                "lookback": 1,
                "align_method": "forward_fill"
                "align_data": True
            }
        }

# 6) Initialize backsim (Event Engine) & run
engine = BackSim(
    tick_size="1h",  # use the smallest resolution needed by any strategy
    asset_universe=asset_universe,
    portfolio=portfolio,
    broker=broker,
    strategies=[MyStrategy()],
    external_data_sources={"fundamentals_csv": fundamentals_data_source}
)

engine.run(start_date='2020-01-01', end_date='2020-12-31')
```

Key points:

- The AssetUniverse contains only standardized frequency OHLCV data.
- Additional or alternative data stays in one or more DataSource objects.
- Frequency alignment (e.g. daily fundamentals vs. 1-hour price bars) is left to the user’s custom or built-in data_slicer logic. The strategy is also free to use data at mixed frequencies.
- The Event Engine orchestrates the time steps, asking each Strategy to produce orders, updating the Portfolio, processing fills, and logging results.
