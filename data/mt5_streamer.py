import MetaTrader5 as mt5
import pandas as pd
import os
from datetime import datetime

class MT5DataStreamer:
    """
    Handles high-fidelity historical data extraction and live polling from MetaTrader 5.
    Downloads asymmetric timeframes required for the Tri-Brain Spatial Tensor.
    """
    def __init__(self, symbol: str = "XAUUSD"):
        self.symbol = symbol
        self.timeframes = {
            "15m": mt5.TIMEFRAME_M15,
            "30m": mt5.TIMEFRAME_M30,
            "1H": mt5.TIMEFRAME_H1,
            "4H": mt5.TIMEFRAME_H4
        }
        
        if not mt5.initialize():
            print("Failed to initialize MT5. Check if terminal is open.")
            mt5.shutdown()
            raise ConnectionError("MT5 Initialization Failed.")
        print(f"Successfully connected to MT5. Targeting symbol: {self.symbol}")

    def fetch_historical_data(self, timeframe_label: str, num_bars: int = 100000) -> pd.DataFrame:
        """
        Pulls raw OHLCV data for a specific timeframe.
        100,000 bars on M15 gives us roughly 4 years of continuous market structure.
        """
        mt5_tf = self.timeframes.get(timeframe_label)
        if not mt5_tf:
            raise ValueError(f"Invalid timeframe label: {timeframe_label}")

        print(f"Fetching {num_bars} bars for {self.symbol} at {timeframe_label}...")
        
        # Pull rates from the current moment backwards
        rates = mt5.copy_rates_from_pos(self.symbol, mt5_tf, 0, num_bars)
        
        if rates is None or len(rates) == 0:
            raise ValueError(f"No data returned for {self.symbol} on {timeframe_label}. Check MT5 history limits.")

        # Convert to a Pandas DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        # Drop unnecessary MT5 columns to keep the tensor clean
        df = df[['open', 'high', 'low', 'close', 'tick_volume']]
        df.rename(columns={'tick_volume': 'volume'}, inplace=True)
        
        return df

    def build_historical_dataset(self, output_dir: str = "data/raw_parquet"):
        """
        Extracts all required timeframes and saves them to Parquet for fast PyTorch loading.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for tf_label in self.timeframes.keys():
            df = self.fetch_historical_data(tf_label, num_bars=100000)
            
            filepath = os.path.join(output_dir, f"XAUUSD_{tf_label}_raw.parquet")
            df.to_parquet(filepath, engine='pyarrow', compression='snappy')
            print(f"Saved {tf_label} data to {filepath} | Shape: {df.shape}")

        print("Historical Dataset Extraction Complete.")

    def shutdown(self):
        mt5.shutdown()
        print("MT5 Connection Closed.")

if __name__ == "__main__":
    # Extraction Execution
    streamer = MT5DataStreamer(symbol="XAUUSD")
    try:
        streamer.build_historical_dataset()
    finally:
        streamer.shutdown()