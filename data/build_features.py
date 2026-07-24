import numpy as np
import pandas as pd


class MTFDatasetGenerator:

    def __init__(self, atr_period: int = 14):
        self.atr_period = atr_period
        self.tf_mapping = {
            "15m": "15min",
            "30m": "30min",
            "1H": "1h",
            "4H": "4h",
        }
        self.tf_minutes = {"15m": 15, "30m": 30, "1H": 60, "4H": 240}

    def resample_ohlcv(
        self, df_1m: pd.DataFrame, freq_str: str
    ) -> pd.DataFrame:
        """Resamples 1-minute OHLCV data into target higher timeframe candles safely."""
        resampled = (
            df_1m.resample(freq_str)
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
        )
        return resampled

    def extract_session_and_daily_levels(
        self, df_1m: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Extracts Previous Daily High/Low (PDH/PDL), Daily Pivots,
        Asia Session (00:00 - 08:00 UTC) & London Session (08:00 - 16:00 UTC) Highs/Lows from 1m data.
        """
        levels = pd.DataFrame(index=df_1m.index)

        # 1. Previous Daily High/Low & Pivot
        daily_df = self.resample_ohlcv(df_1m, "1D")
        daily_high = daily_df["high"].shift(1)
        daily_low = daily_df["low"].shift(1)
        daily_close = daily_df["close"].shift(1)

        pivot_point = (daily_high + daily_low + daily_close) / 3.0

        # Forward fill daily levels onto 1m timeline
        levels["pdh"] = daily_high.reindex(df_1m.index, method="ffill")
        levels["pdl"] = daily_low.reindex(df_1m.index, method="ffill")
        levels["pivot"] = pivot_point.reindex(df_1m.index, method="ffill")

        # 2. Asian Session High/Low (00:00 - 08:00 UTC)
        asia_mask = (df_1m.index.hour >= 0) & (df_1m.index.hour < 8)
        asia_high = (
            df_1m["high"].where(asia_mask).groupby(df_1m.index.date).transform("max")
        )
        asia_low = (
            df_1m["low"].where(asia_mask).groupby(df_1m.index.date).transform("min")
        )

        levels["asia_high"] = asia_high.ffill()
        levels["asia_low"] = asia_low.ffill()

        # 3. London Session High/Low (08:00 - 16:00 UTC)
        london_mask = (df_1m.index.hour >= 8) & (df_1m.index.hour < 16)
        london_high = (
            df_1m["high"]
            .where(london_mask)
            .groupby(df_1m.index.date)
            .transform("max")
        )
        london_low = (
            df_1m["low"].where(london_mask).groupby(df_1m.index.date).transform("min")
        )

        levels["london_high"] = london_high.ffill()
        levels["london_low"] = london_low.ffill()

        return levels

    def process_raw_1m_dataset(
        self, raw_1m_filepath: str
    ) -> dict[str, pd.DataFrame]:
        """
        Accepts a single raw 1m CSV/Parquet file, auto-resamples 15m, 30m, 1H, 4H datasets,
        extracts session levels, and computes ATR-normalized tanh-squashed spatial features.
        """
        print(f"Loading raw 1m dataset from {raw_1m_filepath}...")
        if raw_1m_filepath.endswith(".csv"):
            df_1m = pd.read_csv(raw_1m_filepath, parse_dates=True, index_col=0)
        else:
            df_1m = pd.read_parquet(raw_1m_filepath)

        # Standardize column headers
        df_1m.columns = [c.lower() for c in df_1m.columns]

        print("Extracting Session Liquidity & Daily Levels...")
        session_levels = self.extract_session_and_daily_levels(df_1m)

        mtf_processed = {}

        for tf_label, freq in self.tf_mapping.items():
            print(f"Auto-resampling & processing {tf_label} channel...")
            resampled_df = self.resample_ohlcv(df_1m, freq)

            # Calculate ATR for volatility scaling
            high_low = resampled_df["high"] - resampled_df["low"]
            high_close = np.abs(resampled_df["high"] - resampled_df["close"].shift(1))
            low_close = np.abs(resampled_df["low"] - resampled_df["close"].shift(1))
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = (
                true_range.rolling(window=self.atr_period)
                .mean()
                .bfill()
            )

            # Build Wick-to-Body Zones
            upper_zone_top = resampled_df["high"]
            upper_zone_bot = np.maximum(resampled_df["open"], resampled_df["close"])
            lower_zone_top = np.minimum(resampled_df["open"], resampled_df["close"])
            lower_zone_bot = resampled_df["low"]

            features = pd.DataFrame(index=resampled_df.index)
            close = resampled_df["close"]

            # 1. Volatility-Normalized Tanh Zone Distances
            features["dist_upper_top_tanh"] = np.tanh((close - upper_zone_top) / atr)
            features["dist_upper_bot_tanh"] = np.tanh((close - upper_zone_bot) / atr)
            features["dist_lower_top_tanh"] = np.tanh((close - lower_zone_top) / atr)
            features["dist_lower_bot_tanh"] = np.tanh((close - lower_zone_bot) / atr)

            # 2. Session/Daily Level Distances
            tf_session_levels = session_levels.reindex(
                resampled_df.index, method="ffill"
            )
            features["dist_pdh_tanh"] = np.tanh(
                (close - tf_session_levels["pdh"]) / atr
            )
            features["dist_pdl_tanh"] = np.tanh(
                (close - tf_session_levels["pdl"]) / atr
            )
            features["dist_pivot_tanh"] = np.tanh(
                (close - tf_session_levels["pivot"]) / atr
            )
            features["dist_asia_high_tanh"] = np.tanh(
                (close - tf_session_levels["asia_high"]) / atr
            )
            features["dist_asia_low_tanh"] = np.tanh(
                (close - tf_session_levels["asia_low"]) / atr
            )

            # 3. Cyclical Completion Meters
            tf_mins = self.tf_minutes[tf_label]
            minute_of_day = resampled_df.index.hour * 60 + resampled_df.index.minute
            tau = (minute_of_day % tf_mins) / float(tf_mins)
            features["completion_sin"] = np.sin(2 * np.pi * tau)
            features["completion_cos"] = np.cos(2 * np.pi * tau)

            mtf_processed[tf_label] = features.dropna()

        print(
            "Automated 1m Resampling & Multi-Timeframe Feature Pipeline Complete!"
        )
        return mtf_processed


if __name__ == "__main__":
    # Generator Test
    dates = pd.date_range("2026-01-01", periods=10000, freq="1min")
    dummy_1m = pd.DataFrame(
        {
            "open": np.random.randn(10000).cumsum() + 2000,
            "high": np.random.randn(10000).cumsum() + 2002,
            "low": np.random.randn(10000).cumsum() + 1998,
            "close": np.random.randn(10000).cumsum() + 2000,
            "volume": np.random.randint(10, 500, size=10000),
        },
        index=dates,
    )

    dummy_1m.to_csv("data/dummy_1m.csv")

    generator = MTFDatasetGenerator()
    dataset_dict = generator.process_raw_1m_dataset("data/dummy_1m.csv")

    for tf, df in dataset_dict.items():
        print(f"Timeframe: {tf} | Processed Shape: {df.shape}")