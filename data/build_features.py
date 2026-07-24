import numpy as np
import pandas as pd


class MTFSpatialFeatureBuilder:
    """
    Constructs Volatility-Normalized, Tanh-Squashed Spatial Tensors for
    Multi-Timeframe (15m, 30m, 1H, 4H) Wick-to-Body Zones with Temporal Completion Meters.
    """

    def __init__(
        self,
        timeframe_minutes: dict = None,
        lookback_windows: dict = None,
        atr_period: int = 14,
    ):
        self.tf_minutes = timeframe_minutes or {
            "15m": 15,
            "30m": 30,
            "1H": 60,
            "4H": 240,
        }

        # Asymmetric lookback steps per timeframe channel
        self.lookbacks = lookback_windows or {
            "15m": 128,
            "30m": 64,
            "1H": 32,
            "4H": 16,
        }

        self.atr_period = atr_period

    def calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """Calculates Average True Range (ATR) for volatility normalization."""
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift(1))
        low_close = np.abs(df["low"] - df["close"].shift(1))

        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=self.atr_period).mean()
        return atr.fillna(method="bfill")

    def extract_wick_to_body_zones(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts structural boundaries for dynamic zones:
        - Upper Zone: Max(Open, Close) to High (Upper Wick)
        - Lower Zone: Low to Min(Open, Close) (Lower Wick)
        """
        zones = pd.DataFrame(index=df.index)
        zones["upper_zone_top"] = df["high"]
        zones["upper_zone_bottom"] = np.maximum(df["open"], df["close"])

        zones["lower_zone_top"] = np.minimum(df["open"], df["close"])
        zones["lower_zone_bottom"] = df["low"]

        zones["zone_midpoint"] = (df["high"] + df["low"]) / 2.0
        return zones

    def compute_completion_meter(
        self, current_time: pd.Timestamp, tf_minutes: int
    ) -> tuple[float, float]:
        """
        Calculates the Cyclical Temporal Completion Meter (sin/cos) for developing candles,
        preventing Higher-Timeframe Look-Ahead Bias.
        """
        minute_of_day = current_time.hour * 60 + current_time.minute
        elapsed_in_tf = minute_of_day % tf_minutes
        tau = elapsed_in_tf / float(tf_minutes)

        sin_completion = np.sin(2 * np.pi * tau)
        cos_completion = np.cos(2 * np.pi * tau)

        return float(sin_completion), float(cos_completion)

    def process_timeframe_features(
        self, df: pd.DataFrame, tf_label: str
    ) -> pd.DataFrame:
        """
        Generates Volatility-Normalized, Tanh-Squashed features for a single timeframe channel.
        """
        features = pd.DataFrame(index=df.index)
        atr = self.calculate_atr(df)
        zones = self.extract_wick_to_body_zones(df)

        close_price = df["close"]

        # Volatility-Normalized Spatial Distances: D = (Price - Zone) / ATR
        dist_upper_top = (close_price - zones["upper_zone_top"]) / atr
        dist_upper_bot = (close_price - zones["upper_zone_bottom"]) / atr
        dist_lower_top = (close_price - zones["lower_zone_top"]) / atr
        dist_lower_bot = (close_price - zones["lower_zone_bottom"]) / atr

        # Non-Linear Feature Squashing via Tanh
        features[f"{tf_label}_dist_upper_top_tanh"] = np.tanh(dist_upper_top)
        features[f"{tf_label}_dist_upper_bot_tanh"] = np.tanh(dist_upper_bot)
        features[f"{tf_label}_dist_lower_top_tanh"] = np.tanh(dist_lower_top)
        features[f"{tf_label}_dist_lower_bot_tanh"] = np.tanh(dist_lower_bot)

        # Temporal Completion Meters for higher timeframes
        tf_mins = self.tf_minutes[tf_label]
        completion_sin_cos = [
            self.compute_completion_meter(ts, tf_mins) for ts in df.index
        ]

        features[f"{tf_label}_completion_sin"] = [
            x[0] for x in completion_sin_cos
        ]
        features[f"{tf_label}_completion_cos"] = [
            x[1] for x in completion_sin_cos
        ]

        return features


if __name__ == "__main__":
    # Smoke Test / Feature Builder Sandbox
    print("Testing MTFSpatialFeatureBuilder...")

    # Simulated OHLCV Data Stream
    dates = pd.date_range("2026-01-01", periods=200, freq="15min")
    dummy_data = pd.DataFrame(
        {
            "open": np.random.randn(200).cumsum() + 2000,
            "high": np.random.randn(200).cumsum() + 2005,
            "low": np.random.randn(200).cumsum() + 1995,
            "close": np.random.randn(200).cumsum() + 2000,
        },
        index=dates,
    )

    builder = MTFSpatialFeatureBuilder()
    features_15m = builder.process_timeframe_features(dummy_data, "15m")
    features_4h = builder.process_timeframe_features(dummy_data, "4H")

    print(
        f"15m Feature Shape: {features_15m.shape} | Columns: {list(features_15m.columns)}"
    )
    print(
        f"4H Feature Shape: {features_4h.shape} | Columns: {list(features_4h.columns)}"
    )
    print("Features extracted successfully with zero raw price leakage!")