//! API response types for Bybit

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Bybit API error types
#[derive(Error, Debug)]
pub enum BybitError {
    #[error("HTTP request failed: {0}")]
    RequestError(#[from] reqwest::Error),

    #[error("API error: code={code}, message={message}")]
    ApiError { code: i32, message: String },

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Invalid symbol: {0}")]
    InvalidSymbol(String),

    #[error("Rate limit exceeded")]
    RateLimitExceeded,
}

/// Generic API response wrapper
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ApiResponse<T> {
    pub ret_code: i32,
    pub ret_msg: String,
    pub result: Option<T>,
    pub time: Option<u64>,
}

/// Klines result from API
#[derive(Debug, Deserialize)]
pub struct KlinesResult {
    pub symbol: String,
    pub category: String,
    pub list: Vec<Vec<String>>,
}

/// OHLCV candlestick data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kline {
    /// Start time (Unix timestamp in ms)
    pub timestamp: u64,
    /// Open price
    pub open: f64,
    /// High price
    pub high: f64,
    /// Low price
    pub low: f64,
    /// Close price
    pub close: f64,
    /// Trading volume
    pub volume: f64,
    /// Turnover (quote volume)
    pub turnover: f64,
}

impl Kline {
    /// Parse kline from Bybit API array format
    /// Format: [startTime, openPrice, highPrice, lowPrice, closePrice, volume, turnover]
    pub fn from_bybit_array(arr: &[String]) -> Result<Self, BybitError> {
        if arr.len() < 7 {
            return Err(BybitError::ParseError(format!(
                "Expected 7 elements, got {}",
                arr.len()
            )));
        }

        Ok(Self {
            timestamp: arr[0]
                .parse()
                .map_err(|e| BybitError::ParseError(format!("Invalid timestamp: {}", e)))?,
            open: arr[1]
                .parse()
                .map_err(|e| BybitError::ParseError(format!("Invalid open: {}", e)))?,
            high: arr[2]
                .parse()
                .map_err(|e| BybitError::ParseError(format!("Invalid high: {}", e)))?,
            low: arr[3]
                .parse()
                .map_err(|e| BybitError::ParseError(format!("Invalid low: {}", e)))?,
            close: arr[4]
                .parse()
                .map_err(|e| BybitError::ParseError(format!("Invalid close: {}", e)))?,
            volume: arr[5]
                .parse()
                .map_err(|e| BybitError::ParseError(format!("Invalid volume: {}", e)))?,
            turnover: arr[6]
                .parse()
                .map_err(|e| BybitError::ParseError(format!("Invalid turnover: {}", e)))?,
        })
    }

    /// Calculate log return from previous close
    pub fn log_return(&self, prev_close: f64) -> f64 {
        if prev_close > 0.0 {
            (self.close / prev_close).ln()
        } else {
            0.0
        }
    }

    /// Calculate simple return from previous close
    pub fn simple_return(&self, prev_close: f64) -> f64 {
        if prev_close > 0.0 {
            (self.close - prev_close) / prev_close
        } else {
            0.0
        }
    }

    /// Calculate high-low range as percentage of close
    pub fn range_pct(&self) -> f64 {
        if self.close > 0.0 {
            (self.high - self.low) / self.close
        } else {
            0.0
        }
    }
}

/// Ticker result from API
#[derive(Debug, Deserialize)]
pub struct TickersResult {
    pub category: String,
    pub list: Vec<TickerRaw>,
}

/// Raw ticker data from API
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TickerRaw {
    pub symbol: String,
    pub last_price: String,
    pub index_price: Option<String>,
    pub mark_price: Option<String>,
    pub prev_price24h: Option<String>,
    pub price24h_pcnt: Option<String>,
    pub high_price24h: Option<String>,
    pub low_price24h: Option<String>,
    pub volume24h: Option<String>,
    pub turnover24h: Option<String>,
    pub open_interest: Option<String>,
    pub funding_rate: Option<String>,
    pub next_funding_time: Option<String>,
    pub bid1_price: Option<String>,
    pub bid1_size: Option<String>,
    pub ask1_price: Option<String>,
    pub ask1_size: Option<String>,
}

impl TickerRaw {
    /// Convert to structured Ticker
    pub fn to_ticker(&self) -> Result<Ticker, BybitError> {
        Ok(Ticker {
            symbol: self.symbol.clone(),
            last_price: self
                .last_price
                .parse()
                .map_err(|e| BybitError::ParseError(format!("Invalid last_price: {}", e)))?,
            index_price: self.index_price.as_ref().and_then(|s| s.parse().ok()),
            mark_price: self.mark_price.as_ref().and_then(|s| s.parse().ok()),
            prev_price_24h: self.prev_price24h.as_ref().and_then(|s| s.parse().ok()),
            price_change_24h_pct: self.price24h_pcnt.as_ref().and_then(|s| s.parse().ok()),
            high_price_24h: self.high_price24h.as_ref().and_then(|s| s.parse().ok()),
            low_price_24h: self.low_price24h.as_ref().and_then(|s| s.parse().ok()),
            volume_24h: self.volume24h.as_ref().and_then(|s| s.parse().ok()),
            turnover_24h: self.turnover24h.as_ref().and_then(|s| s.parse().ok()),
            open_interest: self.open_interest.as_ref().and_then(|s| s.parse().ok()),
            funding_rate: self.funding_rate.as_ref().and_then(|s| s.parse().ok()),
            bid_price: self.bid1_price.as_ref().and_then(|s| s.parse().ok()),
            bid_size: self.bid1_size.as_ref().and_then(|s| s.parse().ok()),
            ask_price: self.ask1_price.as_ref().and_then(|s| s.parse().ok()),
            ask_size: self.ask1_size.as_ref().and_then(|s| s.parse().ok()),
        })
    }
}

/// Structured ticker information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ticker {
    pub symbol: String,
    pub last_price: f64,
    pub index_price: Option<f64>,
    pub mark_price: Option<f64>,
    pub prev_price_24h: Option<f64>,
    pub price_change_24h_pct: Option<f64>,
    pub high_price_24h: Option<f64>,
    pub low_price_24h: Option<f64>,
    pub volume_24h: Option<f64>,
    pub turnover_24h: Option<f64>,
    pub open_interest: Option<f64>,
    pub funding_rate: Option<f64>,
    pub bid_price: Option<f64>,
    pub bid_size: Option<f64>,
    pub ask_price: Option<f64>,
    pub ask_size: Option<f64>,
}

/// Order book level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookLevel {
    pub price: f64,
    pub quantity: f64,
}

/// Order book snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    pub symbol: String,
    pub timestamp: u64,
    pub bids: Vec<OrderBookLevel>,
    pub asks: Vec<OrderBookLevel>,
}

impl OrderBook {
    /// Calculate mid price
    pub fn mid_price(&self) -> Option<f64> {
        match (self.bids.first(), self.asks.first()) {
            (Some(bid), Some(ask)) => Some((bid.price + ask.price) / 2.0),
            _ => None,
        }
    }

    /// Calculate bid-ask spread
    pub fn spread(&self) -> Option<f64> {
        match (self.bids.first(), self.asks.first()) {
            (Some(bid), Some(ask)) => Some(ask.price - bid.price),
            _ => None,
        }
    }

    /// Calculate spread as percentage of mid price
    pub fn spread_pct(&self) -> Option<f64> {
        match (self.mid_price(), self.spread()) {
            (Some(mid), Some(spread)) if mid > 0.0 => Some(spread / mid),
            _ => None,
        }
    }

    /// Calculate order book imbalance
    pub fn imbalance(&self, depth: usize) -> f64 {
        let bid_volume: f64 = self.bids.iter().take(depth).map(|l| l.quantity).sum();
        let ask_volume: f64 = self.asks.iter().take(depth).map(|l| l.quantity).sum();
        let total = bid_volume + ask_volume;

        if total > 0.0 {
            (bid_volume - ask_volume) / total
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kline_from_bybit_array() {
        let arr = vec![
            "1704067200000".to_string(),
            "42000.5".to_string(),
            "42500.0".to_string(),
            "41800.0".to_string(),
            "42100.0".to_string(),
            "1234.56".to_string(),
            "51234567.89".to_string(),
        ];

        let kline = Kline::from_bybit_array(&arr).unwrap();

        assert_eq!(kline.timestamp, 1704067200000);
        assert!((kline.open - 42000.5).abs() < 0.01);
        assert!((kline.close - 42100.0).abs() < 0.01);
    }

    #[test]
    fn test_kline_returns() {
        let kline = Kline {
            timestamp: 0,
            open: 100.0,
            high: 105.0,
            low: 98.0,
            close: 102.0,
            volume: 1000.0,
            turnover: 100000.0,
        };

        let log_ret = kline.log_return(100.0);
        assert!((log_ret - 0.0198).abs() < 0.001);

        let simple_ret = kline.simple_return(100.0);
        assert!((simple_ret - 0.02).abs() < 0.001);
    }

    #[test]
    fn test_orderbook_calculations() {
        let orderbook = OrderBook {
            symbol: "BTCUSDT".to_string(),
            timestamp: 0,
            bids: vec![
                OrderBookLevel { price: 42000.0, quantity: 1.0 },
                OrderBookLevel { price: 41999.0, quantity: 2.0 },
            ],
            asks: vec![
                OrderBookLevel { price: 42001.0, quantity: 1.5 },
                OrderBookLevel { price: 42002.0, quantity: 2.5 },
            ],
        };

        assert!((orderbook.mid_price().unwrap() - 42000.5).abs() < 0.01);
        assert!((orderbook.spread().unwrap() - 1.0).abs() < 0.01);

        let imbalance = orderbook.imbalance(2);
        // (1+2) - (1.5+2.5) / (1+2+1.5+2.5) = (3 - 4) / 7 = -0.143
        assert!((imbalance - (-0.143)).abs() < 0.01);
    }
}
