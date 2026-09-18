use std::{
    collections::HashSet,
    time::{Duration, SystemTime},
};

use anyhow::{ensure, Result};
use reqwest::header::HeaderMap;

use super::TypeSafeError;

#[derive(Debug, Clone)]
pub struct RetryPolicy {
    pub max_retries: u32,
    pub backoff_initial: Duration,
    pub backoff_max: Duration,
    /// Fraction randomly subtracted from exponential backoff, between zero and one.
    pub jitter: f64,
    pub statuses: HashSet<u16>,
    pub respect_retry_after: bool,
    pub max_retry_after: Duration,
    pub connection_errors: bool,
    pub timeouts: bool,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 2,
            backoff_initial: Duration::from_millis(500),
            backoff_max: Duration::from_secs(5),
            jitter: 0.25,
            statuses: [408, 429].into_iter().chain(500..600).collect(),
            respect_retry_after: true,
            max_retry_after: Duration::from_secs(60),
            connection_errors: true,
            timeouts: true,
        }
    }
}

impl RetryPolicy {
    pub(crate) fn validate(&self) -> Result<()> {
        ensure!(
            self.jitter.is_finite() && (0.0..=1.0).contains(&self.jitter),
            TypeSafeError::InvalidInput("Retry jitter must be between zero and one".into())
        );
        ensure!(
            self.statuses
                .iter()
                .all(|status| (100..=999).contains(status)),
            TypeSafeError::InvalidInput("Retry statuses must be HTTP status codes".into())
        );
        Ok(())
    }

    pub(crate) fn delay(&self, attempt: u32, headers: Option<&HeaderMap>) -> Duration {
        if self.respect_retry_after {
            if let Some(delay) =
                headers.and_then(|headers| parse_retry_after(headers, SystemTime::now()))
            {
                if delay <= self.max_retry_after {
                    return delay;
                }
            }
        }
        let exponential = self
            .backoff_initial
            .saturating_mul(2u32.saturating_pow(attempt))
            .min(self.backoff_max);
        exponential.mul_f64(1.0 - rand::random::<f64>() * self.jitter)
    }
}

/// Parse milliseconds first, then fractional seconds or an HTTP date.
pub fn parse_retry_after(headers: &HeaderMap, now: SystemTime) -> Option<Duration> {
    fn seconds(raw: &str, divisor: f64) -> Option<Duration> {
        Duration::try_from_secs_f64(raw.trim().parse::<f64>().ok()? / divisor).ok()
    }
    if let Some(delay) = headers
        .get("retry-after-ms")
        .and_then(|value| value.to_str().ok())
        .and_then(|raw| seconds(raw, 1000.0))
    {
        return Some(delay);
    }
    let raw = headers.get("retry-after")?.to_str().ok()?;
    seconds(raw, 1.0).or_else(|| {
        httpdate::parse_http_date(raw)
            .ok()
            .map(|date| date.duration_since(now).unwrap_or_default())
    })
}
