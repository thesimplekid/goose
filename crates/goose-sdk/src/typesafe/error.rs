use std::time::Duration;

use reqwest::{header::HeaderMap, StatusCode};
use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApiErrorKind {
    BadRequest,
    Authentication,
    PermissionDenied,
    NotFound,
    UnprocessableEntity,
    RateLimited,
    Server,
    Other,
}

#[derive(Debug, thiserror::Error)]
#[error("TypeSafe HTTP {status}: {message}")]
pub struct ApiError {
    pub kind: ApiErrorKind,
    pub status: StatusCode,
    pub headers: HeaderMap,
    pub body: Value,
    pub request_id: Option<String>,
    pub retry_after: Option<Duration>,
    pub message: String,
}

impl ApiError {
    pub(crate) fn new(status: StatusCode, headers: HeaderMap, bytes: &[u8]) -> Self {
        let body = serde_json::from_slice(bytes).unwrap_or_else(|_| {
            if bytes.is_empty() {
                Value::Null
            } else {
                Value::String(String::from_utf8_lossy(bytes).into_owned())
            }
        });
        let message = error_message(&body);
        Self {
            kind: match status.as_u16() {
                400 => ApiErrorKind::BadRequest,
                401 => ApiErrorKind::Authentication,
                403 => ApiErrorKind::PermissionDenied,
                404 => ApiErrorKind::NotFound,
                422 => ApiErrorKind::UnprocessableEntity,
                429 => ApiErrorKind::RateLimited,
                500..=599 => ApiErrorKind::Server,
                _ => ApiErrorKind::Other,
            },
            status,
            request_id: super::client::request_id(&headers),
            retry_after: super::parse_retry_after(&headers, std::time::SystemTime::now()),
            headers,
            body,
            message,
        }
    }
}

fn error_message(body: &Value) -> String {
    for value in [body.get("error"), body.get("message"), body.get("detail")]
        .into_iter()
        .flatten()
    {
        if let Some(message) = value
            .as_str()
            .or_else(|| value.get("message").and_then(Value::as_str))
        {
            return message.to_owned();
        }
    }
    if let Some(details) = body.get("detail").and_then(Value::as_array) {
        let messages: Vec<_> = details
            .iter()
            .filter_map(|detail| {
                let message = detail.get("msg")?.as_str()?;
                let path = detail
                    .get("loc")
                    .and_then(Value::as_array)
                    .map(|parts| {
                        parts
                            .iter()
                            .filter(|part| part.as_str() != Some("body"))
                            .map(|part| {
                                part.as_str()
                                    .map(str::to_owned)
                                    .unwrap_or_else(|| part.to_string())
                            })
                            .collect::<Vec<_>>()
                            .join(".")
                    })
                    .unwrap_or_default();
                Some(if path.is_empty() {
                    message.to_owned()
                } else {
                    format!("{path}: {message}")
                })
            })
            .collect();
        if !messages.is_empty() {
            return messages.join("; ");
        }
    }
    let text = body
        .as_str()
        .map(str::to_owned)
        .unwrap_or_else(|| body.to_string());
    if text.chars().count() > 200 {
        format!("{}…", text.chars().take(200).collect::<String>())
    } else {
        text
    }
}

/// Public errors remain downcastable through the client's anyhow::Result.
#[derive(Debug, thiserror::Error)]
pub enum TypeSafeError {
    #[error("Invalid TypeSafe input: {0}")]
    InvalidInput(String),
    #[error(transparent)]
    Api(#[from] Box<ApiError>),
    #[error("TypeSafe connection failed: {0}")]
    Connection(#[source] reqwest::Error),
    #[error("TypeSafe attempt timed out after {0:?}")]
    Timeout(Duration),
    #[error("Invalid TypeSafe response: {0}")]
    Decode(#[source] serde_json::Error),
    #[error("Invalid TypeSafe response: {0}")]
    Response(String),
}
