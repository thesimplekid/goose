use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::{ensure, Result};
use reqwest::{
    header::{HeaderMap, HeaderValue, ACCEPT, AUTHORIZATION, CONTENT_TYPE},
    Client, Method, StatusCode, Url,
};
use serde::{de::DeserializeOwned, Deserialize};
use serde_json::Value;

use super::{
    validate_entry, Answer, ApiError, Evaluation, ModelCard, Question, RetryPolicy, TypeSafeError,
    DEFAULT_BASE_URL, DEFAULT_MODEL,
};

/// HTTP metadata and the buffered body are retained for diagnostics and custom decoding.
#[derive(Debug)]
pub struct ApiResponse<T> {
    pub data: T,
    pub status: StatusCode,
    pub headers: HeaderMap,
    pub request_id: Option<String>,
    pub raw_body: Vec<u8>,
}

impl<T> ApiResponse<T> {
    fn map<U>(self, f: impl FnOnce(T) -> U) -> ApiResponse<U> {
        ApiResponse {
            data: f(self.data),
            status: self.status,
            headers: self.headers,
            request_id: self.request_id,
            raw_body: self.raw_body,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct RequestOptions {
    /// Per-attempt deadline including the response body. Defaults to the client timeout.
    pub timeout: Option<Duration>,
    /// A complete per-call policy; absent fields should be filled with RetryPolicy::default().
    pub retry: Option<RetryPolicy>,
    pub headers: HeaderMap,
}

/// Metadata-only callback; API keys, input state, headers, and response bodies are excluded.
#[derive(Debug, Clone)]
pub enum RequestEvent {
    AttemptFinished {
        path: String,
        attempt: u32,
        elapsed: Duration,
        status: Option<StatusCode>,
        request_id: Option<String>,
        failed: bool,
    },
    Retrying {
        path: String,
        attempt: u32,
        delay: Duration,
    },
}

/// Called synchronously; implementations should return quickly. Panics are isolated.
pub trait RequestObserver: Send + Sync {
    fn on_event(&self, event: &RequestEvent);
}

#[derive(Default)]
pub struct TypeSafeClientBuilder {
    api_key: Option<String>,
    base_url: Option<String>,
    default_model: Option<String>,
    timeout: Option<Duration>,
    retry: RetryPolicy,
    headers: HeaderMap,
    http: Option<Client>,
    observer: Option<Arc<dyn RequestObserver>>,
}

impl TypeSafeClientBuilder {
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }
    pub fn default_model(mut self, model: impl Into<String>) -> Self {
        self.default_model = Some(model.into());
        self
    }
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }
    pub fn retry(mut self, retry: RetryPolicy) -> Self {
        self.retry = retry;
        self
    }
    pub fn headers(mut self, headers: HeaderMap) -> Self {
        self.headers = headers;
        self
    }
    /// Supply a client configured with your proxy, TLS, pool, and transport settings.
    pub fn http_client(mut self, http: Client) -> Self {
        self.http = Some(http);
        self
    }
    pub fn observer(mut self, observer: Arc<dyn RequestObserver>) -> Self {
        self.observer = Some(observer);
        self
    }

    /// Explicit settings override nonempty environment variables, then SDK defaults.
    pub fn build(self) -> Result<TypeSafeClient> {
        let api_key = self
            .api_key
            .or_else(|| env("TYPESAFE_API_KEY"))
            .ok_or_else(|| {
                TypeSafeError::InvalidInput("Pass an API key or set TYPESAFE_API_KEY".into())
            })?;
        ensure!(
            !api_key.trim().is_empty(),
            TypeSafeError::InvalidInput("TypeSafe API key is empty".into())
        );
        let mut authorization =
            HeaderValue::from_str(&format!("Bearer {api_key}")).map_err(|_| {
                TypeSafeError::InvalidInput("API key is not a valid HTTP header value".into())
            })?;
        authorization.set_sensitive(true);
        let raw_url = self
            .base_url
            .or_else(|| env("TYPESAFE_BASE_URL"))
            .unwrap_or_else(|| DEFAULT_BASE_URL.into());
        let mut base_url =
            Url::parse(&raw_url).map_err(|error| TypeSafeError::InvalidInput(error.to_string()))?;
        ensure!(
            matches!(base_url.scheme(), "http" | "https") && base_url.host_str().is_some(),
            TypeSafeError::InvalidInput("TypeSafe base URL must be HTTP or HTTPS".into())
        );
        ensure!(
            base_url.username().is_empty()
                && base_url.password().is_none()
                && base_url.query().is_none()
                && base_url.fragment().is_none(),
            TypeSafeError::InvalidInput(
                "TypeSafe base URL must not contain credentials, a query, or a fragment".into()
            )
        );
        let prefix = base_url.path().trim_end_matches('/');
        let path = if prefix.ends_with("/v1") {
            format!("{prefix}/")
        } else {
            format!("{prefix}/v1/")
        };
        base_url.set_path(&path);
        let default_model = self
            .default_model
            .or_else(|| env("TYPESAFE_DEFAULT_MODEL"))
            .unwrap_or_else(|| DEFAULT_MODEL.into());
        ensure!(
            !default_model.trim().is_empty(),
            TypeSafeError::InvalidInput("TypeSafe model is empty".into())
        );
        let timeout = self.timeout.unwrap_or(Duration::from_secs(10));
        validate_timeout(timeout)?;
        self.retry.validate()?;
        let http = match self.http {
            Some(http) => http,
            None => Client::builder()
                .redirect(reqwest::redirect::Policy::none())
                .build()
                .map_err(TypeSafeError::Connection)?,
        };
        Ok(TypeSafeClient {
            http,
            base_url,
            authorization,
            default_model,
            timeout,
            retry: self.retry,
            headers: self.headers,
            observer: self.observer,
        })
    }
}

fn env(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .filter(|value| !value.trim().is_empty())
}

#[derive(Clone)]
pub struct TypeSafeClient {
    http: Client,
    base_url: Url,
    authorization: HeaderValue,
    default_model: String,
    timeout: Duration,
    retry: RetryPolicy,
    headers: HeaderMap,
    observer: Option<Arc<dyn RequestObserver>>,
}

impl TypeSafeClient {
    pub fn builder() -> TypeSafeClientBuilder {
        TypeSafeClientBuilder::default()
    }
    pub fn new(api_key: String) -> Result<Self> {
        Self::builder().api_key(api_key).build()
    }
    pub fn from_env() -> Result<Self> {
        Self::builder().build()
    }
    /// `base_url` is an origin or proxy prefix, optionally ending in `/v1`.
    pub fn with_base_url(api_key: String, base_url: &str) -> Result<Self> {
        Self::builder().api_key(api_key).base_url(base_url).build()
    }
    pub fn default_model(&self) -> &str {
        &self.default_model
    }

    /// Use the client's default model. Dropping the future cancels requests and backoff.
    pub async fn system_one(
        &self,
        state: Value,
        questions: HashMap<String, Question>,
    ) -> Result<Evaluation> {
        Ok(self
            .evaluate_with_options(None, state, questions, RequestOptions::default())
            .await?
            .data)
    }

    /// Evaluate with an explicit model, preserving the original GDK entry point.
    pub async fn evaluate(
        &self,
        model: &str,
        state: Value,
        questions: HashMap<String, Question>,
    ) -> Result<Evaluation> {
        Ok(self
            .evaluate_with_options(Some(model), state, questions, RequestOptions::default())
            .await?
            .data)
    }

    pub async fn evaluate_with_options(
        &self,
        model: Option<&str>,
        state: Value,
        questions: HashMap<String, Question>,
        options: RequestOptions,
    ) -> Result<ApiResponse<Evaluation>> {
        let model = model.unwrap_or(&self.default_model);
        ensure!(
            !model.trim().is_empty(),
            TypeSafeError::InvalidInput("TypeSafe model is empty".into())
        );
        validate_entry(&state)?;
        ensure!(
            !questions.is_empty(),
            TypeSafeError::InvalidInput("TypeSafe questions are empty".into())
        );
        for question in questions.values() {
            question.validate()?;
        }
        let body = serde_json::json!({"model": model, "state": state, "questions": questions});
        let response: ApiResponse<Evaluation> = self
            .request(Method::POST, "systemone", Some(body), options)
            .await?;
        ensure!(
            response.data.answers.len() == questions.len(),
            TypeSafeError::Response("answer count does not match questions".into())
        );
        for (id, question) in &questions {
            ensure!(
                matches!(
                    (question, response.data.answers.get(id)),
                    (Question::Noul { .. }, Some(Answer::Noul { .. }))
                        | (Question::Choice { .. }, Some(Answer::Choice { .. }))
                        | (Question::Score { .. }, Some(Answer::Score { .. }))
                ),
                TypeSafeError::Response(format!("missing or mismatched answer for {id}"))
            );
        }
        Ok(response)
    }

    pub async fn list_models(&self) -> Result<Vec<ModelCard>> {
        Ok(self
            .list_models_with_options(RequestOptions::default())
            .await?
            .data)
    }

    pub async fn list_models_with_options(
        &self,
        options: RequestOptions,
    ) -> Result<ApiResponse<Vec<ModelCard>>> {
        #[derive(Deserialize)]
        struct Models {
            models: Vec<ModelCard>,
        }
        let response: ApiResponse<Models> =
            self.request(Method::GET, "models", None, options).await?;
        Ok(response.map(|wire| wire.models))
    }

    fn emit(&self, event: RequestEvent) {
        if let Some(observer) = &self.observer {
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                observer.on_event(&event)
            }));
        }
    }

    async fn request<T: DeserializeOwned>(
        &self,
        method: Method,
        path: &str,
        body: Option<Value>,
        options: RequestOptions,
    ) -> Result<ApiResponse<T>> {
        let timeout = options.timeout.unwrap_or(self.timeout);
        validate_timeout(timeout)?;
        let retry = options.retry.as_ref().unwrap_or(&self.retry);
        retry.validate()?;
        let mut headers = self.headers.clone();
        headers.extend(options.headers);
        headers.insert(AUTHORIZATION, self.authorization.clone());
        headers.insert(ACCEPT, HeaderValue::from_static("application/json"));
        headers.insert(
            "x-typesafe-sdk",
            HeaderValue::from_static(concat!("goose-sdk/", env!("CARGO_PKG_VERSION"))),
        );
        headers.remove("x-typesafe-retry-count");
        headers.remove(CONTENT_TYPE);
        if body.is_some() {
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        }
        let url = self.base_url.join(path)?;
        for attempt in 0..=retry.max_retries {
            let mut attempt_headers = headers.clone();
            if attempt > 0 {
                attempt_headers.insert(
                    "x-typesafe-retry-count",
                    HeaderValue::from_str(&attempt.to_string())?,
                );
            }
            let mut request = self
                .http
                .request(method.clone(), url.clone())
                .headers(attempt_headers);
            if let Some(body) = &body {
                request = request.json(body);
            }
            let started = Instant::now();
            // Body delivery is part of the attempt deadline and retry policy.
            let result = tokio::time::timeout(timeout, async {
                let response = request.send().await?;
                let status = response.status();
                let headers = response.headers().clone();
                let bytes = response.bytes().await?;
                Ok::<_, reqwest::Error>((status, headers, bytes.to_vec()))
            })
            .await;
            let result = match result {
                Ok(Ok(response)) => Ok(response),
                Ok(Err(error)) if error.is_timeout() => Err(TypeSafeError::Timeout(timeout)),
                Ok(Err(error)) => Err(TypeSafeError::Connection(error)),
                Err(_) => Err(TypeSafeError::Timeout(timeout)),
            };
            self.emit(RequestEvent::AttemptFinished {
                path: path.into(),
                attempt,
                elapsed: started.elapsed(),
                status: result.as_ref().ok().map(|(status, _, _)| *status),
                request_id: result
                    .as_ref()
                    .ok()
                    .and_then(|(_, headers, _)| request_id(headers)),
                failed: result
                    .as_ref()
                    .map_or(true, |(status, _, _)| !status.is_success()),
            });
            let delay = match result {
                Ok((status, headers, bytes)) => {
                    if status.is_success() {
                        let data = serde_json::from_slice(&bytes).map_err(TypeSafeError::Decode)?;
                        return Ok(ApiResponse {
                            data,
                            status,
                            request_id: request_id(&headers),
                            headers,
                            raw_body: bytes,
                        });
                    }
                    if attempt == retry.max_retries || !retry.statuses.contains(&status.as_u16()) {
                        return Err(TypeSafeError::Api(Box::new(ApiError::new(
                            status, headers, &bytes,
                        )))
                        .into());
                    }
                    retry.delay(attempt, Some(&headers))
                }
                Err(error) => {
                    let retryable = match &error {
                        TypeSafeError::Timeout(_) => retry.timeouts,
                        TypeSafeError::Connection(error) => {
                            retry.connection_errors && !error.is_builder()
                        }
                        _ => false,
                    };
                    if attempt == retry.max_retries || !retryable {
                        return Err(error.into());
                    }
                    retry.delay(attempt, None)
                }
            };
            self.emit(RequestEvent::Retrying {
                path: path.into(),
                attempt: attempt + 1,
                delay,
            });
            tokio::time::sleep(delay).await;
        }
        unreachable!("final attempt returns a response or error")
    }
}

fn validate_timeout(timeout: Duration) -> Result<()> {
    ensure!(
        !timeout.is_zero(),
        TypeSafeError::InvalidInput("TypeSafe timeout must be positive".into())
    );
    Ok(())
}

pub(crate) fn request_id(headers: &HeaderMap) -> Option<String> {
    headers
        .get("x-typesafe-request-id")
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned)
}
