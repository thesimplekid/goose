#![cfg(feature = "typesafe")]

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};

use goose_sdk::typesafe::{Answer, Question, TypeSafeClient, DEFAULT_MODEL};
use serde_json::{json, Value};
use wiremock::{
    matchers::{body_json, header, method, path},
    Mock, MockServer, ResponseTemplate,
};

fn questions() -> HashMap<String, Question> {
    serde_json::from_value(json!({
        "urgent": {"type": "noul", "instructions": "Is this urgent?"},
        "team": {"type": "choice", "instructions": {"task": "Pick a team"},
            "criteria": {"billing": null, "support": "Technical problems"}},
        "severity": {"type": "score", "instructions": ["Rate severity"],
            "criteria": ["Low", "High"]}
    }))
    .unwrap()
}

fn response() -> Value {
    json!({
        "model": "jev-1.13.0",
        "answers": {
            "urgent": {"type": "noul", "noul": 0.9},
            "team": {"type": "choice", "choice": "billing",
                "probabilities": {"billing": 0.8, "support": 0.2}, "confidence": 0.6},
            "severity": {"type": "score", "score": 0.7,
                "legend": {"0": "Low", "1": "High"},
                "probabilities": {"0": 0.3, "1": 0.7}, "confidence": 0.4}
        },
        "usage": {"input_tokens": 120, "output_tokens": 24}
    })
}

#[tokio::test]
async fn evaluates_all_question_types_and_preserves_results() {
    let server = MockServer::start().await;
    let questions = questions();
    Mock::given(method("POST"))
        .and(path("/proxy/v1/systemone"))
        .and(header("authorization", "Bearer test-key"))
        .and(body_json(json!({"model": DEFAULT_MODEL, "state": {"ticket": "Refund ASAP"}, "questions": questions})))
        .respond_with(ResponseTemplate::new(200).set_body_json(response()))
        .expect(1).mount(&server).await;
    let client =
        TypeSafeClient::with_base_url("test-key".into(), &format!("{}/proxy/v1/", server.uri()))
            .unwrap();
    let result = client
        .evaluate(DEFAULT_MODEL, json!({"ticket": "Refund ASAP"}), questions)
        .await
        .unwrap();
    assert_eq!(result.model, "jev-1.13.0");
    assert_eq!(result.usage.input_tokens, 120);
    assert_eq!(result.usage.output_tokens, 24);
    assert!(matches!(result.answers["urgent"], Answer::Noul { noul } if noul == 0.9));
    match &result.answers["team"] {
        Answer::Choice {
            choice,
            probabilities,
            confidence,
        } => {
            assert_eq!(choice, "billing");
            assert_eq!(probabilities["support"], 0.2);
            assert_eq!(*confidence, 0.6);
        }
        answer => panic!("unexpected answer: {answer:?}"),
    }
    match &result.answers["severity"] {
        Answer::Score {
            score,
            legend,
            probabilities,
            confidence,
        } => {
            assert_eq!(*score, 0.7);
            assert_eq!(legend["1"], "High");
            assert_eq!(probabilities["0"], 0.3);
            assert_eq!(*confidence, 0.4);
        }
        answer => panic!("unexpected answer: {answer:?}"),
    }
}

#[tokio::test]
async fn retries_rate_limits_and_overload_but_stops_after_three_attempts() {
    for status in [408, 429, 500, 502, 503, 529] {
        for recover in [false, true] {
            let server = MockServer::start().await;
            let attempts = Arc::new(AtomicUsize::new(0));
            let observed = attempts.clone();
            Mock::given(path("/v1/systemone"))
                .respond_with(move |_: &wiremock::Request| {
                    let attempt = observed.fetch_add(1, Ordering::SeqCst);
                    if recover && attempt == 1 {
                        ResponseTemplate::new(200).set_body_json(response())
                    } else {
                        ResponseTemplate::new(status).insert_header("retry-after", "0")
                    }
                })
                .mount(&server)
                .await;
            let client = TypeSafeClient::with_base_url("key".into(), &server.uri()).unwrap();
            let result = client
                .evaluate(DEFAULT_MODEL, json!("ticket"), questions())
                .await;
            assert_eq!(result.is_ok(), recover);
            assert_eq!(attempts.load(Ordering::SeqCst), if recover { 2 } else { 3 });
        }
    }
}

#[tokio::test]
async fn authentication_and_validation_failures_are_not_retried() {
    for status in [401, 422] {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(status).set_body_string("API failure"))
            .expect(1)
            .mount(&server)
            .await;
        let client = TypeSafeClient::with_base_url("key".into(), &server.uri()).unwrap();
        let error = client
            .evaluate(DEFAULT_MODEL, json!([]), questions())
            .await
            .unwrap_err();
        assert!(error.to_string().contains(&status.to_string()));
        assert!(error.to_string().contains("API failure"));
    }
}

#[tokio::test]
async fn rejects_invalid_state_and_rubric_before_sending() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200))
        .expect(0)
        .mount(&server)
        .await;
    let client = TypeSafeClient::with_base_url("key".into(), &server.uri()).unwrap();
    assert!(client
        .evaluate(DEFAULT_MODEL, json!(true), questions())
        .await
        .is_err());
    let invalid = HashMap::from([(
        "score".into(),
        Question::Score {
            instructions: json!("Rate"),
            criteria: vec!["Only one".into()],
        },
    )]);
    assert!(client
        .evaluate(DEFAULT_MODEL, json!("ticket"), invalid)
        .await
        .is_err());
}

#[tokio::test]
async fn rejects_missing_mismatched_and_malformed_answers() {
    let mut missing = response();
    missing["answers"].as_object_mut().unwrap().remove("urgent");
    let mut mismatched = response();
    mismatched["answers"]["team"] = json!({"type": "noul", "noul": 0.5});
    for body in [missing, mismatched, json!({"model": "jev-latest"})] {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(body))
            .mount(&server)
            .await;
        let client = TypeSafeClient::with_base_url("key".into(), &server.uri()).unwrap();
        assert!(client
            .evaluate(DEFAULT_MODEL, json!("ticket"), questions())
            .await
            .is_err());
    }
}

#[tokio::test]
async fn cancellation_during_backoff_prevents_retry() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(429).insert_header("retry-after", "30"))
        .expect(1)
        .mount(&server)
        .await;
    let client = TypeSafeClient::with_base_url("key".into(), &server.uri()).unwrap();
    let task = tokio::spawn(async move {
        client
            .evaluate(DEFAULT_MODEL, json!("ticket"), questions())
            .await
    });
    tokio::time::timeout(Duration::from_secs(5), async {
        while server.received_requests().await.unwrap().is_empty() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
}

#[cfg(feature = "uniffi")]
#[tokio::test]
async fn bindings_return_typed_answers() {
    use goose_sdk::bindings::{typesafe_client, TypeSafeAnswer, TypeSafeQuestion};
    let server = MockServer::start().await;
    let mut body = response();
    body["answers"]
        .as_object_mut()
        .unwrap()
        .retain(|id, _| id == "urgent");
    Mock::given(path("/v1/systemone"))
        .and(body_json(
            json!({"model": DEFAULT_MODEL, "state": "ticket", "questions": {
                "urgent": {"type": "noul", "instructions": "Urgent?"}
            }}),
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .expect(1)
        .mount(&server)
        .await;
    let client = typesafe_client("key".into(), Some(server.uri())).unwrap();
    let result = client
        .evaluate(
            DEFAULT_MODEL.into(),
            json!("ticket").to_string(),
            HashMap::from([(
                "urgent".into(),
                TypeSafeQuestion::Noul {
                    instructions: "Urgent?".into(),
                    criteria: HashMap::new(),
                },
            )]),
        )
        .await
        .unwrap();
    assert!(matches!(result.answers["urgent"], TypeSafeAnswer::Noul { noul } if noul == 0.9));
    assert_eq!(result.input_tokens, 120);
}

#[tokio::test]
async fn null_entries_and_structured_criteria_round_trip() {
    let server = MockServer::start().await;
    let questions: HashMap<String, Question> = serde_json::from_value(json!({
        "urgent": {"type":"noul", "criteria": null},
        "team": {"type":"choice", "criteria":{"billing":{"rules":["refund"]}, "support":null}},
        "severity": {"type":"score", "criteria":[null, {"description":"High"}]}
    }))
    .unwrap();
    let mut body = response();
    body["answers"]["severity"]["legend"] = json!({"0":null, "1":{"description":"High"}});
    Mock::given(body_json(
        json!({"model":DEFAULT_MODEL, "state":null, "questions":questions}),
    ))
    .respond_with(ResponseTemplate::new(200).set_body_json(body))
    .expect(1)
    .mount(&server)
    .await;
    let client = TypeSafeClient::with_base_url("key".into(), &server.uri()).unwrap();
    let result = client.system_one(Value::Null, questions).await.unwrap();
    let Answer::Score { legend, .. } = &result.answers["severity"] else {
        panic!("not a score")
    };
    assert_eq!(legend["0"], Value::Null);
    assert_eq!(legend["1"], json!({"description":"High"}));
}

#[tokio::test]
async fn model_listing_preserves_metadata_and_uses_custom_transport() {
    use goose_sdk::typesafe::RequestOptions;
    use reqwest::header::{HeaderMap, HeaderValue};
    let server = MockServer::start().await;
    let body = json!({"models":[{"name":"jev-latest", "description":"Stable", "release_date":"2026-09-18"}]});
    Mock::given(method("GET"))
        .and(path("/proxy/v1/models"))
        .and(header("x-custom-transport", "yes"))
        .and(header("x-project", "override"))
        .and(header("authorization", "Bearer key"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(body.clone())
                .insert_header("x-typesafe-request-id", "req-models"),
        )
        .expect(1)
        .mount(&server)
        .await;
    let transport_headers = HeaderMap::from_iter([(
        "x-custom-transport".parse().unwrap(),
        HeaderValue::from_static("yes"),
    )]);
    let http = reqwest::Client::builder()
        .default_headers(transport_headers)
        .build()
        .unwrap();
    let client = TypeSafeClient::builder()
        .api_key("key")
        .base_url(format!("{}/proxy", server.uri()))
        .headers(HeaderMap::from_iter([(
            "x-project".parse().unwrap(),
            HeaderValue::from_static("default"),
        )]))
        .http_client(http)
        .build()
        .unwrap();
    let result = client
        .list_models_with_options(RequestOptions {
            headers: HeaderMap::from_iter([
                (
                    "x-project".parse().unwrap(),
                    HeaderValue::from_static("override"),
                ),
                (
                    "authorization".parse().unwrap(),
                    HeaderValue::from_static("must-not-override"),
                ),
            ]),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(result.data[0].name, "jev-latest");
    assert_eq!(result.data[0].release_date, "2026-09-18");
    assert_eq!(result.request_id.as_deref(), Some("req-models"));
    assert_eq!(result.status, reqwest::StatusCode::OK);
    assert_eq!(
        serde_json::from_slice::<Value>(&result.raw_body).unwrap(),
        body
    );
}

#[tokio::test]
async fn errors_expose_kind_headers_body_and_retry_delay() {
    use goose_sdk::typesafe::{ApiErrorKind, RequestOptions, RetryPolicy, TypeSafeError};
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(429)
                .set_body_json(json!({"error":{"message":"Slow down"}}))
                .insert_header("x-typesafe-request-id", "req-failed")
                .insert_header("retry-after-ms", "1250"),
        )
        .expect(1)
        .mount(&server)
        .await;
    let client = TypeSafeClient::with_base_url("key".into(), &server.uri()).unwrap();
    let error = client
        .evaluate_with_options(
            None,
            Value::Null,
            questions(),
            RequestOptions {
                retry: Some(RetryPolicy {
                    max_retries: 0,
                    ..Default::default()
                }),
                ..Default::default()
            },
        )
        .await
        .unwrap_err();
    let TypeSafeError::Api(error) = error.downcast_ref::<TypeSafeError>().unwrap() else {
        panic!("not an API error")
    };
    assert_eq!(error.kind, ApiErrorKind::RateLimited);
    assert_eq!(error.status, reqwest::StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(error.request_id.as_deref(), Some("req-failed"));
    assert_eq!(error.retry_after, Some(Duration::from_millis(1250)));
    assert_eq!(error.message, "Slow down");
    assert_eq!(error.body, json!({"error":{"message":"Slow down"}}));
    assert_eq!(error.headers["retry-after-ms"], "1250");
}

#[test]
fn retry_headers_support_milliseconds_fractions_and_dates() {
    use goose_sdk::typesafe::parse_retry_after;
    use reqwest::header::{HeaderMap, HeaderValue};
    let now = std::time::UNIX_EPOCH + Duration::from_secs(1_000_000);
    let mut headers = HeaderMap::new();
    headers.insert("retry-after", HeaderValue::from_static("1.25"));
    assert_eq!(
        parse_retry_after(&headers, now),
        Some(Duration::from_millis(1250))
    );
    headers.insert("retry-after-ms", HeaderValue::from_static("100"));
    assert_eq!(
        parse_retry_after(&headers, now),
        Some(Duration::from_millis(100))
    );
    headers.remove("retry-after-ms");
    headers.insert(
        "retry-after",
        HeaderValue::from_str(&httpdate::fmt_http_date(now + Duration::from_secs(2))).unwrap(),
    );
    assert_eq!(
        parse_retry_after(&headers, now),
        Some(Duration::from_secs(2))
    );
    for invalid in ["-2", "NaN", "inf", "garbage"] {
        headers.insert("retry-after", HeaderValue::from_str(invalid).unwrap());
        assert_eq!(parse_retry_after(&headers, now), None);
    }
}

#[derive(Default)]
struct Events(std::sync::Mutex<Vec<goose_sdk::typesafe::RequestEvent>>);
impl goose_sdk::typesafe::RequestObserver for Events {
    fn on_event(&self, event: &goose_sdk::typesafe::RequestEvent) {
        self.0.lock().unwrap().push(event.clone());
    }
}

#[tokio::test]
async fn timeout_retry_overrides_and_observation_work_together() {
    use goose_sdk::typesafe::{RequestEvent, RequestOptions, RetryPolicy};
    let server = MockServer::start().await;
    let attempts = Arc::new(AtomicUsize::new(0));
    let seen = attempts.clone();
    Mock::given(method("POST"))
        .respond_with(move |_: &wiremock::Request| {
            if seen.fetch_add(1, Ordering::SeqCst) == 0 {
                ResponseTemplate::new(200)
                    .set_delay(Duration::from_secs(1))
                    .set_body_json(response())
            } else {
                ResponseTemplate::new(200).set_body_json(response())
            }
        })
        .expect(2)
        .mount(&server)
        .await;
    let events = Arc::new(Events::default());
    let client = TypeSafeClient::builder()
        .api_key("key")
        .base_url(server.uri())
        .retry(RetryPolicy {
            max_retries: 0,
            ..Default::default()
        })
        .observer(events.clone())
        .build()
        .unwrap();
    client
        .evaluate_with_options(
            None,
            Value::Null,
            questions(),
            RequestOptions {
                timeout: Some(Duration::from_millis(100)),
                retry: Some(RetryPolicy {
                    max_retries: 1,
                    backoff_initial: Duration::ZERO,
                    ..Default::default()
                }),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    let events = events.0.lock().unwrap();
    assert!(matches!(
        &events[0],
        RequestEvent::AttemptFinished {
            failed: true,
            status: None,
            ..
        }
    ));
    assert!(matches!(
        &events[1],
        RequestEvent::Retrying { attempt: 1, .. }
    ));
    assert!(matches!(
        &events[2],
        RequestEvent::AttemptFinished { failed: false, .. }
    ));
}

#[tokio::test]
async fn oversized_retry_after_falls_back_to_configured_backoff() {
    use goose_sdk::typesafe::{RequestEvent, RetryPolicy};
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(503).insert_header("retry-after", "120"))
        .expect(2)
        .mount(&server)
        .await;
    let events = Arc::new(Events::default());
    let client = TypeSafeClient::builder()
        .api_key("key")
        .base_url(server.uri())
        .observer(events.clone())
        .retry(RetryPolicy {
            max_retries: 1,
            backoff_initial: Duration::ZERO,
            ..Default::default()
        })
        .build()
        .unwrap();
    assert!(client.system_one(Value::Null, questions()).await.is_err());
    assert!(events
        .0
        .lock()
        .unwrap()
        .iter()
        .any(|event| matches!(event, RequestEvent::Retrying {delay, ..} if delay.is_zero())));
}

#[tokio::test]
async fn connection_failures_are_typed_and_retried() {
    use goose_sdk::typesafe::{RequestEvent, RetryPolicy, TypeSafeError};
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    drop(listener);
    let events = Arc::new(Events::default());
    let client = TypeSafeClient::builder()
        .api_key("key")
        .base_url(format!("http://{address}"))
        .observer(events.clone())
        .retry(RetryPolicy {
            backoff_initial: Duration::ZERO,
            ..Default::default()
        })
        .build()
        .unwrap();
    let error = client.list_models().await.unwrap_err();
    assert!(matches!(
        error.downcast_ref::<TypeSafeError>(),
        Some(TypeSafeError::Connection(_))
    ));
    assert_eq!(
        events
            .0
            .lock()
            .unwrap()
            .iter()
            .filter(|event| matches!(event, RequestEvent::AttemptFinished { .. }))
            .count(),
        3
    );
}

#[tokio::test]
async fn helpers_and_application_answer_types_use_default_model() {
    use serde::Deserialize;
    #[derive(Debug, Deserialize, PartialEq)]
    #[serde(rename_all = "snake_case")]
    enum Team {
        Billing,
        Support,
    }
    #[derive(Deserialize)]
    struct TeamAnswer {
        choice: Team,
    }
    #[derive(Deserialize)]
    struct Answers {
        team: TeamAnswer,
    }
    let server = MockServer::start().await;
    let questions = HashMap::from([
        ("urgent".into(), Question::noul(Value::Null)),
        (
            "team".into(),
            Question::choice(
                "Pick a team",
                HashMap::from([
                    ("billing".into(), Value::Null),
                    ("support".into(), Value::Null),
                ]),
            ),
        ),
        (
            "severity".into(),
            Question::score(Value::Null, vec![json!("Low"), json!("High")]).unwrap(),
        ),
    ]);
    Mock::given(body_json(
        json!({"model":"jev-pinned", "state":null, "questions":questions}),
    ))
    .respond_with(ResponseTemplate::new(200).set_body_json(response()))
    .expect(1)
    .mount(&server)
    .await;
    let client = TypeSafeClient::builder()
        .api_key("key")
        .base_url(server.uri())
        .default_model("jev-pinned")
        .build()
        .unwrap();
    let result = client.system_one(Value::Null, questions).await.unwrap();
    assert_eq!(
        result.answers_as::<Answers>().unwrap().team.choice,
        Team::Billing
    );
}

#[test]
fn environment_defaults_and_explicit_overrides() {
    if std::env::var_os("GOOSE_TYPESAFE_ENV_TEST").is_some() {
        let client = TypeSafeClient::from_env().unwrap();
        assert_eq!(client.default_model(), "jev-from-env");
        let explicit = TypeSafeClient::builder()
            .api_key("explicit-key")
            .base_url("http://127.0.0.1:1")
            .default_model("jev-explicit")
            .build()
            .unwrap();
        assert_eq!(explicit.default_model(), "jev-explicit");
        assert!(TypeSafeClient::builder().api_key("").build().is_err());
        return;
    }
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "environment_defaults_and_explicit_overrides"])
        .env("GOOSE_TYPESAFE_ENV_TEST", "1")
        .env("TYPESAFE_API_KEY", "env-key")
        .env("TYPESAFE_BASE_URL", "http://127.0.0.1:2")
        .env("TYPESAFE_DEFAULT_MODEL", "jev-from-env")
        .status()
        .unwrap();
    assert!(status.success());
}

#[tokio::test]
async fn interrupted_response_body_is_retried() {
    use goose_sdk::typesafe::RetryPolicy;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        for attempt in 0..2 {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = Vec::new();
            while !request.ends_with(b"\r\n\r\n") {
                request.push(socket.read_u8().await.unwrap());
            }
            if attempt == 0 {
                socket
                    .write_all(
                        b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\nConnection: close\r\n\r\n{",
                    )
                    .await
                    .unwrap();
            } else {
                let body = r#"{"models":[]}"#;
                socket.write_all(format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).as_bytes()).await.unwrap();
            }
            socket.shutdown().await.unwrap();
        }
    });
    let client = TypeSafeClient::builder()
        .api_key("key")
        .base_url(format!("http://{address}"))
        .retry(RetryPolicy {
            backoff_initial: Duration::ZERO,
            ..Default::default()
        })
        .build()
        .unwrap();
    let result = tokio::time::timeout(Duration::from_secs(5), client.list_models())
        .await
        .unwrap()
        .unwrap();
    assert!(result.is_empty());
    server.await.unwrap();
}
