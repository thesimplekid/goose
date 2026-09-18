# goose-sdk

The bindings layer for goose, published as the goose Development Kit (GDK). It
houses the shared types used for both ACP and GDK access, and exposes a
cross-language version of the goose API.

With `--features uniffi` the crate compiles to native bindings for Python and
Kotlin (namespace `goose` / `io.github.aaif_goose`). The UniFFI surface lets
callers construct providers, stream provider completions, perform non-streaming
completion, and pass rich message/tool content across the FFI boundary.

```bash
just python   # build bindings + run examples/uniffi/provider.py
just kotlin   # build the Maven artifact + run examples/uniffi/kotlin
```

## TypeSafe / Jev evaluation

The `typesafe` feature provides `goose_sdk::typesafe::TypeSafeClient` for
[TypeSafe's evaluation API](https://docs.typesafe.ai/api). It is also included
in `uniffi`, exposing `typesafe_client` / `typesafeClient` to Python and Kotlin.
Jev evaluates state using typed `Noul`, `Choice`, and `Score` questions. Use it
alongside a chat provider to classify input, route work, or evaluate output.

Supply your API key explicitly or use `TypeSafeClient::from_env()` in Rust.
The Rust builder resolves explicit settings first, then `TYPESAFE_API_KEY`,
`TYPESAFE_BASE_URL`, and `TYPESAFE_DEFAULT_MODEL`, then SDK defaults. Empty
environment values are ignored. The default model is `jev-latest`; pass a
versioned model ID to pin evaluation behavior.
The result preserves the returned model ID, answers, probabilities, confidence,
score legends, and token usage.

Rust callers enable `features = ["typesafe"]` on their `goose-sdk` dependency
and pass `serde_json::Value` for state, instructions, and criteria descriptions.
Entries can be null, text, objects, or arrays. Missing instructions deserialize
as null; Noul criteria may be omitted or null. Score legends preserve structured
and null descriptions. See the
[Rust example](examples/typesafe.rs):

```bash
cargo run -p goose-sdk --features typesafe --example typesafe
```

Rust applications can use question helpers and the client's default model:

```rust
use std::collections::HashMap;
use goose_sdk::typesafe::{Question, TypeSafeClient};
use serde_json::{json, Value};

async fn example() -> anyhow::Result<()> {
    let client = TypeSafeClient::from_env()?;
    let result = client.system_one(
        json!({"ticket": "I was charged twice"}),
        HashMap::from([("team".into(), Question::choice(
            "Which team should handle this?",
            HashMap::from([
                ("billing".into(), json!({"description": "Payments and refunds"})),
                ("support".into(), Value::Null),
            ]),
        ))]),
    ).await?;
    Ok(())
}
```

`Question::noul` and `Question::score` construct the other question types.
`Evaluation::answers_as::<YourAnswers>()` decodes answers into your own
`Deserialize` struct, allowing named fields and enum choice labels. Unlike
TypeScript's mapped types, this mapping is checked when decoding the response.

Use `client.list_models().await?` for model discovery. For HTTP metadata, use
`evaluate_with_options` or `list_models_with_options`; both return `ApiResponse`
with parsed data, status, headers, request ID, and the raw body.

The builder accepts `timeout`, `retry`, `headers`, and a custom `reqwest::Client`
through `http_client` for proxy/TLS configuration. `RequestOptions` overrides
timeout, retry policy, and headers per call. A supplied retry policy replaces
the client policy; use Rust struct updates to retain the defaults. Authentication,
JSON content type, and SDK retry headers cannot be overridden by custom headers.

Errors use `anyhow::Result` and can be downcast to `TypeSafeError`. HTTP failures
carry an `ApiError` with a typed category, status, parsed body, headers, request
ID, and retry delay. Connection failures, timeouts, invalid inputs, and malformed
responses have distinct variants.

Python and Kotlin accept JSON-encoded state and typed questions with plain-text
instructions. An empty map omits optional Noul criteria. Choice criteria map
option names to optional descriptions; Score criteria contain at least two
ordered levels. After building the bindings, the
[Python example](examples/uniffi/typesafe.py) can be run with:

```bash
python crates/goose-sdk/examples/uniffi/typesafe.py
```

Kotlin, from a coroutine:

```kotlin
val client = typesafeClient(System.getenv("TYPESAFE_API_KEY"))
val result = client.evaluate(
    typesafeDefaultModel(),
    """{"ticket":"I was charged twice"}""",
    mapOf("team" to TypeSafeQuestion.Choice(
        instructions = "Which team should handle this?",
        criteria = mapOf("billing" to "Payments and refunds", "support" to null),
    )),
)
val team = result.answers.getValue("team") as TypeSafeAnswer.Choice
println("${team.choice}: ${team.confidence}")
```

For a proxy, use Rust's `with_base_url` or the bindings' optional `base_url` /
`baseUrl` argument. The URL can include a prefix and an optional trailing `/v1`.
By default, the client retries HTTP 408, 429, all 5xx responses, connection
failures (including interrupted response bodies), and timeouts twice. Backoff
starts at 500 ms, doubles up to 5 seconds, and subtracts up to 25% jitter.
`retry-after-ms` takes precedence over `Retry-After`, which accepts fractional
seconds and HTTP dates. Server delays up to 60 seconds are honored; larger
delays fall back to exponential backoff. Each attempt has a 10-second timeout,
including body delivery. There is no total retry deadline; Rust callers can
wrap the future in `tokio::time::timeout` to set one.
Dropping the Rust future or cancelling the binding call cancels pending work.
An optional Rust `RequestObserver` receives attempt metadata and retry events
for integration with application logging or tracing. It receives no keys,
headers, or payloads. There is no automatic logging, and these calls do not
emit the chat provider observability hooks.

The existing Python/Kotlin question constructors remain text-based. Their
score legends preserve text descriptions and JSON-encode structured or null
descriptions; the expanded configuration and model-discovery APIs are Rust APIs.

Rust agent applications can call the client from a custom `goose-agent`
`Operation` for required evaluations, or from an `AsyncTool` registered with
`ToolOperation` when the chat model should choose when to evaluate.

## Observability hooks

Register an `ObservabilityHook` to receive typed provider request lifecycle
events instead of parsing debug output. Hooks are opt-in: with no hook
registered nothing is emitted and the request path is unchanged.

Each request emits `onRequestStart`, then `onResponseStart` once the provider
response is available (the stream opens for streaming requests), then exactly
one `onRequestEnd` carrying the outcome (`Success`, or `Failure` with a typed
`GooseStreamError`), `durationMs`, and token `usage`. All three events share a
`requestId` so they can be correlated with application telemetry. A streaming
read that times out ends the trace, so continuing to read the stream afterwards
never produces a second `onRequestEnd`.

`clearObservabilityHook` also stops delivery for requests that are still in
flight, so no events reach a hook after it is cleared.

```kotlin
class TracingHook : ObservabilityHook {
    override fun onRequestStart(event: RequestStartEvent) {
        tracer.startSpan(event.requestId, event.provider, event.model)
    }

    override fun onResponseStart(event: ResponseStartEvent) {
        tracer.recordTimeToFirstByte(event.requestId, event.elapsedMs)
    }

    override fun onRequestEnd(event: RequestEndEvent) {
        tracer.finishSpan(event.requestId, event.outcome, event.durationMs, event.usage)
    }
}

setObservabilityHook(TracingHook(), capturePayloads = false)
```

Hooks are invoked synchronously on the calling thread and a throwing callback is
caught, so it cannot fail the request. Keep them fast: slow callbacks add
latency to the request they observe.

### Security guidance

`capturePayloads` defaults to `false` and payloads are omitted entirely in that
mode: `RequestStartEvent.payload` and `RequestEndEvent.responseJson` are null,
leaving only non-sensitive metadata (provider, model, latency, usage).

Enable `capturePayloads` only when you control the sink. It exposes the system
prompt, the full conversation, and tool schemas, which routinely contain
credentials, customer data, and other secrets. Apply your own redaction before
persisting or exporting these fields.

Response capture is asymmetric: `RequestEndEvent.responseJson` is populated for
`complete` calls but is always null for `stream` calls, because the streamed
response is delivered to the caller chunk by chunk and is never buffered by the
SDK. Assemble the streamed body from the chunks you already receive if you need
it.

## Python package

The PyPI package is published as `goose-sdk` and imports as `goose`.
Build a local wheel from the repository root with:

```bash
just --justfile crates/goose-sdk/justfile python-wheel
```

This regenerates the UniFFI Python bindings, copies the release native library
into the package, and writes the wheel to `crates/goose-sdk/python/dist/`.

## Maven package

The Maven Central artifact is published as `io.github.aaif-goose:gdk` and uses
the Rust crate version from `crates/goose-sdk/Cargo.toml`.

```bash
just --justfile crates/goose-sdk/justfile maven-package
```

This regenerates the UniFFI Kotlin bindings and packages them with the native
library in a JVM jar. CI builds the native libraries for supported platforms and
can optionally publish the combined artifact to Maven Central.
