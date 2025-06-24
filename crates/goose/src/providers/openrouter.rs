use anyhow::{Error, Result};
use async_trait::async_trait;
use reqwest::Client;
use serde_json::Value;
use std::time::Duration;

use super::base::{ConfigKey, Provider, ProviderMetadata, ProviderUsage, Usage};
use super::errors::ProviderError;
use crate::message::Message;
use crate::model::ModelConfig;
use crate::providers::formats::openai::{create_request, get_usage, response_to_message};
use crate::providers::utils::{
    emit_debug_trace, get_model, handle_provider_response, is_anthropic_model, is_google_model,
    update_request_for_anthropic, ProviderResponseType,
};
use mcp_core::tool::Tool;
use url::Url;

pub const OPENROUTER_DEFAULT_MODEL: &str = "anthropic/claude-3.5-sonnet";
pub const OPENROUTER_MODEL_PREFIX_ANTHROPIC: &str = "anthropic";

// OpenRouter can run many models, we suggest the default
pub const OPENROUTER_KNOWN_MODELS: &[&str] = &[
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3.7-sonnet",
    "anthropic/claude-sonnet-4",
    "google/gemini-2.5-pro",
    "deepseek/deepseek-r1-0528",
];
pub const OPENROUTER_DOC_URL: &str = "https://openrouter.ai/models";

#[derive(serde::Serialize)]
pub struct OpenRouterProvider {
    #[serde(skip)]
    client: Client,
    host: String,
    api_key: String,
    model: ModelConfig,
}

impl Default for OpenRouterProvider {
    fn default() -> Self {
        let model = ModelConfig::new(OpenRouterProvider::metadata().default_model);
        OpenRouterProvider::from_env(model).expect("Failed to initialize OpenRouter provider")
    }
}

impl OpenRouterProvider {
    pub fn from_env(model: ModelConfig) -> Result<Self> {
        let config = crate::config::Config::global();
        let api_key: String = config.get_secret("OPENROUTER_API_KEY")?;
        let host: String = config
            .get_param("OPENROUTER_HOST")
            .unwrap_or_else(|_| "https://openrouter.ai".to_string());

        let client = Client::builder()
            .timeout(Duration::from_secs(600))
            .build()?;

        Ok(Self {
            client,
            host,
            api_key,
            model,
        })
    }

    async fn post(&self, payload: Value) -> Result<Value, ProviderError> {
        println!("{payload}");
        let base_url = Url::parse(&self.host)
            .map_err(|e| ProviderError::RequestFailed(format!("Invalid base URL: {e}")))?;
        let url = base_url.join("api/v1/chat/completions").map_err(|e| {
            ProviderError::RequestFailed(format!("Failed to construct endpoint URL: {e}"))
        })?;

        let response = self
            .client
            .post(url)
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("HTTP-Referer", "https://block.github.io/goose")
            .header("X-Title", "Goose")
            .json(&payload)
            .send()
            .await?;

        // Determine the provider type based on the model
        let provider_type = if is_google_model(&payload) {
            ProviderResponseType::Google
        } else {
            ProviderResponseType::OpenAI
        };

        // Handle response based on provider type
        let response_value = handle_provider_response(response, provider_type).await?;

        // OpenRouter can return errors in 200 OK responses, so we have to check for errors explicitly
        // https://openrouter.ai/docs/api-reference/errors
        if let Some(error_obj) = response_value.get("error") {
            // If there's an error object, extract the error message and code
            let error_message = error_obj
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("Unknown OpenRouter error");

            let error_code = error_obj.get("code").and_then(|c| c.as_u64()).unwrap_or(0);

            // Check for context length errors in the error message
            if error_code == 400 && error_message.contains("maximum context length") {
                return Err(ProviderError::ContextLengthExceeded(
                    error_message.to_string(),
                ));
            }

            // Return appropriate error based on the OpenRouter error code
            match error_code {
                401 | 403 => return Err(ProviderError::Authentication(error_message.to_string())),
                429 => return Err(ProviderError::RateLimitExceeded(error_message.to_string())),
                500 | 503 => return Err(ProviderError::ServerError(error_message.to_string())),
                _ => return Err(ProviderError::RequestFailed(error_message.to_string())),
            }
        }

        Ok(response_value)
    }
}

fn create_request_based_on_model(
    model_config: &ModelConfig,
    system: &str,
    messages: &[Message],
    tools: &[Tool],
) -> anyhow::Result<Value, Error> {
    let mut payload = create_request(
        model_config,
        system,
        messages,
        tools,
        &super::utils::ImageFormat::OpenAi,
    )?;

    // Apply anthropic-specific modifications if needed
    if is_anthropic_model(&model_config.model_name) {
        payload = update_request_for_anthropic(&payload);
    }

    Ok(payload)
}

#[async_trait]
impl Provider for OpenRouterProvider {
    fn metadata() -> ProviderMetadata {
        ProviderMetadata::new(
            "openrouter",
            "OpenRouter",
            "Router for many model providers",
            OPENROUTER_DEFAULT_MODEL,
            OPENROUTER_KNOWN_MODELS.to_vec(),
            OPENROUTER_DOC_URL,
            vec![
                ConfigKey::new("OPENROUTER_API_KEY", true, true, None),
                ConfigKey::new(
                    "OPENROUTER_HOST",
                    false,
                    false,
                    Some("https://openrouter.ai"),
                ),
            ],
        )
    }

    fn get_model_config(&self) -> ModelConfig {
        self.model.clone()
    }

    #[tracing::instrument(
        skip(self, system, messages, tools),
        fields(model_config, input, output, input_tokens, output_tokens, total_tokens)
    )]
    async fn complete(
        &self,
        system: &str,
        messages: &[Message],
        tools: &[Tool],
    ) -> Result<(Message, ProviderUsage), ProviderError> {
        // Create the base payload
        let payload = create_request_based_on_model(&self.model, system, messages, tools)?;

        // Make request
        let response = self.post(payload.clone()).await?;

        // Parse response
        let message = response_to_message(response.clone())?;
        let usage = match get_usage(&response) {
            Ok(usage) => usage,
            Err(ProviderError::UsageError(e)) => {
                tracing::debug!("Failed to get usage data: {}", e);
                Usage::default()
            }
            Err(e) => return Err(e),
        };
        let model = get_model(&response);
        emit_debug_trace(&self.model, &payload, &response, &usage);
        Ok((message, ProviderUsage::new(model, usage)))
    }
}
