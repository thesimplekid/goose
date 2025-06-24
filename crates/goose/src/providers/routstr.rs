use anyhow::Result;
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use std::time::Duration;

use super::base::{ConfigKey, Provider, ProviderMetadata, ProviderUsage, Usage};
use super::errors::ProviderError;
use super::formats::openai::{create_request, get_usage, response_to_message};
use crate::message::Message;
use crate::model::ModelConfig;
use crate::providers::utils::{
    emit_debug_trace, get_model, handle_provider_response, is_anthropic_model,
    update_request_for_anthropic, ImageFormat, ProviderResponseType,
};
use mcp_core::tool::Tool;

pub const ROUTSTR_HOST: &str = "https://api.routstr.com";
pub const ROUTSTR_DEFAULT_MODEL: &str = "meta-llama/llama-3.2-1b-instruct";
pub const ROUTSTR_KNOWN_MODELS: &[&str] = &[
    "meta-llama/llama-3.2-1b-instruct",
    "deepseek/deepseek-r1-0528-qwen3-8b",
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-sonnet-4",
];
pub const ROUTSTR_DOC_URL: &str = "https://routstr.com/docs";
pub const ROUTSTR_DEFAULT_MINT_URL: &str = "https://mint.minibits.cash/Bitcoin";
pub const ROUTSTR_DEFAULT_CURRENCY_UNIT: &str = "sat";

/// Pricing information for a model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPricing {
    /// Price per input token (prompt)
    pub prompt: f64,
    /// Price per output token (completion)
    pub completion: f64,
}

/// Individual model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model identifier (e.g., "gpt-4")
    pub id: String,
    /// Object type, should be "model"
    pub object: String,
    /// Unix timestamp of when the model was created
    pub created: i64,
    /// Organization that owns the model (e.g., "openai", "anthropic")
    pub owned_by: String,
    /// Permission information (typically empty array)
    pub permission: Vec<Value>,
    /// Pricing information for this model
    pub pricing: ModelPricing,
    /// Maximum context length supported by the model
    pub context_length: u32,
}

/// Response structure for the /v1/models endpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsResponse {
    /// Object type, should be "list"
    pub object: String,
    /// Array of model information
    pub data: Vec<ModelInfo>,
}

#[derive(Debug, serde::Serialize, Clone)]
pub struct RoutstrProvider {
    #[serde(skip)]
    client: Arc<Client>,
    host: String,
    model: ModelConfig,
    api_key: String,
}

impl Default for RoutstrProvider {
    fn default() -> Self {
        let model = ModelConfig::new(RoutstrProvider::metadata().default_model).with_toolshim(true);
        // For the default implementation, we'll create a provider without pricing information
        // The pricing will be fetched lazily when needed
        Self::from_env(model).expect("Failed to initialize Routstr provider")
    }
}

impl RoutstrProvider {
    pub fn from_env(model: ModelConfig) -> Result<Self> {
        let config = crate::config::Config::global();
        let host = config
            .get_param("ROUTSTR_HOST")
            .unwrap_or_else(|_| ROUTSTR_HOST.to_string());

        let client = Client::builder()
            .timeout(Duration::from_secs(600))
            .build()?;

        let current_token: String = config.get_secret("ROUTSTR_API_KEY")?;

        let provider = Self {
            client: Arc::new(client),
            host,
            model,
            api_key: current_token,
        };

        Ok(provider)
    }

    async fn post(&self, payload: Value) -> Result<Value, ProviderError> {
        println!("{}", payload);
        let base_url = url::Url::parse(&self.host)
            .map_err(|e| ProviderError::RequestFailed(format!("Invalid base URL: {e}")))?;
        let url = base_url.join("v1/chat/completions").map_err(|e| {
            ProviderError::RequestFailed(format!("Failed to construct endpoint URL: {e}"))
        })?;

        // This will check balance and get token (get_auth_token includes balance check)
        let auth_token = &self.api_key;

        let response = self
            .client
            .post(url)
            .header("Authorization", format!("Bearer {auth_token}"))
            .json(&payload)
            .send()
            .await?;

        handle_provider_response(response, ProviderResponseType::OpenAI).await
    }

    /// Get models
    async fn get_models_info(&self) -> Result<ModelsResponse, ProviderError> {
        let base_url = url::Url::parse(&self.host)
            .map_err(|e| ProviderError::RequestFailed(format!("Invalid base URL: {e}")))?;
        let url = base_url.join("/v1/models").map_err(|e| {
            ProviderError::RequestFailed(format!("Failed to construct endpoint URL: {e}"))
        })?;

        let client = Client::builder()
            .timeout(Duration::from_secs(600))
            .build()?;

        let response = client
            .get(url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await?;

        let models_response: ModelsResponse = response.json().await.map_err(|e| {
            ProviderError::RequestFailed(format!("Failed to parse models response: {e}"))
        })?;

        Ok(models_response)
    }
}

#[async_trait]
impl Provider for RoutstrProvider {
    fn metadata() -> ProviderMetadata {
        ProviderMetadata::new(
            "routstr",
            "Routstr",
            "LLM provider with CDK wallet payment integration",
            ROUTSTR_DEFAULT_MODEL,
            ROUTSTR_KNOWN_MODELS.to_vec(),
            ROUTSTR_DOC_URL,
            vec![
                ConfigKey::new("ROUTSTR_HOST", true, false, Some(ROUTSTR_HOST)),
                ConfigKey::new(
                    "ROUTSTR_BASE_PATH",
                    true,
                    false,
                    Some("v1/chat/completions"),
                ),
                ConfigKey::new("OPENAI_TIMEOUT", false, false, Some("600")),
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
        // Create request with provided tools
        let mut payload =
            create_request(&self.model, system, messages, tools, &ImageFormat::OpenAi)?;

        // Apply anthropic-specific modifications if needed
        if is_anthropic_model(&self.model.model_name) {
            payload = update_request_for_anthropic(&payload);
        }

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

    fn supports_embeddings(&self) -> bool {
        false
    }

    async fn fetch_supported_models_async(&self) -> Result<Option<Vec<String>>, ProviderError> {
        if let Ok(models) = self.get_models_info().await {
            let model_ids = models.data.into_iter().map(|m| m.id.clone()).collect();
            Ok(Some(model_ids))
        } else {
            Ok(None)
        }
    }
}
