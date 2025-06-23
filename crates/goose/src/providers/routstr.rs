use anyhow::Result;
use async_trait::async_trait;
use bip39::Mnemonic;
use cdk::nuts::CurrencyUnit;
use cdk::wallet::{SendOptions, Wallet};
use cdk::Amount;
use cdk_sqlite::WalletSqliteDatabase;
use home::home_dir;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;

use super::base::{ConfigKey, Provider, ProviderMetadata, ProviderUsage, Usage};
use super::errors::ProviderError;
use super::formats::openai::{create_request, get_usage, response_to_message};
use super::toolshim::{augment_message_with_tool_calls, RoutstrInterpreter};
use super::utils::{emit_debug_trace, get_model, handle_response_openai_compat, ImageFormat};
use crate::message::Message;
use crate::model::ModelConfig;
use crate::token_counter::TokenCounter;
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
    #[serde(skip)]
    wallet: Arc<Wallet>,
    prompt_cost: Option<f64>,
    completion_cost: Option<f64>,
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

        // Get mint URL and currency unit from config
        let mint_url = config
            .get_param("ROUTSTR_MINT_URL")
            .unwrap_or_else(|_| ROUTSTR_DEFAULT_MINT_URL.to_string());

        let currency_unit_str = config
            .get_param("ROUTSTR_CURRENCY_UNIT")
            .unwrap_or_else(|_| ROUTSTR_DEFAULT_CURRENCY_UNIT.to_string());

        let currency_unit = match currency_unit_str.to_lowercase().as_str() {
            "sat" => CurrencyUnit::Sat,
            _ => {
                return Err(anyhow::anyhow!(
                    "Invalid currency unit specified. Must be one of: Sat"
                ))
            }
        };

        let client = Client::builder()
            .timeout(Duration::from_secs(600))
            .build()?;

        // Initialize CDK wallet
        let work_dir = home_dir().unwrap().join(".cdk-cli");
        fs::create_dir_all(&work_dir)?;
        let cdk_wallet_path = work_dir.join("cdk-cli.sqlite");

        let wallet_db = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(WalletSqliteDatabase::new(&cdk_wallet_path))
        })?;

        let seed_path = work_dir.join("seed");

        let mnemonic = match fs::metadata(seed_path.clone()) {
            Ok(_) => {
                let contents = fs::read_to_string(seed_path.clone())?;
                Mnemonic::from_str(&contents)?
            }
            Err(_e) => {
                let mnemonic = Mnemonic::generate(12)?;
                tracing::info!("Creating new seed");

                fs::write(seed_path, mnemonic.to_string())?;

                mnemonic
            }
        };

        let seed = mnemonic.to_seed_normalized("");

        let wallet = Wallet::new(&mint_url, currency_unit, Arc::new(wallet_db), &seed, None)?;

        // Fetch model pricing information
        let (prompt_cost, completion_cost, max_context) = match tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(Self::get_models_info(&host, &wallet))
        }) {
            Ok(models_response) => {
                // Find the pricing for the current model
                if let Some(model_info) = models_response
                    .data
                    .iter()
                    .find(|m| m.id == model.model_name)
                {
                    println!(
                        "Found pricing for model {}: prompt=${}, completion=${}",
                        model.model_name, model_info.pricing.prompt, model_info.pricing.completion
                    );
                    (
                        Some(model_info.pricing.prompt),
                        Some(model_info.pricing.completion),
                        Some(model_info.context_length),
                    )
                } else {
                    tracing::warn!(
                        "No pricing found for model {}, using defaults",
                        model.model_name
                    );
                    (None, None, None)
                }
            }
            Err(e) => {
                eprintln!("Failed to fetch model pricing, using defaults: {}", e);
                (None, None, None)
            }
        };

        let mut model = model;

        if let Some(max_con) = max_context {
            model.context_limit = Some(max_con as usize);
        }

        let provider = Self {
            client: Arc::new(client),
            host,
            model,
            wallet: Arc::new(wallet),
            prompt_cost,
            completion_cost,
        };

        Ok(provider)
    }

    /// Ensures the wallet has sufficient balance, returning an error if it's too low
    pub async fn ensure_wallet_balance(&self) -> Result<(), ProviderError> {
        let balance = self.wallet.total_balance().await.map_err(|e| {
            ProviderError::Authentication(format!("Failed to get wallet balance: {e}"))
        })?;

        if balance < 50.into() {
            tracing::warn!(
                "Wallet balance below 100 sats. Current balance: {}",
                balance
            );
            return Err(ProviderError::Authentication(
                "Insufficient wallet balance. Please top up your wallet to continue.".to_string(),
            ));
        }

        Ok(())
    }

    async fn get_auth_token(&self, amount: u64) -> Result<String, ProviderError> {
        // Check balance before attempting to generate a token

        // Generate payment token using CDK wallet
        let prepare_send = self
            .wallet
            .prepare_send(amount.into(), SendOptions::default())
            .await
            .map_err(|e| ProviderError::Authentication(format!("Failed to prepare payment: {e}")))
            .unwrap();

        let token = self
            .wallet
            .send(prepare_send, None)
            .await
            .map_err(|e| ProviderError::Authentication(format!("Failed to send payment: {e}")))?
            .to_string();

        Ok(token)
    }

    async fn post(&self, payload: Value, input_token_count: usize) -> Result<Value, ProviderError> {
        let base_url = url::Url::parse(&self.host)
            .map_err(|e| ProviderError::RequestFailed(format!("Invalid base URL: {e}")))?;
        let url = base_url.join("v1/chat/completions").map_err(|e| {
            ProviderError::RequestFailed(format!("Failed to construct endpoint URL: {e}"))
        })?;

        let wallet_balance = self.wallet.total_balance().await.unwrap();

        let input_cost = self.prompt_cost.map(|a| a * input_token_count as f64);

        let max_output_tokens = self.model.context_limit.map(|a| a - input_token_count);

        let max_output_cost =
            max_output_tokens.and_then(|a| self.completion_cost.map(|c| a as f64 * c));

        let max_total_cost =
            max_output_cost.and_then(|a| input_cost.map(|c| (c.ceil() + a.ceil()) as u64));

        let amount = if let Some(max_total_cost) = max_total_cost {
            if Amount::from(max_total_cost) >= wallet_balance {
                u64::from(wallet_balance)
            } else {
                max_total_cost
            }
        } else {
            u64::from(wallet_balance)
        };

        // This will check balance and get token (get_auth_token includes balance check)
        let auth_token = self.get_auth_token(amount).await?;

        let response = self
            .client
            .post(url)
            .header("Authorization", format!("Bearer {auth_token}"))
            .json(&payload)
            .send()
            .await;

        if let Err(err) = self.handle_refund(&auth_token).await {
            tracing::error!("Could not get refund for {}", auth_token);
            tracing::error!("{}", err);
        }

        handle_response_openai_compat(response?).await
    }

    async fn handle_refund(&self, token: &str) -> Result<(), ProviderError> {
        let base_url = url::Url::parse(&self.host)
            .map_err(|e| ProviderError::RequestFailed(format!("Invalid base URL: {e}")))?;
        let url = base_url.join("/v1/wallet/refund").map_err(|e| {
            ProviderError::RequestFailed(format!("Failed to construct endpoint URL: {e}"))
        })?;

        let response = self
            .client
            .post(url)
            .header("Authorization", format!("Bearer {token}"))
            .send()
            .await?;

        let response: Value = response.json().await?;

        if let Some(token) = response.get("token") {
            match self
                .wallet
                .receive(
                    &token.to_string().trim_matches('"'),
                    cdk::wallet::ReceiveOptions::default(),
                )
                .await
            {
                Ok(amount) => {
                    tracing::debug!("Claimed change from mint: {} sats.", amount);
                }
                Err(e) => {
                    tracing::error!("Failed to claim change: {}", e);
                    tracing::error!("{}", token);
                }
            }
        }

        Ok(())
    }

    /// Returns the current wallet balance in satoshis
    pub async fn get_balance(&self) -> Result<u64, ProviderError> {
        let balance = self.wallet.total_balance().await.map_err(|e| {
            ProviderError::RequestFailed(format!("Failed to get wallet balance: {e}"))
        })?;
        Ok(balance.into())
    }

    /// Get models
    pub async fn get_models_info(
        host: &str,
        wallet: &Wallet,
    ) -> Result<ModelsResponse, ProviderError> {
        let base_url = url::Url::parse(&host)
            .map_err(|e| ProviderError::RequestFailed(format!("Invalid base URL: {e}")))?;
        let url = base_url.join("/v1/models").map_err(|e| {
            ProviderError::RequestFailed(format!("Failed to construct endpoint URL: {e}"))
        })?;

        let token = wallet
            .prepare_send(1.into(), SendOptions::default())
            .await
            .map_err(|e| {
                ProviderError::Authentication(format!("Failed to prepare payment: {e}"))
            })?;

        let token = wallet
            .send(token, None)
            .await
            .map_err(|e| ProviderError::Authentication(format!("Failed to send payment: {e}")))?;

        let client = Client::builder()
            .timeout(Duration::from_secs(600))
            .build()?;

        let response = client
            .get(url)
            .header("Authorization", format!("Bearer {token}"))
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
                ConfigKey::new("CDK_DIR", false, false, None),
                ConfigKey::new(
                    "ROUTSTR_MINT_URL",
                    true,
                    false,
                    Some(ROUTSTR_DEFAULT_MINT_URL),
                ),
                ConfigKey::new(
                    "ROUTSTR_CURRENCY_UNIT",
                    true,
                    false,
                    Some(ROUTSTR_DEFAULT_CURRENCY_UNIT),
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
        let token_counter = TokenCounter::new(self.model.tokenizer_name());

        let token_count = token_counter.count_everything(system, messages, tools, &[]);

        // Create payload without tools since the endpoint doesn't support them
        let payload = create_request(&self.model, system, messages, &[], &ImageFormat::OpenAi)?;

        // Make request
        let response = self.post(payload.clone(), token_count).await?;

        // Parse response
        let mut message = response_to_message(response.clone())?;

        // If tools are provided, augment the message with tool calls using the tool shim
        if !tools.is_empty() {
            message = augment_message_with_tool_calls(
                &RoutstrInterpreter::new(self.wallet.clone())?,
                message,
                tools,
            )
            .await?;
        }

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
        Ok(None)
    }
}
