use std::sync::Arc;

use super::{
    anthropic::AnthropicProvider,
    azure::AzureProvider,
    base::{Provider, ProviderMetadata},
    bedrock::BedrockProvider,
    databricks::DatabricksProvider,
    gcpvertexai::GcpVertexAIProvider,
    githubcopilot::GithubCopilotProvider,
    google::GoogleProvider,
    groq::GroqProvider,
    lead_worker::LeadWorkerProvider,
    ollama::OllamaProvider,
    openai::OpenAiProvider,
    openrouter::OpenRouterProvider,
    routstr::RoutstrProvider,
    sagemaker_tgi::SageMakerTgiProvider,
    snowflake::SnowflakeProvider,
    venice::VeniceProvider,
    xai::XaiProvider,
};
use crate::model::ModelConfig;
use anyhow::Result;

#[cfg(test)]
use super::errors::ProviderError;
#[cfg(test)]
use mcp_core::tool::Tool;

fn default_lead_turns() -> usize {
    3
}
fn default_failure_threshold() -> usize {
    2
}
fn default_fallback_turns() -> usize {
    2
}

pub fn providers() -> Vec<ProviderMetadata> {
    vec![
        AnthropicProvider::metadata(),
        AzureProvider::metadata(),
        BedrockProvider::metadata(),
        DatabricksProvider::metadata(),
        GcpVertexAIProvider::metadata(),
        GithubCopilotProvider::metadata(),
        GoogleProvider::metadata(),
        GroqProvider::metadata(),
        OllamaProvider::metadata(),
        OpenAiProvider::metadata(),
        OpenRouterProvider::metadata(),
        RoutstrProvider::metadata(),
        SageMakerTgiProvider::metadata(),
        VeniceProvider::metadata(),
        SnowflakeProvider::metadata(),
        XaiProvider::metadata(),
    ]
}

pub fn create(name: &str, model: ModelConfig) -> Result<Arc<dyn Provider>> {
    let config = crate::config::Config::global();

    // Check for lead model environment variables
    if let Ok(lead_model_name) = config.get_param::<String>("GOOSE_LEAD_MODEL") {
        tracing::info!("Creating lead/worker provider from environment variables");

        return create_lead_worker_from_env(name, &model, &lead_model_name);
    }

    // Default: create regular provider
    create_provider(name, model)
}

/// Create a lead/worker provider from environment variables
fn create_lead_worker_from_env(
    default_provider_name: &str,
    default_model: &ModelConfig,
    lead_model_name: &str,
) -> Result<Arc<dyn Provider>> {
    let config = crate::config::Config::global();

    // Get lead provider (optional, defaults to main provider)
    let lead_provider_name = config
        .get_param::<String>("GOOSE_LEAD_PROVIDER")
        .unwrap_or_else(|_| default_provider_name.to_string());

    // Get configuration parameters with defaults
    let lead_turns = config
        .get_param::<usize>("GOOSE_LEAD_TURNS")
        .unwrap_or(default_lead_turns());
    let failure_threshold = config
        .get_param::<usize>("GOOSE_LEAD_FAILURE_THRESHOLD")
        .unwrap_or(default_failure_threshold());
    let fallback_turns = config
        .get_param::<usize>("GOOSE_LEAD_FALLBACK_TURNS")
        .unwrap_or(default_fallback_turns());

    // Create model configs
    let lead_model_config = ModelConfig::new(lead_model_name.to_string());
    let worker_model_config = default_model.clone();

    // Create the providers
    let lead_provider = create_provider(&lead_provider_name, lead_model_config)?;
    let worker_provider = create_provider(default_provider_name, worker_model_config)?;

    // Create the lead/worker provider with configured settings
    Ok(Arc::new(LeadWorkerProvider::new_with_settings(
        lead_provider,
        worker_provider,
        lead_turns,
        failure_threshold,
        fallback_turns,
    )))
}

fn create_provider(name: &str, model: ModelConfig) -> Result<Arc<dyn Provider>> {
    // We use Arc instead of Box to be able to clone for multiple async tasks
    match name {
        "openai" => Ok(Arc::new(OpenAiProvider::from_env(model)?)),
        "anthropic" => Ok(Arc::new(AnthropicProvider::from_env(model)?)),
        "azure_openai" => Ok(Arc::new(AzureProvider::from_env(model)?)),
        "aws_bedrock" => Ok(Arc::new(BedrockProvider::from_env(model)?)),
        "databricks" => Ok(Arc::new(DatabricksProvider::from_env(model)?)),
        "groq" => Ok(Arc::new(GroqProvider::from_env(model)?)),
        "ollama" => Ok(Arc::new(OllamaProvider::from_env(model)?)),
        "openrouter" => Ok(Arc::new(OpenRouterProvider::from_env(model)?)),
        "gcp_vertex_ai" => Ok(Arc::new(GcpVertexAIProvider::from_env(model)?)),
        "google" => Ok(Arc::new(GoogleProvider::from_env(model)?)),
        "sagemaker_tgi" => Ok(Arc::new(SageMakerTgiProvider::from_env(model)?)),
        "venice" => Ok(Arc::new(VeniceProvider::from_env(model)?)),
        "snowflake" => Ok(Arc::new(SnowflakeProvider::from_env(model)?)),
        "github_copilot" => Ok(Arc::new(GithubCopilotProvider::from_env(model)?)),
        "xai" => Ok(Arc::new(XaiProvider::from_env(model)?)),
        "routstr" => Ok(Arc::new(RoutstrProvider::from_env(model)?)),
        _ => Err(anyhow::anyhow!("Unknown provider: {}", name)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{Message, MessageContent};
    use crate::providers::base::{ProviderMetadata, ProviderUsage, Usage};
    use chrono::Utc;
    use mcp_core::{content::TextContent, Role};
    use std::env;

    #[derive(Clone)]
    struct MockTestProvider {
        name: String,
        model_config: ModelConfig,
    }

    #[async_trait::async_trait]
    impl Provider for MockTestProvider {
        fn metadata() -> ProviderMetadata {
            ProviderMetadata::new(
                "mock_test",
                "Mock Test Provider",
                "A mock provider for testing",
                "mock-model",
                vec!["mock-model"],
                "",
                vec![],
            )
        }

        fn get_model_config(&self) -> ModelConfig {
            self.model_config.clone()
        }

        async fn complete(
            &self,
            _system: &str,
            _messages: &[Message],
            _tools: &[Tool],
        ) -> Result<(Message, ProviderUsage), ProviderError> {
            Ok((
                Message {
                    role: Role::Assistant,
                    created: Utc::now().timestamp(),
                    content: vec![MessageContent::Text(TextContent {
                        text: format!(
                            "Response from {} with model {}",
                            self.name, self.model_config.model_name
                        ),
                        annotations: None,
                    })],
                },
                ProviderUsage::new(self.model_config.model_name.clone(), Usage::default()),
            ))
        }
    }

    #[test]
    fn test_create_lead_worker_provider() {
        // Save current env vars
        let saved_lead = env::var("GOOSE_LEAD_MODEL").ok();
        let saved_provider = env::var("GOOSE_LEAD_PROVIDER").ok();
        let saved_turns = env::var("GOOSE_LEAD_TURNS").ok();

        // Test with basic lead model configuration
        env::set_var("GOOSE_LEAD_MODEL", "gpt-4o");

        // This will try to create a lead/worker provider
        let result = create("openai", ModelConfig::new("gpt-4o-mini".to_string()));

        // The creation might succeed or fail depending on API keys, but we can verify the logic path
        match result {
            Ok(_) => {
                // If it succeeds, it means we created a lead/worker provider successfully
                // This would happen if API keys are available in the test environment
            }
            Err(error) => {
                // If it fails, it should be due to missing API keys, confirming we tried to create providers
                let error_msg = error.to_string();
                assert!(error_msg.contains("OPENAI_API_KEY") || error_msg.contains("secret"));
            }
        }

        // Test with different lead provider
        env::set_var("GOOSE_LEAD_PROVIDER", "anthropic");
        env::set_var("GOOSE_LEAD_TURNS", "5");

        let _result = create("openai", ModelConfig::new("gpt-4o-mini".to_string()));
        // Similar validation as above - will fail due to missing API keys but confirms the logic

        // Restore env vars
        match saved_lead {
            Some(val) => env::set_var("GOOSE_LEAD_MODEL", val),
            None => env::remove_var("GOOSE_LEAD_MODEL"),
        }
        match saved_provider {
            Some(val) => env::set_var("GOOSE_LEAD_PROVIDER", val),
            None => env::remove_var("GOOSE_LEAD_PROVIDER"),
        }
        match saved_turns {
            Some(val) => env::set_var("GOOSE_LEAD_TURNS", val),
            None => env::remove_var("GOOSE_LEAD_TURNS"),
        }
    }

    #[test]
    fn test_lead_model_env_vars_with_defaults() {
        // Save current env vars
        let saved_vars = [
            ("GOOSE_LEAD_MODEL", env::var("GOOSE_LEAD_MODEL").ok()),
            ("GOOSE_LEAD_PROVIDER", env::var("GOOSE_LEAD_PROVIDER").ok()),
            ("GOOSE_LEAD_TURNS", env::var("GOOSE_LEAD_TURNS").ok()),
            (
                "GOOSE_LEAD_FAILURE_THRESHOLD",
                env::var("GOOSE_LEAD_FAILURE_THRESHOLD").ok(),
            ),
            (
                "GOOSE_LEAD_FALLBACK_TURNS",
                env::var("GOOSE_LEAD_FALLBACK_TURNS").ok(),
            ),
        ];

        // Clear all lead env vars
        for (key, _) in &saved_vars {
            env::remove_var(key);
        }

        // Set only the required lead model
        env::set_var("GOOSE_LEAD_MODEL", "grok-3");

        // This should use defaults for all other values
        let result = create("openai", ModelConfig::new("gpt-4o-mini".to_string()));

        // Should attempt to create lead/worker provider (will fail due to missing API keys but confirms logic)
        match result {
            Ok(_) => {
                // Success means we have API keys and created the provider
            }
            Err(error) => {
                // Should fail due to missing API keys, confirming we tried to create providers
                let error_msg = error.to_string();
                assert!(error_msg.contains("OPENAI_API_KEY") || error_msg.contains("secret"));
            }
        }

        // Test with custom values
        env::set_var("GOOSE_LEAD_TURNS", "7");
        env::set_var("GOOSE_LEAD_FAILURE_THRESHOLD", "4");
        env::set_var("GOOSE_LEAD_FALLBACK_TURNS", "3");

        let _result = create("openai", ModelConfig::new("gpt-4o-mini".to_string()));
        // Should still attempt to create lead/worker provider with custom settings

        // Restore all env vars
        for (key, value) in saved_vars {
            match value {
                Some(val) => env::set_var(key, val),
                None => env::remove_var(key),
            }
        }
    }

    #[test]
    fn test_create_regular_provider_without_lead_config() {
        // Save current env vars
        let saved_lead = env::var("GOOSE_LEAD_MODEL").ok();
        let saved_provider = env::var("GOOSE_LEAD_PROVIDER").ok();
        let saved_turns = env::var("GOOSE_LEAD_TURNS").ok();
        let saved_threshold = env::var("GOOSE_LEAD_FAILURE_THRESHOLD").ok();
        let saved_fallback = env::var("GOOSE_LEAD_FALLBACK_TURNS").ok();

        // Ensure all GOOSE_LEAD_* variables are not set
        env::remove_var("GOOSE_LEAD_MODEL");
        env::remove_var("GOOSE_LEAD_PROVIDER");
        env::remove_var("GOOSE_LEAD_TURNS");
        env::remove_var("GOOSE_LEAD_FAILURE_THRESHOLD");
        env::remove_var("GOOSE_LEAD_FALLBACK_TURNS");

        // This should try to create a regular provider
        let result = create("openai", ModelConfig::new("gpt-4o-mini".to_string()));

        // The creation might succeed or fail depending on API keys
        match result {
            Ok(_) => {
                // If it succeeds, it means we created a regular provider successfully
                // This would happen if API keys are available in the test environment
            }
            Err(error) => {
                // If it fails, it should be due to missing API keys
                let error_msg = error.to_string();
                assert!(error_msg.contains("OPENAI_API_KEY") || error_msg.contains("secret"));
            }
        }

        // Restore env vars
        if let Some(val) = saved_lead {
            env::set_var("GOOSE_LEAD_MODEL", val);
        }
        if let Some(val) = saved_provider {
            env::set_var("GOOSE_LEAD_PROVIDER", val);
        }
        if let Some(val) = saved_turns {
            env::set_var("GOOSE_LEAD_TURNS", val);
        }
        if let Some(val) = saved_threshold {
            env::set_var("GOOSE_LEAD_FAILURE_THRESHOLD", val);
        }
        if let Some(val) = saved_fallback {
            env::set_var("GOOSE_LEAD_FALLBACK_TURNS", val);
        }
    }
}
