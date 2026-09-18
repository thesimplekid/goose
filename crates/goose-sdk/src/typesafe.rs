//! Typed access to TypeSafe's Jev evaluation API, independent of chat providers.

use std::collections::HashMap;

use anyhow::{ensure, Result};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::Value;

mod client;
mod error;
mod retry;

pub use client::{
    ApiResponse, RequestEvent, RequestObserver, RequestOptions, TypeSafeClient,
    TypeSafeClientBuilder,
};
pub use error::{ApiError, ApiErrorKind, TypeSafeError};
pub use retry::{parse_retry_after, RetryPolicy};

pub const DEFAULT_MODEL: &str = "jev-latest";
pub const DEFAULT_BASE_URL: &str = "https://api.typesafe.ai";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Question {
    Noul {
        #[serde(default)]
        instructions: Value,
        #[serde(
            default,
            deserialize_with = "nullable_criteria",
            skip_serializing_if = "HashMap::is_empty"
        )]
        criteria: HashMap<String, Value>,
    },
    Choice {
        #[serde(default)]
        instructions: Value,
        criteria: HashMap<String, Value>,
    },
    Score {
        #[serde(default)]
        instructions: Value,
        criteria: Vec<Value>,
    },
}

fn nullable_criteria<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<HashMap<String, Value>, D::Error> {
    Ok(Option::<HashMap<String, Value>>::deserialize(deserializer)?.unwrap_or_default())
}

impl Question {
    pub fn noul(instructions: impl Into<Value>) -> Self {
        Self::Noul {
            instructions: instructions.into(),
            criteria: HashMap::new(),
        }
    }

    pub fn choice(instructions: impl Into<Value>, criteria: HashMap<String, Value>) -> Self {
        Self::Choice {
            instructions: instructions.into(),
            criteria,
        }
    }

    pub fn score(instructions: impl Into<Value>, criteria: Vec<Value>) -> Result<Self> {
        let question = Self::Score {
            instructions: instructions.into(),
            criteria,
        };
        question.validate()?;
        Ok(question)
    }

    pub(crate) fn validate(&self) -> Result<()> {
        let instructions = match self {
            Self::Noul {
                instructions,
                criteria,
            } => {
                ensure!(
                    criteria.keys().all(|key| key == "true" || key == "false"),
                    TypeSafeError::InvalidInput("Noul criteria must use true or false keys".into())
                );
                for value in criteria.values() {
                    validate_entry(value)?;
                }
                instructions
            }
            Self::Choice {
                instructions,
                criteria,
            } => {
                for value in criteria.values() {
                    validate_entry(value)?;
                }
                instructions
            }
            Self::Score {
                instructions,
                criteria,
            } => {
                ensure!(
                    criteria.len() >= 2,
                    TypeSafeError::InvalidInput("Score requires at least two levels".into())
                );
                for value in criteria {
                    validate_entry(value)?;
                }
                instructions
            }
        };
        validate_entry(instructions)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Answer {
    Noul {
        noul: f64,
    },
    Choice {
        choice: String,
        probabilities: HashMap<String, f64>,
        confidence: f64,
    },
    Score {
        score: f64,
        legend: HashMap<String, Value>,
        probabilities: HashMap<String, f64>,
        confidence: f64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evaluation {
    pub model: String,
    pub answers: HashMap<String, Answer>,
    pub usage: Usage,
}

impl Evaluation {
    /// Decode named answers into an application struct, including enum choice labels.
    pub fn answers_as<T: DeserializeOwned>(&self) -> Result<T> {
        Ok(serde_json::from_value(serde_json::to_value(&self.answers)?)
            .map_err(TypeSafeError::Decode)?)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCard {
    pub name: String,
    pub description: String,
    pub release_date: String,
}

pub(crate) fn validate_entry(value: &Value) -> Result<()> {
    ensure!(
        matches!(
            value,
            Value::Null | Value::String(_) | Value::Object(_) | Value::Array(_)
        ),
        TypeSafeError::InvalidInput(
            "TypeSafe entries must be null, a string, object, or array".into()
        )
    );
    Ok(())
}
