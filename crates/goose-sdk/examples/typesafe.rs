use std::collections::HashMap;

use goose_sdk::typesafe::{Question, TypeSafeClient};
use serde_json::json;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let client = TypeSafeClient::from_env()?;
    let evaluation = client
        .system_one(
            json!({"ticket": "I was charged twice. Please fix this ASAP."}),
            HashMap::from([
                ("urgent".into(), Question::noul("Is this urgent?")),
                (
                    "team".into(),
                    Question::choice(
                        "Which team should handle this?",
                        HashMap::from([
                            (
                                "billing".into(),
                                json!({"description": "Payments and refunds"}),
                            ),
                            ("support".into(), json!(null)),
                        ]),
                    ),
                ),
                (
                    "severity".into(),
                    Question::score(
                        "How severe is the problem?",
                        vec![
                            json!("Minor"),
                            json!("Major"),
                            json!({"description": "Critical"}),
                        ],
                    )?,
                ),
            ]),
        )
        .await?;
    println!("{}", serde_json::to_string_pretty(&evaluation)?);
    Ok(())
}
