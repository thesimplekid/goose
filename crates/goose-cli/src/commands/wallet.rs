use anyhow::Result;
use bip39::Mnemonic;
use cdk::nuts::CurrencyUnit;
use cdk::wallet::{SendOptions, Wallet};
use cdk::Amount;
use cdk_sqlite::WalletSqliteDatabase;
use goose::config::Config;
use home::home_dir;
use reqwest::Client;
use serde_json::Value;
use std::fs;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;

const DEFAULT_MINT_URL: &str = "https://mint.minibits.cash/Bitcoin";

pub async fn handle_wallet_balance() -> Result<()> {
    let wallet = initialize_wallet().await?;

    if let Some(current_token) = get_current_token().ok() {
        handle_refund(&current_token, &wallet).await?;
    }

    let balance = wallet.total_balance().await?;

    println!("sats: {}", balance);

    let prep_send = wallet.prepare_send(balance, SendOptions::default()).await?;

    let send = wallet.send(prep_send, None).await?;

    set_current_token(send.to_string())?;

    Ok(())
}

fn get_current_token() -> Result<String> {
    let config = Config::global();

    Ok(config.get_secret("ROUTSTR_API_KEY")?)
}

fn set_current_token(token: String) -> Result<()> {
    let config = Config::global();

    config.set_secret("ROUTSTR_API_KEY", Value::String(token))?;

    Ok(())
}

async fn handle_refund(current_token: &str, wallet: &Wallet) -> Result<()> {
    let config = Config::global();

    let host: String = config.get_param("ROUTSTR_HOST")?;

    let base_url = url::Url::parse(&host)?;
    let url = base_url.join("/v1/wallet/refund")?;

    let client = Client::builder()
        .timeout(Duration::from_secs(600))
        .build()?;

    let response = client
        .post(url)
        .header("Authorization", format!("Bearer {current_token}"))
        .send()
        .await?;

    let response: Value = response.json().await?;

    if let Some(token) = response.get("token") {
        match wallet
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

pub async fn handle_wallet_topup(top_up_token: String) -> Result<()> {
    let wallet = initialize_wallet().await?;

    if top_up_token.trim().is_empty() {
        println!("No token provided. Operation cancelled.");
        return Ok(());
    }

    let config = Config::global();

    if let Some(current_token) = get_current_token().ok() {
        handle_refund(&current_token, &wallet).await?;
    }

    match wallet
        .receive(
            &top_up_token.to_string().trim_matches('"'),
            cdk::wallet::ReceiveOptions::default(),
        )
        .await
    {
        Ok(amount) => {
            tracing::debug!("Claimed change from mint: {} sats.", amount);
        }
        Err(e) => {
            tracing::error!("Failed to claim change: {}", e);
            tracing::error!("{}", top_up_token);
        }
    }

    let balance = wallet.total_balance().await?;

    let prep_send = wallet.prepare_send(balance, SendOptions::default()).await?;

    let token = wallet.send(prep_send, None).await?;

    config.set_secret("ROUTSTR_API_KEY", Value::String(token.to_string()))?;

    Ok(())
}

pub async fn handle_wallet_withdraw(amount: Option<u64>) -> Result<()> {
    let wallet = initialize_wallet().await?;

    if let Some(current_token) = get_current_token().ok() {
        handle_refund(&current_token, &wallet).await?;
    }

    let balance = wallet.total_balance().await?;

    let amount = amount.map(Amount::from);

    let amount = amount.unwrap_or(balance);

    let prep_send = wallet.prepare_send(amount, SendOptions::default()).await?;

    let send = wallet.send(prep_send, None).await?;

    println!("{}", send);

    Ok(())
}

async fn initialize_wallet() -> Result<Wallet> {
    let work_dir = home_dir().unwrap().join(".cdk-gooose");
    fs::create_dir_all(&work_dir)?;
    let cdk_wallet_path = work_dir.join("cdk-goose.sqlite");

    let wallet_db = WalletSqliteDatabase::new(&cdk_wallet_path).await?;

    let seed_path = work_dir.join("seed");

    let mnemonic = match fs::metadata(seed_path.clone()) {
        Ok(_) => {
            let contents = fs::read_to_string(seed_path.clone())?;
            Mnemonic::from_str(&contents)?
        }
        Err(_) => {
            let mnemonic = Mnemonic::generate(12)?;
            tracing::info!("Creating new seed");
            fs::write(seed_path, mnemonic.to_string())?;
            mnemonic
        }
    };

    let seed = mnemonic.to_seed_normalized("");
    let currency_unit = CurrencyUnit::Sat;

    let wallet = Wallet::new(
        DEFAULT_MINT_URL,
        currency_unit,
        Arc::new(wallet_db),
        &seed,
        None,
    )?;

    Ok(wallet)
}
