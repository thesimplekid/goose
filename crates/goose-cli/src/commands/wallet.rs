use anyhow::Result;
use bip39::Mnemonic;
use cdk::nuts::CurrencyUnit;
use cdk::nuts::Token;
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

    let pre_swap = wallet.swap_from_unspent(balance, None, false).await?;

    let token = Token::new(wallet.mint_url, pre_swap, None, wallet.unit);

    set_current_token(token.to_string())?;

    Ok(())
}

fn get_current_token() -> Result<String> {
    let config = Config::global();

    Ok(config
        .get_param::<String>("ROUTSTR_API_KEY")?
        .to_string()
        .trim()
        .to_string())
}

fn set_current_token(token: String) -> Result<()> {
    let config = Config::global();

    config.set_param("ROUTSTR_API_KEY", Value::String(token.trim().to_string()))?;

    Ok(())
}

fn clear_current_token() -> Result<()> {
    let config = Config::global();
    config.delete("ROUTSTR_API_KEY")?;
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
        clear_current_token()?;
    }

    Ok(())
}

pub async fn handle_wallet_topup(top_up_token: String) -> Result<()> {
    let wallet = initialize_wallet().await?;

    if top_up_token.trim().is_empty() {
        println!("No token provided. Operation cancelled.");
        return Ok(());
    }

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

    let pre_swap = wallet.swap_from_unspent(balance, None, false).await?;

    let token = Token::new(wallet.mint_url, pre_swap, None, wallet.unit);

    set_current_token(token.to_string())?;

    Ok(())
}

pub async fn handle_wallet_withdraw(amount: Option<u64>) -> Result<()> {
    let wallet = initialize_wallet().await?;

    if let Some(current_token) = get_current_token().ok() {
        println!("{}", current_token);
        handle_refund(&current_token, &wallet).await?;
    }

    let balance = wallet.total_balance().await?;

    if balance > Amount::ZERO {
        let amount = amount.map(Amount::from);

        let amount = amount.unwrap_or(balance);

        let prep_send = wallet.prepare_send(amount, SendOptions::default()).await?;

        let send = wallet.send(prep_send, None).await?;

        println!("{}", send);
    } else {
        println!("Wallet is empty.");
    }

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
