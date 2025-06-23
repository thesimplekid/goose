use anyhow::Result;
use bip39::Mnemonic;
use cdk::nuts::CurrencyUnit;
use cdk::wallet::{ReceiveOptions, SendOptions, Wallet};
use cdk::Amount;
use cdk_sqlite::WalletSqliteDatabase;
use home::home_dir;
use std::fs;
use std::str::FromStr;
use std::sync::Arc;

const DEFAULT_MINT_URL: &str = "https://mint.minibits.cash/Bitcoin";

pub async fn handle_wallet_balance() -> Result<()> {
    let wallet = initialize_wallet().await?;
    let balance = wallet.total_balance().await?;
    println!("Current wallet balance: {} sats", balance);
    Ok(())
}

pub async fn handle_wallet_topup(token: String) -> Result<()> {
    let wallet = initialize_wallet().await?;

    if token.trim().is_empty() {
        println!("No token provided. Operation cancelled.");
        return Ok(());
    }

    match wallet.receive(&token, ReceiveOptions::default()).await {
        Ok(_) => {
            let new_balance = wallet.total_balance().await?;
            println!("Successfully topped up wallet!");
            println!("New balance: {} sats", new_balance);
            Ok(())
        }
        Err(e) => {
            eprintln!("Failed to process token: {}", e);
            Err(e.into())
        }
    }
}

pub async fn handle_wallet_withdraw(amount: Option<u64>) -> Result<()> {
    let wallet = initialize_wallet().await?;

    let current_balance = wallet.total_balance().await?;
    if current_balance == Amount::ZERO {
        println!("Wallet is empty. Cannot withdraw.");
        return Ok(());
    }

    let withdraw_amount = match amount {
        Some(amt) => {
            let amt = Amount::from(amt);
            if amt > current_balance {
                println!(
                    "Cannot withdraw {} sats. Current balance is {} sats.",
                    amt, current_balance
                );
                return Ok(());
            }
            amt
        }
        None => current_balance,
    };

    let prepared_send = wallet
        .prepare_send(withdraw_amount, SendOptions::default())
        .await?;

    match wallet.send(prepared_send, None).await {
        Ok(token) => {
            let new_balance = wallet.total_balance().await?;
            println!(
                "Successfully created withdrawal token for {} sats",
                withdraw_amount
            );
            println!("Token: {}", token);
            println!("New balance: {} sats", new_balance);
            Ok(())
        }
        Err(e) => {
            eprintln!("Failed to create withdrawal token: {}", e);
            Err(e.into())
        }
    }
}

async fn initialize_wallet() -> Result<Wallet> {
    let work_dir = home_dir().unwrap().join(".cdk-cli");
    fs::create_dir_all(&work_dir)?;
    let cdk_wallet_path = work_dir.join("cdk-cli.sqlite");

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
