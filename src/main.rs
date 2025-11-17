// Copyright (c) 2025 Julius ML
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

use clap::{Parser, Subcommand};

use post_cortex::{ConversationMemorySystem, SystemConfig};
use std::collections::HashMap;
use uuid::Uuid;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Create a new conversation session
    CreateSession,

    /// Add an interaction to a session
    AddInteraction {
        #[arg(short, long)]
        session_id: String,
        #[arg(short, long)]
        interaction_type: String,
        #[arg(short, long, value_parser = parse_key_val)]
        content: Vec<(String, String)>,
    },

    /// Query conversation context
    QueryContext {
        #[arg(short, long)]
        session_id: String,
        #[arg(short, long)]
        query_type: String,
        #[arg(short, long, value_parser = parse_key_val)]
        parameters: Vec<(String, String)>,
    },

    /// Create a session checkpoint
    CreateCheckpoint {
        #[arg(short, long)]
        session_id: String,
    },

    /// Load a session from checkpoint
    LoadCheckpoint {
        #[arg(short, long)]
        checkpoint_id: String,
        #[arg(short, long)]
        session_id: String,
    },
}

fn parse_key_val(s: &str) -> Result<(String, String), String> {
    let pos = s
        .find('=')
        .ok_or_else(|| format!("invalid KEY=value: no `=` found in `{s}`"))?;
    Ok((s[..pos].to_string(), s[pos + 1..].to_string()))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Post-Cortex CLI started");

    let cli = Cli::parse();

    match cli.command {
        Commands::CreateSession => {
            let config = SystemConfig::default();
            let system = ConversationMemorySystem::new(config).await?;
            let session_id = system.create_session(None, None).await?;
            println!("Session created with ID: {session_id}");
        }

        Commands::AddInteraction {
            session_id,
            interaction_type,
            content,
        } => {
            let session_id = Uuid::parse_str(&session_id)?;
            let content_map: HashMap<String, String> = content.into_iter().collect();

            let result = post_cortex::tools::mcp::update_conversation_context(
                interaction_type,
                content_map,
                None,
                session_id,
            )
            .await?;

            if result.success {
                println!("Interaction added successfully");
                if let Some(data) = result.data {
                    println!("Data: {data:#?}");
                }
            } else {
                println!("Error: {}", result.message);
            }
        }

        Commands::QueryContext {
            session_id,
            query_type,
            parameters,
        } => {
            let session_id = Uuid::parse_str(&session_id)?;
            let params_map: HashMap<String, String> = parameters.into_iter().collect();

            let result = post_cortex::tools::mcp::query_conversation_context(
                query_type, params_map, session_id,
            )
            .await?;

            if result.success {
                println!("Query successful");
                if let Some(data) = result.data {
                    println!("Result: {:#?}", data);
                }
            } else {
                println!("Error: {}", result.message);
            }
        }

        Commands::CreateCheckpoint { session_id } => {
            let session_id = Uuid::parse_str(&session_id)?;
            let result = post_cortex::tools::mcp::create_session_checkpoint(session_id).await?;

            if result.success {
                println!("Checkpoint created successfully");
                if let Some(data) = result.data {
                    println!("Checkpoint ID: {}", data["checkpoint_id"]);
                }
            } else {
                println!("Error: {}", result.message);
            }
        }

        Commands::LoadCheckpoint {
            checkpoint_id,
            session_id,
        } => {
            let session_id = Uuid::parse_str(&session_id)?;
            let result =
                post_cortex::tools::mcp::load_session_checkpoint(checkpoint_id, session_id).await?;

            if result.success {
                println!("Session loaded from checkpoint successfully");
            } else {
                println!("Error: {}", result.message);
            }
        }
    }

    Ok(())
}
