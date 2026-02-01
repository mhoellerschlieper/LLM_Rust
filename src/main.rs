// main.rs
// Description: Binary entry point with menu loop. Builds model from tokenizer,
//              supports checkpoint save and load.
// History:
// - 2026-02-01: Add menu loop and checkpoint save and load.
// - 2026-02-01: Fix checkpoint load by rebuilding model from checkpoint tokenizer vocab.
// Author: Marcus Schlieper

mod layer;
mod math;
mod tokenizer;
mod train;
mod utils;

use std::io::Write;

use crate::layer::{Embeddings, Llm, OutputProjection, TransformerBlock};
use crate::tokenizer::{BpeTokenizer, BpeTokenizerConfig};
use crate::train::{Dataset, DatasetType};

pub const MAX_SEQ_LEN: usize = 80;
pub const EMBEDDING_DIM: usize = 128;
pub const HIDDEN_DIM: usize = 256;

fn read_line_ascii_trimmed() -> Result<String, String> {
    let mut s_input = String::new();
    std::io::stdin()
        .read_line(&mut s_input)
        .map_err(|_| "input_read_error".to_string())?;
    Ok(s_input.trim().to_string())
}

// Build a fresh model whose dimensions match the tokenizer vocab.
fn build_llm_from_tokenizer(bpe: BpeTokenizer) -> Llm {
    let vocab = bpe.vocab.clone();

    let embeddings = Embeddings::new(vocab.clone());
    let block1 = TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM);
    let block2 = TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM);
    let block3 = TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM);

    // IMPORTANT: vocab size must match tokenizer vocab size.
    let out = OutputProjection::new(EMBEDDING_DIM, vocab.words.len());

    let mut llm = Llm::new(
        vocab,
        vec![
            Box::new(embeddings),
            Box::new(block1),
            Box::new(block2),
            Box::new(block3),
            Box::new(out),
        ],
    );

    llm.set_bpe_tokenizer(bpe);
    llm
}

fn main() {
    let s_prompt = String::from("User: How do mountains form?");

    let dataset = Dataset::new(
        "data/pretraining_data.json",
        "data/chat_training_data.json",
        DatasetType::JSON,
    );

    // NOTE: For now, keep initial tokenizer training to allow immediate usage.
    // The important fix is that loading rebuilds the model to match checkpoint vocab.
    let mut v_corpus: Vec<String> = Vec::new();
    v_corpus.extend(dataset.pretraining_data.clone());
    v_corpus.extend(dataset.chat_training_data.clone());

    let mut config = BpeTokenizerConfig::default();
    config.i_vocab_target = 2000;
    config.i_min_pair_count = 2;
    // config.u64_seed can be changed for experimentation, but remains stored in checkpoint.

    let bpe = match BpeTokenizer::train_from_corpus_with_config(&v_corpus, config) {
        Ok(tok) => tok,
        Err(e) => {
            eprintln!("Tokenizer training failed: {}", e);
            return;
        }
    };

    let mut llm = build_llm_from_tokenizer(bpe);

    println!("\n=== MODEL INFORMATION ===");
    println!("Network architecture: {}", llm.network_description());
    println!(
        "Model configuration -> max_seq_len: {}, embedding_dim: {}, hidden_dim: {}",
        MAX_SEQ_LEN, EMBEDDING_DIM, HIDDEN_DIM
    );
    println!("Total parameters: {}", llm.total_parameters());

    let mut s_checkpoint_path: String = "checkpoints/llm_checkpoint.json".to_string();

    loop {
        println!("\n--- Menu Mode ---");
        println!("Commands:");
        println!("  t Train");
        println!("  l Load checkpoint");
        println!("  s Save checkpoint");
        println!("  a Ask");
        println!("  e Exit");
        print!("\nEnter command: ");
        let _ = std::io::stdout().flush();

        let s_cmd = match read_line_ascii_trimmed() {
            Ok(s) => s.to_lowercase(),
            Err(e) => {
                println!("Input error: {}", e);
                continue;
            }
        };

        if s_cmd == "e" {
            println!("Exit.");
            break;
        }

        if s_cmd == "t" {
            let v_pretraining_examples: Vec<&str> = dataset
                .pretraining_data
                .iter()
                .map(|s| s.as_str())
                .collect();

            let v_chat_training_examples: Vec<&str> = dataset
                .chat_training_data
                .iter()
                .map(|s| s.as_str())
                .collect();

            println!("\n=== PRE-TRAINING MODEL ===");
            println!(
                "Pre-training on {} examples for {} epochs with learning rate {}",
                dataset.pretraining_data.len(),
                100,
                0.0005
            );

            if let Err(e) = llm.train(v_pretraining_examples, 100, 0.0005) {
                eprintln!("Training failed: {}", e);
                continue;
            }

            println!("\n=== INSTRUCTION TUNING ===");
            println!(
                "Instruction tuning on {} examples for {} epochs with learning rate {}",
                dataset.chat_training_data.len(),
                200,
                0.0001
            );

            if let Err(e) = llm.train(v_chat_training_examples, 200, 0.0001) {
                eprintln!("Training failed: {}", e);
                continue;
            }

            continue;
        }

        if s_cmd == "s" {
            print!("Enter checkpoint path or press Enter for default: ");
            let _ = std::io::stdout().flush();

            let s_path = match read_line_ascii_trimmed() {
                Ok(s) => s,
                Err(e) => {
                    println!("Input error: {}", e);
                    continue;
                }
            };

            if !s_path.is_empty() {
                s_checkpoint_path = s_path;
            }

            match llm.save_checkpoint(&s_checkpoint_path) {
                Ok(()) => println!("Saved checkpoint: {}", s_checkpoint_path),
                Err(e) => println!("Save failed: {}", e),
            }

            continue;
        }

        if s_cmd == "l" {
            print!("Enter checkpoint path or press Enter for default: ");
            let _ = std::io::stdout().flush();

            let s_path = match read_line_ascii_trimmed() {
                Ok(s) => s,
                Err(e) => {
                    println!("Input error: {}", e);
                    continue;
                }
            };

            if !s_path.is_empty() {
                s_checkpoint_path = s_path;
            }

            // IMPORTANT: Rebuild model to match checkpoint tokenizer and vocab size.
            match Llm::load_checkpoint_rebuild(&s_checkpoint_path) {
                Ok(llm_loaded) => {
                    llm = llm_loaded;
                    println!("Loaded checkpoint: {}", s_checkpoint_path);
                }
                Err(e) => println!("Load failed: {}", e),
            }

            continue;
        }

        if s_cmd == "a" {
            print!("Enter prompt: ");
            let _ = std::io::stdout().flush();

            let s_user = match read_line_ascii_trimmed() {
                Ok(s) => s,
                Err(e) => {
                    println!("Input error: {}", e);
                    continue;
                }
            };

            if s_user.is_empty() {
                println!("Empty prompt.");
                continue;
            }

            let s_formatted = format!("User: {}", s_user);
            match llm.predict(&s_formatted) {
                Ok(s_out) => println!("Model output: {}", s_out),
                Err(e) => println!("Model output error: {}", e),
            }

            continue;
        }

        println!("Unknown command.");
    }
}
