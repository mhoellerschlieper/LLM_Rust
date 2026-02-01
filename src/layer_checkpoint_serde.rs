// layer_checkpoint_serde.rs
// Description: Serde helpers for checkpoint structs.
// History:
// - 2026-02-01: Provide Serialize and Deserialize for checkpoint structs without changing core logic.
// Author: Marcus Schlieper

// Note: This module is optional. If preferred, add the derives directly to the structs.
// This file can be removed if derives are added in place.

use serde::{Deserialize, Serialize};

use crate::layer::LlmCheckpoint;
use crate::tokenizer::{BpeMerge, BpeTokenizerCheckpoint};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SerdeBpeMerge {
    pub s_left: String,
    pub s_right: String,
    pub s_merged: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SerdeBpeTokenizerCheckpoint {
    pub v_vocab_words: Vec<String>,
    pub v_merges: Vec<SerdeBpeMerge>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SerdeLlmCheckpoint {
    pub s_magic: String,
    pub s_version: String,

    pub i_max_seq_len: usize,
    pub i_embedding_dim: usize,
    pub i_hidden_dim: usize,

    pub tokenizer: SerdeBpeTokenizerCheckpoint,
    pub v_params: Vec<f32>,
}

impl From<&BpeMerge> for SerdeBpeMerge {
    fn from(m: &BpeMerge) -> Self {
        Self {
            s_left: m.s_left.clone(),
            s_right: m.s_right.clone(),
            s_merged: m.s_merged.clone(),
        }
    }
}

impl From<&BpeTokenizerCheckpoint> for SerdeBpeTokenizerCheckpoint {
    fn from(cp: &BpeTokenizerCheckpoint) -> Self {
        Self {
            v_vocab_words: cp.v_vocab_words.clone(),
            v_merges: cp.v_merges.iter().map(SerdeBpeMerge::from).collect(),
        }
    }
}

impl From<&LlmCheckpoint> for SerdeLlmCheckpoint {
    fn from(cp: &LlmCheckpoint) -> Self {
        Self {
            s_magic: cp.s_magic.clone(),
            s_version: cp.s_version.clone(),
            i_max_seq_len: cp.i_max_seq_len,
            i_embedding_dim: cp.i_embedding_dim,
            i_hidden_dim: cp.i_hidden_dim,
            tokenizer: SerdeBpeTokenizerCheckpoint::from(&cp.tokenizer),
            v_params: cp.v_params.clone(),
        }
    }
}

impl SerdeLlmCheckpoint {
    pub fn into_domain(self) -> Result<LlmCheckpoint, String> {
        let v_merges: Vec<BpeMerge> = self
            .tokenizer
            .v_merges
            .into_iter()
            .map(|m| BpeMerge {
                s_left: m.s_left,
                s_right: m.s_right,
                s_merged: m.s_merged,
            })
            .collect();

        let tok = BpeTokenizerCheckpoint {
            v_vocab_words: self.tokenizer.v_vocab_words,
            v_merges: v_merges,
        };

        Ok(LlmCheckpoint {
            s_magic: self.s_magic,
            s_version: self.s_version,
            i_max_seq_len: self.i_max_seq_len,
            i_embedding_dim: self.i_embedding_dim,
            i_hidden_dim: self.i_hidden_dim,
            tokenizer: tok,
            v_params: self.v_params,
        })
    }
}
