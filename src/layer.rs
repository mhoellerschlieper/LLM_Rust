// layer.rs
// Description: Model layers and core LLM implementation (forward, backward, train, predict).
//              Consolidates embeddings, transformer blocks, self attention, feed forward,
//              layer norm, output projection, and Adam optimizer.
//              Adds checkpoint save and load for model parameters and tokenizer.
// History:
// - 2026-02-01: Consolidate project into 6 files: main, layer, train, math, tokenizer, utils.
// - 2026-02-01: Add checkpoint save and load for model parameters and tokenizer.
// Author: Marcus Schlieper

use std::cmp::Ordering;

use ndarray::{Array1, Array2, Axis};
use rand_distr::{Distribution, Normal};

use crate::{EMBEDDING_DIM, HIDDEN_DIM, MAX_SEQ_LEN};
use crate::math;
use crate::tokenizer::{BpeTokenizer, BpeTokenizerCheckpoint, S_EOS};
use crate::utils;
use std::ops::AddAssign;
use serde::{Serialize, Deserialize};

#[derive(Clone, Debug)]
pub struct Vocab {
    pub encode: std::collections::HashMap<String, usize>,
    pub decode: std::collections::HashMap<usize, String>,
    pub words: Vec<String>,
}

impl Default for Vocab {
    fn default() -> Self {
        Self::new(Self::default_words())
    }
}

impl Vocab {
    pub fn new(v_words: Vec<&str>) -> Self {
        let mut m_encode: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        let mut m_decode: std::collections::HashMap<usize, String> = std::collections::HashMap::new();

        for (i_id, s_word) in v_words.iter().enumerate() {
            m_encode.insert((*s_word).to_string(), i_id);
            m_decode.insert(i_id, (*s_word).to_string());
        }

        Self {
            encode: m_encode,
            decode: m_decode,
            words: v_words.iter().map(|w| (*w).to_string()).collect(),
        }
    }

    pub fn encode(&self, s_word: &str) -> Option<usize> {
        self.encode.get(s_word).copied()
    }

    pub fn decode(&self, i_token_id: usize) -> Option<&String> {
        self.decode.get(&i_token_id)
    }

    pub fn default_words() -> Vec<&'static str> {
        // Stable, explicit special tokens. Must match tokenizer.rs constants.
        vec!["<pad>", "<unk>", "<bos>", "<eos>", "hello", "world"]
    }
}

pub trait Layer {
    fn layer_type(&self) -> &str;

    // Conventions:
    // - token id input: [1, seq_len] as f32 (before embeddings)
    // - embedded and later: [seq_len, embedding_dim]
    // - logits: [seq_len, vocab_size]
    fn forward(&mut self, a_input: &Array2<f32>) -> Array2<f32>;

    fn backward(&mut self, a_grads: &Array2<f32>, d_lr: f32) -> Array2<f32>;

    fn parameters(&self) -> usize;

    // Checkpoint hooks.
    fn get_parameters_flat(&self) -> Vec<f32>;
    fn set_parameters_flat(&mut self, v_params: &[f32]) -> Result<usize, String>;
}

#[derive(Clone, Debug)]
pub struct Adam {
    d_beta1: f32,
    d_beta2: f32,
    d_eps: f32,
    i_t: usize,
    m_m: Array2<f32>,
    m_v: Array2<f32>,
}

impl Adam {
    pub fn new(t_shape: (usize, usize)) -> Self {
        Self {
            d_beta1: 0.9,
            d_beta2: 0.999,
            d_eps: 1e-8,
            i_t: 0,
            m_m: Array2::zeros(t_shape),
            m_v: Array2::zeros(t_shape),
        }
    }

    pub fn step(&mut self, a_params: &mut Array2<f32>, a_grads: &Array2<f32>, d_lr: f32) {
        if !d_lr.is_finite() || d_lr <= 0.0 {
            return;
        }
        if a_params.raw_dim() != a_grads.raw_dim() {
            return;
        }

        self.i_t = self.i_t.saturating_add(1);

        // m = b1*m + (1-b1)*g
        self.m_m = &self.m_m * self.d_beta1 + a_grads * (1.0 - self.d_beta1);

        // v = b2*v + (1-b2)*g^2
        let a_grads_sq = a_grads.mapv(|x| x * x);
        self.m_v = &self.m_v * self.d_beta2 + a_grads_sq * (1.0 - self.d_beta2);

        let d_t = self.i_t as f32;
        let d_b1t = self.d_beta1.powf(d_t);
        let d_b2t = self.d_beta2.powf(d_t);

        let a_m_hat = self.m_m.mapv(|x| x / (1.0 - d_b1t).max(1e-12));
        let a_v_hat = self.m_v.mapv(|x| x / (1.0 - d_b2t).max(1e-12));

        let a_denom = a_v_hat.mapv(|x| x.sqrt() + self.d_eps);
        let a_update = a_m_hat / a_denom;

        *a_params = &*a_params - &(d_lr * a_update);
    }
}

pub struct Embeddings {
    vocab: Vocab,
    // Weight matrix: [vocab_size, embedding_dim]
    w_embed: Array2<f32>,
    cached_ids: Option<Vec<usize>>,
    optimizer: Adam,
}

impl Embeddings {
    pub fn new(vocab: Vocab) -> Self {
        let i_vocab = vocab.words.len();
        let mut rng = rand::rng();

        let std = (2.0 / (i_vocab as f32).max(1.0)).sqrt();
        let normal = Normal::new(0.0, std).unwrap();

        Self {
            vocab,
            w_embed: Array2::from_shape_fn((i_vocab, EMBEDDING_DIM), |_| normal.sample(&mut rng)),
            cached_ids: None,
            optimizer: Adam::new((i_vocab, EMBEDDING_DIM)),
        }
    }
}

impl Layer for Embeddings {
    fn layer_type(&self) -> &str {
        "Embeddings"
    }

    fn forward(&mut self, a_input: &Array2<f32>) -> Array2<f32> {
        // a_input: [1, seq_len] containing token ids as f32
        let (i_rows, i_cols) = a_input.dim();
        if i_rows != 1 || i_cols == 0 {
            return Array2::zeros((0, EMBEDDING_DIM));
        }

        let mut v_ids: Vec<usize> = Vec::with_capacity(i_cols);
        for j in 0..i_cols {
            let d_val = a_input[[0, j]];
            if !d_val.is_finite() {
                v_ids.push(0);
                continue;
            }
            let i_id = d_val.max(0.0) as usize;
            v_ids.push(i_id.min(self.vocab.words.len().saturating_sub(1)));
        }
        self.cached_ids = Some(v_ids.clone());

        let mut a_out = Array2::zeros((i_cols, EMBEDDING_DIM));
        for (i_pos, &i_id) in v_ids.iter().enumerate() {
            if i_id < self.w_embed.nrows() {
                a_out.row_mut(i_pos).assign(&self.w_embed.row(i_id));
            }
        }
        a_out
    }

    fn backward(&mut self, a_grads: &Array2<f32>, d_lr: f32) -> Array2<f32> {
        let v_ids = match self.cached_ids.as_ref() {
            Some(v) => v,
            None => return Array2::zeros((1, 0)),
        };

        // Accumulate gradients into embedding matrix.
        let mut a_grad_w = Array2::zeros(self.w_embed.raw_dim());
        for (i_pos, &i_id) in v_ids.iter().enumerate() {
            if i_pos < a_grads.nrows() && i_id < a_grad_w.nrows() {
                let a_row = a_grads.row(i_pos);
                a_grad_w.row_mut(i_id).add_assign(&a_row);
            }
        }

        self.optimizer.step(&mut self.w_embed, &a_grad_w, d_lr);

        // Gradient wrt token ids is undefined, return shape [1, seq_len] zeros.
        Array2::zeros((1, v_ids.len()))
    }

    fn parameters(&self) -> usize {
        self.w_embed.len()
    }

    fn get_parameters_flat(&self) -> Vec<f32> {
        self.w_embed.iter().copied().collect()
    }

    fn set_parameters_flat(&mut self, v_params: &[f32]) -> Result<usize, String> {
        let i_needed = self.w_embed.len();
        if v_params.len() < i_needed {
            return Err("checkpoint_not_enough_params_embeddings".to_string());
        }
        for (i, d) in v_params.iter().take(i_needed).enumerate() {
            self.w_embed.as_slice_mut().unwrap()[i] = *d;
        }
        Ok(i_needed)
    }
}

pub struct LayerNorm {
    epsilon: f32,
    gamma: Array2<f32>,
    beta: Array2<f32>,

    cached_input: Option<Array2<f32>>,
    cached_mean: Option<Array2<f32>>,
    cached_std: Option<Array2<f32>>,

    optimizer_gamma: Adam,
    optimizer_beta: Adam,
}

impl LayerNorm {
    pub fn new(i_embedding_dim: usize) -> Self {
        Self {
            epsilon: 1e-5,
            gamma: Array2::ones((1, i_embedding_dim)),
            beta: Array2::zeros((1, i_embedding_dim)),
            cached_input: None,
            cached_mean: None,
            cached_std: None,
            optimizer_gamma: Adam::new((1, i_embedding_dim)),
            optimizer_beta: Adam::new((1, i_embedding_dim)),
        }
    }

    pub fn normalize(&mut self, a_input: &Array2<f32>) -> Array2<f32> {
        if a_input.nrows() == 0 || a_input.ncols() == 0 {
            return a_input.clone();
        }

        let a_mean = a_input.mean_axis(Axis(1)).unwrap().insert_axis(Axis(1));
        let a_std = a_input.std_axis(Axis(1), 0.0).insert_axis(Axis(1));

        self.cached_input = Some(a_input.clone());
        self.cached_mean = Some(a_mean.clone());
        self.cached_std = Some(a_std.clone());

        let a_normed = (a_input - &a_mean) / (&a_std + self.epsilon);
        &self.gamma * &a_normed + &self.beta
    }
}

impl Layer for LayerNorm {
    fn layer_type(&self) -> &str {
        "LayerNorm"
    }

    fn forward(&mut self, a_input: &Array2<f32>) -> Array2<f32> {
        self.normalize(a_input)
    }

    fn backward(&mut self, a_grads: &Array2<f32>, d_lr: f32) -> Array2<f32> {
        let a_input = match self.cached_input.as_ref() {
            Some(x) => x,
            None => return a_grads.clone(),
        };
        let a_mean = self.cached_mean.as_ref().unwrap();
        let a_std = self.cached_std.as_ref().unwrap();

        let a_normalized = (a_input - a_mean) / (a_std + self.epsilon);
        let d_n_features = a_input.ncols() as f32;

        let a_grad_gamma = (&a_normalized * a_grads).sum_axis(Axis(0)).insert_axis(Axis(0));
        let a_grad_beta = a_grads.sum_axis(Axis(0)).insert_axis(Axis(0));

        let a_grad_normalized = &self.gamma * a_grads;

        let a_variance = a_std * a_std + self.epsilon;
        let a_grad_var = (&a_grad_normalized * &a_normalized)
            .sum_axis(Axis(1))
            .insert_axis(Axis(1))
            * (-0.5)
            / a_variance
                .mapv(|x| x * x.sqrt())
                .mapv(|x| x.max(1e-12));

        let a_grad_mean = a_grad_normalized.sum_axis(Axis(1)).insert_axis(Axis(1)) * (-1.0)
            / (a_std + self.epsilon)
            + &a_grad_var
                * (a_input - a_mean).sum_axis(Axis(1)).insert_axis(Axis(1))
                * (-2.0)
                / d_n_features.max(1.0);

        let a_grad_input = &a_grad_normalized / (a_std + self.epsilon)
            + &a_grad_var * 2.0 * (a_input - a_mean) / d_n_features.max(1.0)
            + &a_grad_mean / d_n_features.max(1.0);

        self.optimizer_gamma.step(&mut self.gamma, &a_grad_gamma, d_lr);
        self.optimizer_beta.step(&mut self.beta, &a_grad_beta, d_lr);

        a_grad_input
    }

    fn parameters(&self) -> usize {
        self.gamma.len() + self.beta.len()
    }

    fn get_parameters_flat(&self) -> Vec<f32> {
        let mut v: Vec<f32> = Vec::with_capacity(self.gamma.len() + self.beta.len());
        v.extend(self.gamma.iter().copied());
        v.extend(self.beta.iter().copied());
        v
    }

    fn set_parameters_flat(&mut self, v_params: &[f32]) -> Result<usize, String> {
        let i_needed = self.gamma.len() + self.beta.len();
        if v_params.len() < i_needed {
            return Err("checkpoint_not_enough_params_layer_norm".to_string());
        }

        let i_gamma = self.gamma.len();
        let v_gamma = &v_params[..i_gamma];
        let v_beta = &v_params[i_gamma..i_needed];

        for (i, d) in v_gamma.iter().enumerate() {
            self.gamma.as_slice_mut().unwrap()[i] = *d;
        }
        for (i, d) in v_beta.iter().enumerate() {
            self.beta.as_slice_mut().unwrap()[i] = *d;
        }

        Ok(i_needed)
    }
}

pub struct FeedForward {
    w1: Array2<f32>,
    b1: Array2<f32>,
    w2: Array2<f32>,
    b2: Array2<f32>,

    cached_input: Option<Array2<f32>>,
    cached_hidden_pre: Option<Array2<f32>>,
    cached_hidden_post: Option<Array2<f32>>,

    opt_w1: Adam,
    opt_b1: Adam,
    opt_w2: Adam,
    opt_b2: Adam,
}

impl FeedForward {
    pub fn new(i_embedding_dim: usize, i_hidden_dim: usize) -> Self {
        let mut rng = rand::rng();

        let std_w1 = (2.0 / (i_embedding_dim as f32).max(1.0)).sqrt();
        let std_w2 = (2.0 / (i_hidden_dim as f32).max(1.0)).sqrt();
        let normal_w1 = Normal::new(0.0, std_w1).unwrap();
        let normal_w2 = Normal::new(0.0, std_w2).unwrap();

        Self {
            w1: Array2::from_shape_fn((i_embedding_dim, i_hidden_dim), |_| normal_w1.sample(&mut rng)),
            b1: Array2::zeros((1, i_hidden_dim)),
            w2: Array2::from_shape_fn((i_hidden_dim, i_embedding_dim), |_| normal_w2.sample(&mut rng)),
            b2: Array2::zeros((1, i_embedding_dim)),
            cached_input: None,
            cached_hidden_pre: None,
            cached_hidden_post: None,
            opt_w1: Adam::new((i_embedding_dim, i_hidden_dim)),
            opt_b1: Adam::new((1, i_hidden_dim)),
            opt_w2: Adam::new((i_hidden_dim, i_embedding_dim)),
            opt_b2: Adam::new((1, i_embedding_dim)),
        }
    }

    fn relu(a: &Array2<f32>) -> Array2<f32> {
        a.mapv(|x| x.max(0.0))
    }
}

impl Layer for FeedForward {
    fn layer_type(&self) -> &str {
        "FeedForward"
    }

    fn forward(&mut self, a_input: &Array2<f32>) -> Array2<f32> {
        let a_hidden_pre = a_input.dot(&self.w1) + &self.b1;
        let a_hidden_post = Self::relu(&a_hidden_pre);
        let a_out = a_hidden_post.dot(&self.w2) + &self.b2;

        self.cached_input = Some(a_input.clone());
        self.cached_hidden_pre = Some(a_hidden_pre);
        self.cached_hidden_post = Some(a_hidden_post);

        // Residual connection.
        a_out + a_input
    }

    fn backward(&mut self, a_grads: &Array2<f32>, d_lr: f32) -> Array2<f32> {
        let a_input = self.cached_input.as_ref().ok_or("cache_missing_ff_input".to_string()).unwrap();
        let a_hidden_pre = self.cached_hidden_pre.as_ref().unwrap();
        let a_hidden_post = self.cached_hidden_post.as_ref().unwrap();

        let a_grad_w2 = a_hidden_post.t().dot(a_grads);
        let a_grad_b2 = a_grads.sum_axis(Axis(0)).insert_axis(Axis(0));

        let a_grad_hidden_post = a_grads.dot(&self.w2.t());
        let a_relu_grad = a_hidden_pre.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
        let a_grad_hidden_pre = a_grad_hidden_post * a_relu_grad;

        let a_grad_w1 = a_input.t().dot(&a_grad_hidden_pre);
        let a_grad_b1 = a_grad_hidden_pre.sum_axis(Axis(0)).insert_axis(Axis(0));

        let a_grad_input_ff = a_grad_hidden_pre.dot(&self.w1.t());

        // Residual gradient.
        let a_grad_input = a_grad_input_ff + a_grads;

        self.opt_w2.step(&mut self.w2, &a_grad_w2, d_lr);
        self.opt_b2.step(&mut self.b2, &a_grad_b2, d_lr);
        self.opt_w1.step(&mut self.w1, &a_grad_w1, d_lr);
        self.opt_b1.step(&mut self.b1, &a_grad_b1, d_lr);

        a_grad_input
    }

    fn parameters(&self) -> usize {
        self.w1.len() + self.b1.len() + self.w2.len() + self.b2.len()
    }

    fn get_parameters_flat(&self) -> Vec<f32> {
        let mut v: Vec<f32> = Vec::new();
        v.extend(self.w1.iter().copied());
        v.extend(self.b1.iter().copied());
        v.extend(self.w2.iter().copied());
        v.extend(self.b2.iter().copied());
        v
    }

    fn set_parameters_flat(&mut self, v_params: &[f32]) -> Result<usize, String> {
        let i_needed = self.w1.len() + self.b1.len() + self.w2.len() + self.b2.len();
        if v_params.len() < i_needed {
            return Err("checkpoint_not_enough_params_feed_forward".to_string());
        }

        let mut i_pos: usize = 0;

        for i in 0..self.w1.len() {
            self.w1.as_slice_mut().unwrap()[i] = v_params[i_pos];
            i_pos += 1;
        }
        for i in 0..self.b1.len() {
            self.b1.as_slice_mut().unwrap()[i] = v_params[i_pos];
            i_pos += 1;
        }
        for i in 0..self.w2.len() {
            self.w2.as_slice_mut().unwrap()[i] = v_params[i_pos];
            i_pos += 1;
        }
        for i in 0..self.b2.len() {
            self.b2.as_slice_mut().unwrap()[i] = v_params[i_pos];
            i_pos += 1;
        }

        Ok(i_needed)
    }
}

pub struct SelfAttention {
    embedding_dim: usize,
    w_q: Array2<f32>,
    w_k: Array2<f32>,
    w_v: Array2<f32>,
    cached_input: Option<Array2<f32>>,
    opt_w_q: Adam,
    opt_w_k: Adam,
    opt_w_v: Adam,
}

impl SelfAttention {
    pub fn new(i_embedding_dim: usize) -> Self {
        let mut rng = rand::rng();
        let std = (2.0 / (i_embedding_dim as f32).max(1.0)).sqrt();
        let normal = Normal::new(0.0, std).unwrap();

        Self {
            embedding_dim: i_embedding_dim,
            w_q: Array2::from_shape_fn((i_embedding_dim, i_embedding_dim), |_| normal.sample(&mut rng)),
            w_k: Array2::from_shape_fn((i_embedding_dim, i_embedding_dim), |_| normal.sample(&mut rng)),
            w_v: Array2::from_shape_fn((i_embedding_dim, i_embedding_dim), |_| normal.sample(&mut rng)),
            cached_input: None,
            opt_w_q: Adam::new((i_embedding_dim, i_embedding_dim)),
            opt_w_k: Adam::new((i_embedding_dim, i_embedding_dim)),
            opt_w_v: Adam::new((i_embedding_dim, i_embedding_dim)),
        }
    }

    fn softmax(a_scores: &Array2<f32>) -> Array2<f32> {
        math::softmax_rows(a_scores)
    }

    fn compute_qkv(&self, a_input: &Array2<f32>) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
        (a_input.dot(&self.w_q), a_input.dot(&self.w_k), a_input.dot(&self.w_v))
    }

    fn attention(&self, a_q: &Array2<f32>, a_k: &Array2<f32>, a_v: &Array2<f32>) -> Array2<f32> {
        let d_scale = (self.embedding_dim as f32).sqrt().max(1e-12);
        let mut a_scores = a_q.dot(&a_k.t()) / d_scale;

        // Causal mask.
        let i_seq_len = a_scores.nrows();
        for i in 0..i_seq_len {
            for j in (i + 1)..i_seq_len {
                a_scores[[i, j]] = f32::NEG_INFINITY;
            }
        }

        let a_weights = Self::softmax(&a_scores);
        a_weights.dot(a_v)
    }

    fn softmax_backward(a_softmax: &Array2<f32>, a_grad_out: &Array2<f32>) -> Array2<f32> {
        // Row-wise Jacobian-vector product for softmax.
        let mut a_grad_in = a_softmax.clone();
        for i in 0..a_softmax.nrows() {
            let a_row = a_softmax.row(i);
            let a_grow = a_grad_out.row(i);

            let d_dot: f32 = a_row.iter().zip(a_grow.iter()).map(|(&y, &dy)| y * dy).sum();
            for j in 0..a_softmax.ncols() {
                a_grad_in[[i, j]] = a_softmax[[i, j]] * (a_grad_out[[i, j]] - d_dot);
            }
        }
        a_grad_in
    }
}

impl Layer for SelfAttention {
    fn layer_type(&self) -> &str {
        "SelfAttention"
    }

    fn forward(&mut self, a_input: &Array2<f32>) -> Array2<f32> {
        self.cached_input = Some(a_input.clone());
        let (a_q, a_k, a_v) = self.compute_qkv(a_input);
        let a_attn = self.attention(&a_q, &a_k, &a_v);
        a_attn + a_input
    }

    fn backward(&mut self, a_grads: &Array2<f32>, d_lr: f32) -> Array2<f32> {
        let a_input = self.cached_input.as_ref().expect("forward must be run first");
        let (a_q, a_k, a_v) = self.compute_qkv(a_input);

        let d_scale = (self.embedding_dim as f32).sqrt().max(1e-12);
        let mut a_scores = a_q.dot(&a_k.t()) / d_scale;

        let i_seq_len = a_scores.nrows();
        for i in 0..i_seq_len {
            for j in (i + 1)..i_seq_len {
                a_scores[[i, j]] = f32::NEG_INFINITY;
            }
        }

        let a_weights = Self::softmax(&a_scores);

        let a_grad_weights = a_grads.dot(&a_v.t());
        let a_grad_v = a_weights.t().dot(a_grads);

        let a_grad_scores = Self::softmax_backward(&a_weights, &a_grad_weights);

        let a_grad_q = a_grad_scores.dot(&a_k) / d_scale;
        let a_grad_k = a_grad_scores.t().dot(&a_q) / d_scale;

        let a_grad_w_q = a_input.t().dot(&a_grad_q);
        let a_grad_w_k = a_input.t().dot(&a_grad_k);
        let a_grad_w_v = a_input.t().dot(&a_grad_v);

        let a_grad_in_attn =
            a_grad_q.dot(&self.w_q.t()) + a_grad_k.dot(&self.w_k.t()) + a_grad_v.dot(&self.w_v.t());

        let a_grad_in = a_grad_in_attn + a_grads;

        self.opt_w_q.step(&mut self.w_q, &a_grad_w_q, d_lr);
        self.opt_w_k.step(&mut self.w_k, &a_grad_w_k, d_lr);
        self.opt_w_v.step(&mut self.w_v, &a_grad_w_v, d_lr);

        a_grad_in
    }

    fn parameters(&self) -> usize {
        self.w_q.len() + self.w_k.len() + self.w_v.len()
    }

    fn get_parameters_flat(&self) -> Vec<f32> {
        let mut v: Vec<f32> = Vec::new();
        v.extend(self.w_q.iter().copied());
        v.extend(self.w_k.iter().copied());
        v.extend(self.w_v.iter().copied());
        v
    }

    fn set_parameters_flat(&mut self, v_params: &[f32]) -> Result<usize, String> {
        let i_needed = self.w_q.len() + self.w_k.len() + self.w_v.len();
        if v_params.len() < i_needed {
            return Err("checkpoint_not_enough_params_self_attention".to_string());
        }

        let mut i_pos: usize = 0;

        for i in 0..self.w_q.len() {
            self.w_q.as_slice_mut().unwrap()[i] = v_params[i_pos];
            i_pos += 1;
        }
        for i in 0..self.w_k.len() {
            self.w_k.as_slice_mut().unwrap()[i] = v_params[i_pos];
            i_pos += 1;
        }
        for i in 0..self.w_v.len() {
            self.w_v.as_slice_mut().unwrap()[i] = v_params[i_pos];
            i_pos += 1;
        }

        Ok(i_needed)
    }
}

// ---------------------------
// MultiHeadSelfAttention
// ---------------------------
pub struct MultiHeadSelfAttention {
    i_embedding_dim: usize,
    i_num_heads: usize,
    i_head_dim: usize,

    // Projection matrices: [embedding_dim, embedding_dim]
    w_q: Array2<f32>,
    w_k: Array2<f32>,
    w_v: Array2<f32>,
    w_o: Array2<f32>,

    // Cache for backward
    cached_input: Option<Array2<f32>>,
    cached_q_all: Option<Array2<f32>>,
    cached_k_all: Option<Array2<f32>>,
    cached_v_all: Option<Array2<f32>>,
    cached_concat: Option<Array2<f32>>,
    cached_weights: Option<Vec<Array2<f32>>>, // per head: [seq, seq]

    opt_w_q: Adam,
    opt_w_k: Adam,
    opt_w_v: Adam,
    opt_w_o: Adam,
}

impl MultiHeadSelfAttention {
    pub fn new(i_embedding_dim: usize, i_num_heads: usize) -> Self {
        // Validation, fail early.
        if i_embedding_dim == 0 {
            panic!("embedding_dim_must_be_positive");
        }
        if i_num_heads == 0 {
            panic!("num_heads_must_be_positive");
        }
        if i_embedding_dim % i_num_heads != 0 {
            panic!("embedding_dim_must_be_divisible_by_num_heads");
        }

        let i_head_dim = i_embedding_dim / i_num_heads;

        let mut rng = rand::rng();
        let d_std = (2.0 / (i_embedding_dim as f32).max(1.0)).sqrt();
        let normal = Normal::new(0.0, d_std).unwrap();

        Self {
            i_embedding_dim,
            i_num_heads,
            i_head_dim,

            w_q: Array2::from_shape_fn((i_embedding_dim, i_embedding_dim), |_| normal.sample(&mut rng)),
            w_k: Array2::from_shape_fn((i_embedding_dim, i_embedding_dim), |_| normal.sample(&mut rng)),
            w_v: Array2::from_shape_fn((i_embedding_dim, i_embedding_dim), |_| normal.sample(&mut rng)),
            w_o: Array2::from_shape_fn((i_embedding_dim, i_embedding_dim), |_| normal.sample(&mut rng)),

            cached_input: None,
            cached_q_all: None,
            cached_k_all: None,
            cached_v_all: None,
            cached_concat: None,
            cached_weights: None,

            opt_w_q: Adam::new((i_embedding_dim, i_embedding_dim)),
            opt_w_k: Adam::new((i_embedding_dim, i_embedding_dim)),
            opt_w_v: Adam::new((i_embedding_dim, i_embedding_dim)),
            opt_w_o: Adam::new((i_embedding_dim, i_embedding_dim)),
        }
    }

    fn softmax(a_scores: &Array2<f32>) -> Array2<f32> {
        math::softmax_rows(a_scores)
    }

    fn softmax_backward(a_softmax: &Array2<f32>, a_grad_out: &Array2<f32>) -> Array2<f32> {
        // Row-wise Jacobian-vector product for softmax: dS = S * (dY - sum(dY * S))
        let mut a_grad_in = a_softmax.clone();
        for i in 0..a_softmax.nrows() {
            let a_row = a_softmax.row(i);
            let a_grow = a_grad_out.row(i);

            let d_dot: f32 = a_row
                .iter()
                .zip(a_grow.iter())
                .map(|(&y, &dy)| y * dy)
                .sum();

            for j in 0..a_softmax.ncols() {
                a_grad_in[[i, j]] = a_softmax[[i, j]] * (a_grad_out[[i, j]] - d_dot);
            }
        }
        a_grad_in
    }

    fn split_heads(&self, a_x: &Array2<f32>) -> Result<Vec<Array2<f32>>, String> {
        // a_x: [seq, embedding] -> vec heads [seq, head_dim]
        if a_x.ncols() != self.i_embedding_dim {
            return Err("mhsa_split_heads_dim_mismatch".to_string());
        }

        let i_seq_len = a_x.nrows();
        let mut v_heads: Vec<Array2<f32>> = Vec::with_capacity(self.i_num_heads);

        for i_h in 0..self.i_num_heads {
            let i_start = i_h * self.i_head_dim;
            let i_end = i_start + self.i_head_dim;
            let a_view = a_x.slice(ndarray::s![.., i_start..i_end]).to_owned();

            if a_view.nrows() != i_seq_len || a_view.ncols() != self.i_head_dim {
                return Err("mhsa_split_heads_slice_error".to_string());
            }
            v_heads.push(a_view);
        }

        Ok(v_heads)
    }

    fn concat_heads(&self, v_heads: &[Array2<f32>]) -> Result<Array2<f32>, String> {
        if v_heads.len() != self.i_num_heads {
            return Err("mhsa_concat_heads_count_mismatch".to_string());
        }

        let i_seq_len = v_heads[0].nrows();
        for a_h in v_heads.iter() {
            if a_h.nrows() != i_seq_len || a_h.ncols() != self.i_head_dim {
                return Err("mhsa_concat_heads_shape_mismatch".to_string());
            }
        }

        let mut a_out = Array2::<f32>::zeros((i_seq_len, self.i_embedding_dim));
        for i_h in 0..self.i_num_heads {
            let i_start = i_h * self.i_head_dim;
            let i_end = i_start + self.i_head_dim;
            let mut a_slice = a_out.slice_mut(ndarray::s![.., i_start..i_end]);
            a_slice.assign(&v_heads[i_h]);
        }

        Ok(a_out)
    }

    fn apply_causal_mask_inplace(a_scores: &mut Array2<f32>) {
        let i_seq_len = a_scores.nrows();
        for i in 0..i_seq_len {
            for j in (i + 1)..i_seq_len {
                a_scores[[i, j]] = f32::NEG_INFINITY;
            }
        }
    }

    fn attention_head_forward(
        &self,
        a_q: &Array2<f32>,
        a_k: &Array2<f32>,
        a_v: &Array2<f32>,
    ) -> (Array2<f32>, Array2<f32>) {
        // Returns (head_out, weights)
        let d_scale = (self.i_head_dim as f32).sqrt().max(1e-12);

        let mut a_scores = a_q.dot(&a_k.t()) / d_scale;
        Self::apply_causal_mask_inplace(&mut a_scores);

        let a_weights = Self::softmax(&a_scores);
        let a_out = a_weights.dot(a_v);

        (a_out, a_weights)
    }
}

impl Layer for MultiHeadSelfAttention {
    fn layer_type(&self) -> &str {
        "MultiHeadSelfAttention"
    }

    fn forward(&mut self, a_input: &Array2<f32>) -> Array2<f32> {
        // a_input: [seq, embedding_dim]
        if a_input.nrows() == 0 || a_input.ncols() == 0 {
            return a_input.clone();
        }
        if a_input.ncols() != self.i_embedding_dim {
            return Array2::<f32>::zeros((0, 0));
        }

        // Cache input.
        self.cached_input = Some(a_input.clone());

        // Project.
        let a_q_all = a_input.dot(&self.w_q);
        let a_k_all = a_input.dot(&self.w_k);
        let a_v_all = a_input.dot(&self.w_v);

        self.cached_q_all = Some(a_q_all.clone());
        self.cached_k_all = Some(a_k_all.clone());
        self.cached_v_all = Some(a_v_all.clone());

        // Split heads.
        let v_q = match self.split_heads(&a_q_all) {
            Ok(v) => v,
            Err(_) => return Array2::<f32>::zeros((0, 0)),
        };
        let v_k = match self.split_heads(&a_k_all) {
            Ok(v) => v,
            Err(_) => return Array2::<f32>::zeros((0, 0)),
        };
        let v_v = match self.split_heads(&a_v_all) {
            Ok(v) => v,
            Err(_) => return Array2::<f32>::zeros((0, 0)),
        };

        // Per-head attention.
        let mut v_head_out: Vec<Array2<f32>> = Vec::with_capacity(self.i_num_heads);
        let mut v_weights: Vec<Array2<f32>> = Vec::with_capacity(self.i_num_heads);

        for i_h in 0..self.i_num_heads {
            let (a_h_out, a_w) = self.attention_head_forward(&v_q[i_h], &v_k[i_h], &v_v[i_h]);
            v_head_out.push(a_h_out);
            v_weights.push(a_w);
        }

        self.cached_weights = Some(v_weights);

        // Concat and output projection.
        let a_concat = match self.concat_heads(&v_head_out) {
            Ok(a) => a,
            Err(_) => return Array2::<f32>::zeros((0, 0)),
        };
        self.cached_concat = Some(a_concat.clone());

        let a_proj = a_concat.dot(&self.w_o);

        // Residual.
        a_proj + a_input
    }

    fn backward(&mut self, a_grads: &Array2<f32>, d_lr: f32) -> Array2<f32> {
        // Correct backward for MHSA with causal scaled dot product attention.
        // Shapes:
        // a_input: [seq, emb]
        // a_grads: [seq, emb] gradient wrt output of (proj + residual)
        //
        // Output:
        // grad wrt input: [seq, emb]

        // Safety checks.
        if !d_lr.is_finite() || d_lr <= 0.0 {
            return a_grads.clone();
        }

        let a_input = match self.cached_input.as_ref() {
            Some(x) => x,
            None => return a_grads.clone(),
        };
        let a_q_all = match self.cached_q_all.as_ref() {
            Some(x) => x,
            None => return a_grads.clone(),
        };
        let a_k_all = match self.cached_k_all.as_ref() {
            Some(x) => x,
            None => return a_grads.clone(),
        };
        let a_v_all = match self.cached_v_all.as_ref() {
            Some(x) => x,
            None => return a_grads.clone(),
        };
        let a_concat = match self.cached_concat.as_ref() {
            Some(x) => x,
            None => return a_grads.clone(),
        };
        let v_weights = match self.cached_weights.as_ref() {
            Some(v) => v,
            None => return a_grads.clone(),
        };

        if a_input.raw_dim() != a_grads.raw_dim() {
            return a_grads.clone();
        }

        // Residual path: output = proj + input
        // So grad splits: dL/dproj = a_grads, dL/dinput_residual = a_grads
        let a_grad_proj = a_grads;

        // Output projection: proj = concat * w_o
        let a_grad_w_o = a_concat.t().dot(a_grad_proj);
        let a_grad_concat = a_grad_proj.dot(&self.w_o.t()); // [seq, emb]

        // Split grad_concat to heads.
        let v_grad_head_out = match self.split_heads(&a_grad_concat) {
            Ok(v) => v,
            Err(_) => return a_grads.clone(),
        };

        // Split q/k/v to heads.
        let v_q = match self.split_heads(a_q_all) {
            Ok(v) => v,
            Err(_) => return a_grads.clone(),
        };
        let v_k = match self.split_heads(a_k_all) {
            Ok(v) => v,
            Err(_) => return a_grads.clone(),
        };
        let v_v = match self.split_heads(a_v_all) {
            Ok(v) => v,
            Err(_) => return a_grads.clone(),
        };

        // Accumulate per-head grads.
        let mut v_grad_q: Vec<Array2<f32>> = Vec::with_capacity(self.i_num_heads);
        let mut v_grad_k: Vec<Array2<f32>> = Vec::with_capacity(self.i_num_heads);
        let mut v_grad_v: Vec<Array2<f32>> = Vec::with_capacity(self.i_num_heads);

        let d_scale = (self.i_head_dim as f32).sqrt().max(1e-12);
        let i_seq_len = a_input.nrows();

        for i_h in 0..self.i_num_heads {
            let a_q = &v_q[i_h];
            let a_k = &v_k[i_h];
            let a_v = &v_v[i_h];
            let a_w = &v_weights[i_h]; // [seq, seq]
            let a_grad_h_out = &v_grad_head_out[i_h]; // [seq, head_dim]

            // head_out = W * V
            // dW = dO * V^T
            // dV = W^T * dO
            let a_grad_w = a_grad_h_out.dot(&a_v.t()); // [seq, seq]
            let a_grad_v_h = a_w.t().dot(a_grad_h_out); // [seq, head_dim]

            // W = softmax(scores)
            // dScores = softmax_backward(W, dW)
            let mut a_grad_scores = Self::softmax_backward(a_w, &a_grad_w); // [seq, seq]

            // Apply causal mask gradient: masked positions are constant -inf in forward,
            // therefore treat gradients there as zero to avoid spurious updates.
            for i in 0..i_seq_len {
                for j in (i + 1)..i_seq_len {
                    a_grad_scores[[i, j]] = 0.0;
                }
            }

            // scores = (Q K^T) / scale
            // dQ = dScores * K / scale
            // dK = dScores^T * Q / scale
            let a_grad_q_h = a_grad_scores.dot(a_k) / d_scale;
            let a_grad_k_h = a_grad_scores.t().dot(a_q) / d_scale;

            v_grad_q.push(a_grad_q_h);
            v_grad_k.push(a_grad_k_h);
            v_grad_v.push(a_grad_v_h);
        }

        // Concat per-head grads to embedding space.
        let a_grad_q_all = match self.concat_heads(&v_grad_q) {
            Ok(a) => a,
            Err(_) => return a_grads.clone(),
        };
        let a_grad_k_all = match self.concat_heads(&v_grad_k) {
            Ok(a) => a,
            Err(_) => return a_grads.clone(),
        };
        let a_grad_v_all = match self.concat_heads(&v_grad_v) {
            Ok(a) => a,
            Err(_) => return a_grads.clone(),
        };

        // Linear projections:
        // Q = X * w_q, K = X * w_k, V = X * w_v
        // dW = X^T * dY
        // dX = dY * W^T
        let a_grad_w_q = a_input.t().dot(&a_grad_q_all);
        let a_grad_w_k = a_input.t().dot(&a_grad_k_all);
        let a_grad_w_v = a_input.t().dot(&a_grad_v_all);

        let a_grad_x_from_q = a_grad_q_all.dot(&self.w_q.t());
        let a_grad_x_from_k = a_grad_k_all.dot(&self.w_k.t());
        let a_grad_x_from_v = a_grad_v_all.dot(&self.w_v.t());

        // Total grad wrt input is sum of:
        // - residual path
        // - projections paths via q/k/v
        let a_grad_input_total = a_grads.clone() + a_grad_x_from_q + a_grad_x_from_k + a_grad_x_from_v;

        // Update parameters.
        self.opt_w_o.step(&mut self.w_o, &a_grad_w_o, d_lr);
        self.opt_w_q.step(&mut self.w_q, &a_grad_w_q, d_lr);
        self.opt_w_k.step(&mut self.w_k, &a_grad_w_k, d_lr);
        self.opt_w_v.step(&mut self.w_v, &a_grad_w_v, d_lr);

        a_grad_input_total
    }

    fn parameters(&self) -> usize {
        self.w_q.len() + self.w_k.len() + self.w_v.len() + self.w_o.len()
    }

    fn get_parameters_flat(&self) -> Vec<f32> {
        let mut v: Vec<f32> = Vec::new();
        v.extend(self.w_q.iter().copied());
        v.extend(self.w_k.iter().copied());
        v.extend(self.w_v.iter().copied());
        v.extend(self.w_o.iter().copied());
        v
    }

    fn set_parameters_flat(&mut self, v_params: &[f32]) -> Result<usize, String> {
        let i_needed = self.w_q.len() + self.w_k.len() + self.w_v.len() + self.w_o.len();
        if v_params.len() < i_needed {
            return Err("checkpoint_not_enough_params_multi_head_self_attention".to_string());
        }

        let mut i_pos: usize = 0;

        for i in 0..self.w_q.len() {
            self.w_q.as_slice_mut().unwrap()[i] = v_params[i_pos];
            i_pos += 1;
        }
        for i in 0..self.w_k.len() {
            self.w_k.as_slice_mut().unwrap()[i] = v_params[i_pos];
            i_pos += 1;
        }
        for i in 0..self.w_v.len() {
            self.w_v.as_slice_mut().unwrap()[i] = v_params[i_pos];
            i_pos += 1;
        }
        for i in 0..self.w_o.len() {
            self.w_o.as_slice_mut().unwrap()[i] = v_params[i_pos];
            i_pos += 1;
        }

        Ok(i_needed)
    }
}

// ---------------------------
// TransformerBlock update
// ---------------------------
pub struct TransformerBlock {
    attention: MultiHeadSelfAttention,
    feed_forward: FeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
}

impl TransformerBlock {
    pub fn new(i_embedding_dim: usize, i_hidden_dim: usize) -> Self {
        // Expert default: embedding_dim 128 is divisible by 4.
        // If EMBEDDING_DIM changes, new() panics early if invalid.
        let i_num_heads: usize = 4;

        Self {
            attention: MultiHeadSelfAttention::new(i_embedding_dim, i_num_heads),
            feed_forward: FeedForward::new(i_embedding_dim, i_hidden_dim),
            norm1: LayerNorm::new(i_embedding_dim),
            norm2: LayerNorm::new(i_embedding_dim),
        }
    }
}

impl Layer for TransformerBlock {
    fn layer_type(&self) -> &str {
        "TransformerBlock"
    }

    fn forward(&mut self, a_input: &Array2<f32>) -> Array2<f32> {
        let a_attn = self.attention.forward(a_input);
        let a_n1 = self.norm1.normalize(&a_attn);
        let a_ff = self.feed_forward.forward(&a_n1);
        self.norm2.normalize(&a_ff)
    }

    fn backward(&mut self, a_grads: &Array2<f32>, d_lr: f32) -> Array2<f32> {
        let a_g2 = self.norm2.backward(a_grads, d_lr);
        let a_gff = self.feed_forward.backward(&a_g2, d_lr);
        let a_g1 = self.norm1.backward(&a_gff, d_lr);
        self.attention.backward(&a_g1, d_lr)
    }

    fn parameters(&self) -> usize {
        self.attention.parameters()
            + self.feed_forward.parameters()
            + self.norm1.parameters()
            + self.norm2.parameters()
    }

    fn get_parameters_flat(&self) -> Vec<f32> {
        let mut v: Vec<f32> = Vec::new();
        v.extend(self.attention.get_parameters_flat());
        v.extend(self.norm1.get_parameters_flat());
        v.extend(self.feed_forward.get_parameters_flat());
        v.extend(self.norm2.get_parameters_flat());
        v
    }

    fn set_parameters_flat(&mut self, v_params: &[f32]) -> Result<usize, String> {
        let mut i_used: usize = 0;

        let i1 = self.attention.set_parameters_flat(&v_params[i_used..])?;
        i_used += i1;

        let i2 = self.norm1.set_parameters_flat(&v_params[i_used..])?;
        i_used += i2;

        let i3 = self.feed_forward.set_parameters_flat(&v_params[i_used..])?;
        i_used += i3;

        let i4 = self.norm2.set_parameters_flat(&v_params[i_used..])?;
        i_used += i4;

        Ok(i_used)
    }
}

pub struct OutputProjection {
    w_out: Array2<f32>,
    b_out: Array2<f32>,
    optimizer: Adam,
    cached_input: Option<Array2<f32>>,
}

impl OutputProjection {
    pub fn new(i_embedding_dim: usize, i_vocab_size: usize) -> Self {
        let mut rng = rand::rng();
        let std = (2.0 / (i_embedding_dim as f32).max(1.0)).sqrt();
        let normal = Normal::new(0.0, std).unwrap();

        Self {
            w_out: Array2::from_shape_fn((i_embedding_dim, i_vocab_size), |_| normal.sample(&mut rng)),
            b_out: Array2::zeros((1, i_vocab_size)),
            optimizer: Adam::new((i_embedding_dim, i_vocab_size)),
            cached_input: None,
        }
    }
}

impl Layer for OutputProjection {
    fn layer_type(&self) -> &str {
        "OutputProjection"
    }

    fn forward(&mut self, a_input: &Array2<f32>) -> Array2<f32> {
        self.cached_input = Some(a_input.clone());
        a_input.dot(&self.w_out) + &self.b_out
    }

    fn backward(&mut self, a_grads: &Array2<f32>, d_lr: f32) -> Array2<f32> {
        let a_input = self.cached_input.as_ref().expect("forward must be run first");
        let a_grad_w = a_input.t().dot(a_grads);
        let a_grad_b = a_grads.mean_axis(Axis(0)).unwrap().insert_axis(Axis(0));
        let a_grad_in = a_grads.dot(&self.w_out.t());

        self.optimizer.step(&mut self.w_out, &a_grad_w, d_lr);
        self.b_out = &self.b_out - &(d_lr * a_grad_b);

        a_grad_in
    }

    fn parameters(&self) -> usize {
        self.w_out.len() + self.b_out.len()
    }

    fn get_parameters_flat(&self) -> Vec<f32> {
        let mut v: Vec<f32> = Vec::new();
        v.extend(self.w_out.iter().copied());
        v.extend(self.b_out.iter().copied());
        v
    }

    fn set_parameters_flat(&mut self, v_params: &[f32]) -> Result<usize, String> {
        let i_needed = self.w_out.len() + self.b_out.len();
        if v_params.len() < i_needed {
            return Err("checkpoint_not_enough_params_output_projection".to_string());
        }

        let mut i_pos: usize = 0;
        for i in 0..self.w_out.len() {
            self.w_out.as_slice_mut().unwrap()[i] = v_params[i_pos];
            i_pos += 1;
        }
        for i in 0..self.b_out.len() {
            self.b_out.as_slice_mut().unwrap()[i] = v_params[i_pos];
            i_pos += 1;
        }

        Ok(i_needed)
    }
}

#[allow(clippy::upper_case_acronyms)]
pub struct Llm {
    pub vocab: Vocab,
    pub network: Vec<Box<dyn Layer>>,
    pub bpe_tokenizer: Option<BpeTokenizer>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LlmCheckpoint {
    pub s_magic: String,
    pub s_version: String,
    pub i_max_seq_len: usize,
    pub i_embedding_dim: usize,
    pub i_hidden_dim: usize,
    pub tokenizer: BpeTokenizerCheckpoint,
    pub v_params: Vec<f32>,
}

impl LlmCheckpoint {
    pub fn new(
        tokenizer: BpeTokenizerCheckpoint,
        v_params: Vec<f32>,
        i_max_seq_len: usize,
        i_embedding_dim: usize,
        i_hidden_dim: usize,
    ) -> Self {
        Self {
            s_magic: "EXCHAT_LLM_CHECKPOINT".to_string(),
            s_version: "1".to_string(),
            i_max_seq_len,
            i_embedding_dim,
            i_hidden_dim,
            tokenizer,
            v_params,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.s_magic != "EXCHAT_LLM_CHECKPOINT" {
            return Err("checkpoint_magic_mismatch".to_string());
        }
        if self.s_version != "1" {
            return Err("checkpoint_version_unsupported".to_string());
        }
        if self.i_max_seq_len != MAX_SEQ_LEN {
            return Err("checkpoint_max_seq_len_mismatch".to_string());
        }
        if self.i_embedding_dim != EMBEDDING_DIM {
            return Err("checkpoint_embedding_dim_mismatch".to_string());
        }
        if self.i_hidden_dim != HIDDEN_DIM {
            return Err("checkpoint_hidden_dim_mismatch".to_string());
        }
        if self.v_params.is_empty() {
            return Err("checkpoint_empty_params".to_string());
        }
        Ok(())
    }
}

impl Llm {
    pub fn new(vocab: Vocab, network: Vec<Box<dyn Layer>>) -> Self {
        Self {
            vocab,
            network,
            bpe_tokenizer: None,
        }
    }

    pub fn set_bpe_tokenizer(&mut self, bpe_tokenizer: BpeTokenizer) {
        self.vocab = bpe_tokenizer.vocab.clone();
        self.bpe_tokenizer = Some(bpe_tokenizer);
    }

    pub fn network_description(&self) -> String {
        self.network
            .iter()
            .map(|l| l.layer_type())
            .collect::<Vec<&str>>()
            .join(", ")
    }

    pub fn total_parameters(&self) -> usize {
        self.network.iter().map(|l| l.parameters()).sum()
    }

    pub fn decode_ids(&self, v_ids: &[usize]) -> String {
        if let Some(tok) = &self.bpe_tokenizer {
            return tok.decode_ids(v_ids);
        }
        utils::decode_via_vocab_ascii(&self.vocab, v_ids)
    }

    pub fn tokenize(&self, s_text: &str) -> Result<Vec<usize>, String> {
        let tok = self
            .bpe_tokenizer
            .as_ref()
            .ok_or_else(|| "tokenizer_not_set".to_string())?;

        let mut v_ids = tok.encode_text(s_text, false);
        if v_ids.is_empty() {
            return Err("tokenizer_returned_empty".to_string());
        }
        if v_ids.len() > MAX_SEQ_LEN {
            v_ids.truncate(MAX_SEQ_LEN);
        }
        Ok(v_ids)
    }

    // Loads checkpoint and returns a fresh Llm with correct dimensions.
    // This prevents vocab_size mismatches that change OutputProjection shape.
    pub fn load_checkpoint_rebuild(s_path: &str) -> Result<Llm, String> {
        if s_path.trim().is_empty() {
            return Err("checkpoint_path_empty".to_string());
        }

        let s_json =
            std::fs::read_to_string(s_path).map_err(|_| "checkpoint_read_error".to_string())?;
        let cp: LlmCheckpoint = crate::utils::checkpoint_from_json_ascii(&s_json)?;
        cp.validate()?;

        let bpe = BpeTokenizer::from_checkpoint(&cp.tokenizer)?;

        // Build new model using checkpoint tokenizer vocab size.
        let vocab = bpe.vocab.clone();
        let embeddings = crate::layer::Embeddings::new(vocab.clone());
        let block1 = crate::layer::TransformerBlock::new(crate::EMBEDDING_DIM, crate::HIDDEN_DIM);
        let block2 = crate::layer::TransformerBlock::new(crate::EMBEDDING_DIM, crate::HIDDEN_DIM);
        let block3 = crate::layer::TransformerBlock::new(crate::EMBEDDING_DIM, crate::HIDDEN_DIM);
        let out = crate::layer::OutputProjection::new(crate::EMBEDDING_DIM, vocab.words.len());

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

        let i_expected: usize = llm
            .network
            .iter()
            .map(|l| l.get_parameters_flat().len())
            .sum();

        if i_expected != cp.v_params.len() {
            return Err("checkpoint_param_count_mismatch".to_string());
        }

        llm.assign_all_parameters_flat(&cp.v_params)?;
        Ok(llm)
    }

    pub fn predict(&mut self, s_text: &str) -> Result<String, String> {
        let v_out_ids = self.forward_generate(s_text)?;
        Ok(self.decode_ids(&v_out_ids))
    }

    fn forward_generate(&mut self, s_text: &str) -> Result<Vec<usize>, String> {
        let mut v_context = self.tokenize(s_text)?;
        let mut v_generated: Vec<usize> = Vec::new();

        if v_context.len() >= MAX_SEQ_LEN {
            return Ok(v_generated);
        }

        let opt_eos = self.vocab.encode(S_EOS);

        for _ in 0..(MAX_SEQ_LEN - v_context.len()) {
            let a_token_input: Array2<f32> = Array2::from_shape_vec(
                (1, v_context.len()),
                v_context.iter().map(|&x| x as f32).collect::<Vec<f32>>(),
            )
            .map_err(|_| "shape_error_token_input".to_string())?;

            let mut a_act = a_token_input;
            for layer in self.network.iter_mut() {
                a_act = layer.forward(&a_act);
            }

            let a_logits = a_act;
            if a_logits.nrows() == 0 || a_logits.ncols() == 0 {
                return Err("empty_logits".to_string());
            }

            let a_last = a_logits
                .row(a_logits.nrows().saturating_sub(1))
                .to_owned()
                .insert_axis(Axis(0));

            let a_probs = math::softmax_rows(&a_last);
            let v_tokens = Self::greedy_decode(&a_probs);
            if v_tokens.is_empty() {
                return Err("decode_empty".to_string());
            }

            let i_next = *v_tokens.last().unwrap();
            v_generated.push(i_next);
            v_context.push(i_next);

            if let Some(i_eos) = opt_eos {
                if i_next == i_eos {
                    break;
                }
            }
        }

        Ok(v_generated)
    }

    pub fn train(&mut self, v_data: Vec<&str>, i_epochs: usize, d_lr: f32) -> Result<(), String> {
        if v_data.is_empty() || i_epochs == 0 {
            return Err("invalid_training_args".to_string());
        }
        if !d_lr.is_finite() || d_lr <= 0.0 {
            return Err("invalid_learning_rate".to_string());
        }
        if self.bpe_tokenizer.is_none() {
            return Err("tokenizer_not_set".to_string());
        }

        let v_tokenized_data: Vec<Vec<usize>> = v_data
            .iter()
            .map(|s| self.tokenize(s))
            .collect::<Result<Vec<Vec<usize>>, String>>()?
            .into_iter()
            .filter(|v| v.len() >= 2)
            .collect();

        if v_tokenized_data.is_empty() {
            return Err("no_tokenized_rows".to_string());
        }

        // History:
        // - 2026-02-01: Consolidated training loop into layer.rs within Llm::train while keeping Result based error handling.
        for i_epoch in 0..i_epochs {
            let mut d_total_loss: f32 = 0.0;
            let mut i_used_rows: usize = 0;

            for v_row in v_tokenized_data.iter() {
                let v_input_ids = &v_row[..v_row.len() - 1];
                let v_target_ids = &v_row[1..];

                let mut a_input: Array2<f32> = Array2::zeros((1, v_input_ids.len()));
                let a_row = Array1::from_iter(v_input_ids.iter().map(|&x| x as f32));
                a_input.row_mut(0).assign(&a_row);

                let mut a_act = a_input;
                for layer in self.network.iter_mut() {
                    a_act = layer.forward(&a_act);
                }

                let a_logits = a_act;
                if a_logits.nrows() == 0 || a_logits.ncols() == 0 {
                    continue;
                }

                let a_probs = math::softmax_rows(&a_logits);
                d_total_loss += math::cross_entropy_loss_step(&a_probs, v_target_ids);

                let mut a_grads = math::compute_gradients_step(&a_probs, v_target_ids);
                math::clip_gradients(&mut a_grads, 5.0);

                for layer in self.network.iter_mut().rev() {
                    a_grads = layer.backward(&a_grads, d_lr);
                }

                i_used_rows += 1;
            }

            let d_avg_loss = if i_used_rows == 0 {
                0.0
            } else {
                d_total_loss / (i_used_rows as f32).max(1.0)
            };

            println!("Epoch {}: Loss = {:.4}", i_epoch, d_avg_loss);
        }

        Ok(())
    }

    fn greedy_decode(a_probs: &Array2<f32>) -> Vec<usize> {
        a_probs
            .map_axis(Axis(1), |a_row| {
                a_row
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            })
            .to_vec()
    }

    fn collect_all_parameters_flat(&self) -> Vec<f32> {
        let mut v_params: Vec<f32> = Vec::new();
        for layer in self.network.iter() {
            v_params.extend(layer.get_parameters_flat());
        }
        v_params
    }

    fn assign_all_parameters_flat(&mut self, v_params: &[f32]) -> Result<(), String> {
        let mut i_pos: usize = 0;
        for layer in self.network.iter_mut() {
            let i_used = layer.set_parameters_flat(&v_params[i_pos..])?;
            i_pos += i_used;
        }

        if i_pos != v_params.len() {
            // Extra params are treated as mismatch.
            return Err("checkpoint_params_length_mismatch".to_string());
        }

        Ok(())
    }

    pub fn save_checkpoint(&self, s_path: &str) -> Result<(), String> {
        if s_path.trim().is_empty() {
            return Err("checkpoint_path_empty".to_string());
        }

        let tok = self
            .bpe_tokenizer
            .as_ref()
            .ok_or_else(|| "tokenizer_not_set".to_string())?;

        let tokenizer_cp = tok.to_checkpoint();
        let v_params = self.collect_all_parameters_flat();

        let cp = LlmCheckpoint::new(tokenizer_cp, v_params, MAX_SEQ_LEN, EMBEDDING_DIM, HIDDEN_DIM);

        let s_json = crate::utils::checkpoint_to_json_ascii(&cp)?;
        crate::utils::write_file_atomic_ascii(s_path, &s_json)?;
        Ok(())
    }

    pub fn load_checkpoint(&mut self, s_path: &str) -> Result<(), String> {
        if s_path.trim().is_empty() {
            return Err("checkpoint_path_empty".to_string());
        }

        let s_json = std::fs::read_to_string(s_path).map_err(|_| "checkpoint_read_error".to_string())?;
        let cp = crate::utils::checkpoint_from_json_ascii(&s_json)?;

        cp.validate()?;

        let tok = BpeTokenizer::from_checkpoint(&cp.tokenizer)?;
        self.set_bpe_tokenizer(tok);

        // Verify expected parameter count.
        let i_expected: usize = self.network.iter().map(|l| l.get_parameters_flat().len()).sum();
        if i_expected != cp.v_params.len() {
            return Err("checkpoint_param_count_mismatch".to_string());
        }

        self.assign_all_parameters_flat(&cp.v_params)?;
        Ok(())
    }
}
