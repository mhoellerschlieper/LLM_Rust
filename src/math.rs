// math.rs
// Description: Numerical helpers for softmax, cross entropy, gradient computation, and clipping.
// History:
// - 2026-02-01: Consolidate numeric helpers into math.rs.
// Author: Marcus Schlieper

use ndarray::{Array2, Axis};

pub fn softmax_rows(a_logits: &Array2<f32>) -> Array2<f32> {
    let mut a_result = a_logits.clone();

    for mut a_row in a_result.rows_mut() {
        let d_max = a_row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let v_exp: Vec<f32> = a_row.iter().map(|&x| (x - d_max).exp()).collect();
        let d_sum: f32 = v_exp.iter().sum();

        if !d_sum.is_finite() || d_sum <= 0.0 {
            let d_uniform = 1.0 / (a_row.len() as f32).max(1.0);
            for j in 0..a_row.len() {
                a_row[j] = d_uniform;
            }
            continue;
        }

        for (j, &d_e) in v_exp.iter().enumerate() {
            a_row[j] = d_e / d_sum;
        }
    }

    a_result
}

pub fn cross_entropy_loss_step(a_probs: &Array2<f32>, v_target: &[usize]) -> f32 {
    if a_probs.nrows() == 0 || a_probs.ncols() == 0 || v_target.is_empty() {
        return 0.0;
    }

    let i_rows = a_probs.nrows().min(v_target.len());
    let i_vocab = a_probs.ncols();

    let mut d_loss: f32 = 0.0;
    for i in 0..i_rows {
        let i_tgt = v_target[i];
        if i_tgt >= i_vocab {
            continue;
        }
        let d_p = a_probs[[i, i_tgt]];
        d_loss -= d_p.max(1e-15).ln();
    }

    d_loss / (i_rows as f32).max(1.0)
}

pub fn compute_gradients_step(a_probs: &Array2<f32>, v_target: &[usize]) -> Array2<f32> {
    let mut a_grads = a_probs.clone();

    let i_rows = a_probs.nrows();
    let i_cols = a_probs.ncols();

    if i_rows == 0 || i_cols == 0 || v_target.is_empty() {
        return a_grads;
    }

    let i_eff = i_rows.min(v_target.len());
    let d_batch = (i_eff as f32).max(1.0);

    for i in 0..i_eff {
        let i_tgt = v_target[i];
        if i_tgt < i_cols {
            a_grads[[i, i_tgt]] -= 1.0;
        }
    }

    a_grads.mapv_inplace(|x| x / d_batch);
    a_grads
}

pub fn clip_gradients(a_grads: &mut Array2<f32>, d_max_norm: f32) {
    if d_max_norm <= 0.0 || !d_max_norm.is_finite() {
        return;
    }

    let d_norm_sq: f32 = a_grads.iter().map(|&x| x * x).sum::<f32>();
    let d_norm = d_norm_sq.sqrt();

    if d_norm.is_finite() && d_norm > d_max_norm {
        let d_scale = d_max_norm / d_norm;
        a_grads.mapv_inplace(|x| x * d_scale);
    }
}
