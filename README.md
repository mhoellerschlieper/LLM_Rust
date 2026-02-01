# LLM (Rust) &mdash; Minimal Transformer with BPE Tokenizer, Training, and Checkpoints

This repository describes and implements a self-contained Large Language Model (LLM) in Rust, consisting of a compact Transformer architecture, a Byte Pair Encoding (BPE) tokenizer, a simple training loop for pretraining and instruction tuning, and a robust checkpoint format that consistently saves and loads both the tokenizer and model parameters.

The project addresses the practical necessity of developing a complete LLM as a closed system in which data pipeline, tokenization, model, optimization, inference, and persistence are integrated into a reproducible workflow. It deliberately prioritizes a transparent implementation suitable for expert analysis, debugging, and incremental extension.

Implemented features:
- Temperature, top-k, top-p
- Multi-head attention
  
## Contents

- Project overview
- Architecture and components
- Tokenizer (BPE) and determinism
- Training (pretraining and instruction tuning)
- Inference (greedy decoding)
- Checkpoints (save and load with rebuild)
- Data formats
- Build and run
- Security and robustness
- Roadmap toward a complete LLM
- License and contact

## Project Overview

The repository implements a compact Transformer pipeline comprising embeddings, multiple Transformer blocks (self-attention, feed-forward, layer normalization), and an output projection onto the token vocabulary. Tokenization is performed via a BPE tokenizer that is persisted in the checkpoint to ensure consistent vocabulary sizes and, consequently, consistent parameter shapes for the output projection.

A central feature is checkpoint loading with rebuild: when loading, the model is rebuilt based on the tokenizer vocabulary stored in the checkpoint in order to avoid shape mismatches that would otherwise arise as soon as the vocabulary size differs between training and inference.

## Architecture and Components

The implementation is consolidated into a small number of modules and emphasizes self-contained executability.

- `main.rs`
  - CLI with menu loop
  - Train, Save, Load, Ask
  - Initial tokenizer training for immediate usability
- `layer.rs`
  - Core of the model (Layer trait, Embeddings, Self Attention, Feed Forward, LayerNorm, TransformerBlock, OutputProjection)
  - Optimizer (Adam)
  - Llm with Train, Predict, Save, and Load
  - Checkpoint structure `LlmCheckpoint`
- `tokenizer.rs`
  - BPE training, encoding, decoding
  - Deterministic training logic and reproducible configuration
  - Tokenizer checkpoint structure `BpeTokenizerCheckpoint`
- `train.rs`
  - Dataset loader for JSON and CSV
- `utils.rs`
  - ASCII normalization and utility functions
  - JSON serialization and atomic file writing
- `math.rs`
  - Softmax, cross-entropy, gradient computation, gradient clipping

The default configuration (as of the current state) uses the following hyperparameters:

- `MAX_SEQ_LEN = 80`
- `EMBEDDING_DIM = 128`
- `HIDDEN_DIM = 256`
- 3 Transformer blocks
- Output projection dimension: `vocab_size` from the tokenizer vocabulary

## Tokenizer (BPE) and Determinism

The BPE tokenizer is trained from the corpus and uses an ASCII-focused preprocessing pipeline that segments whitespace and conservatively separates punctuation to obtain a robust token structure for simple training data.

Reproducibility is supported by a configuration object `BpeTokenizerConfig`, which is persisted in the checkpoint. This enables subsequent analysis of the tokenizer state and its training parameters&mdash;an important requirement for evaluation, debugging, and A/B comparisons (cf. Goodfellow, Bengio, &amp; Courville, 2016).

## Training (Pretraining and Instruction Tuning)

The project distinguishes two training phases:

- Pretraining on generic texts
- Instruction tuning on chat or dialogue data

Both phases use the same training loop: the model is trained autoregressively for next-token prediction by deriving input tokens and target tokens from a token sequence shifted by one position. The loss is computed via cross-entropy, and gradients propagate backward through the layers via backpropagation.

Gradient clipping is applied to reduce numerical instability, particularly for small models and non-optimized initializations; in practice, this measure often contributes to stabilization (Pascanu, Mikolov, &amp; Bengio, 2013).

## Inference (Greedy Decoding)

Inference uses greedy decoding by computing softmax probabilities for the last token in the sequence and selecting the maximum until an EOS token is produced or `MAX_SEQ_LEN` is reached.

This strategy is intentionally simple to ensure system completeness and traceability, although in production scenarios sampling methods such as top-k or nucleus sampling and temperature scaling typically yield better text quality (Holtzman et al., 2020).

## Checkpoints (Save and Load with Rebuild)

### Why Rebuild Is Required When Loading

Because the output projection matrix has shape `[embedding_dim, vocab_size]`, `vocab_size` directly depends on the tokenizer vocabulary size. Consequently, a different tokenizer at load time inevitably leads to parameter mismatches.

Therefore, the following procedure is used during loading via `Llm::load_checkpoint_rebuild`:

1. Load and validate the checkpoint JSON
2. Reconstruct the tokenizer from the checkpoint
3. Rebuild the model (embeddings and output projection with `vocab_size` from the checkpoint)
4. Apply the parameter vector to the newly created layers

### Atomic Writes

When saving, an atomic write strategy is used (temporary file, then rename) to prevent inconsistent checkpoints in the event of process termination or system issues&mdash;particularly relevant for long training runs and repeated saves.

## Data Formats

### JSON

The current dataset logic expects, for JSON, a list of strings.

Example `pretraining_data.json`:

json
[
  &quot;Some training text.&quot;,
  &quot;Another example sentence.&quot;
]


Example `chat_training_data.json`:

json
[
  &quot;User: Hello Assistant: Hello, how can I help?&quot;,
  &quot;User: Explain transformers. Assistant: ...&quot;
]


### CSV

CSV is read without a header, and each line is concatenated into a string. This format should be understood more as a simple adapter than as a semantically structured chat representation.

## Build and Run

### Requirements

- Rust stable toolchain
- Cargo

### Build

bash
cargo build --release


### Run

bash
cargo run --release


The menu currently provides the following commands:

- `t` Train (pretraining and instruction tuning)
- `s` Save checkpoint
- `l` Load checkpoint (rebuild)
- `a` Ask (enter a prompt, generate an answer)
- `e` Exit

## Security and Robustness

The project implements validations and defensive defaults in several places, including:

- Checkpoint validation via magic value and version
- Parameter length checks when loading
- Learning rate validation
- Sequence length limiting
- Atomic checkpoint saving
- ASCII normalization to reduce Unicode edge cases

At the same time, it should be noted that the system is intended as a research and development foundation rather than a production-ready LLM runtime, especially since sampling, efficient batching logic, mixed precision, KV cache, and formal test suites are not yet fully implemented.

## Roadmap Toward a Complete LLM

A complete LLM in terms of scalability, robust evaluation paths, and production-grade inference typically requires several technical extensions, which may be considered next steps for this repository:

1. Tokenizer extensions
   - Support for Unicode normalization and controlled byte fallbacks
   - Consistent special tokens and clear prompt templates for chat
2. Training infrastructure
   - Mini-batches, shuffling, gradient accumulation
   - Validation split, early stopping, and metrics
   - Mixed precision and more stable initializations
3. Inference quality
   
   - Repetition penalty and stopping criteria
   - KV cache for efficient autoregression
4. Model architecture
   
   - Positional embeddings or RoPE
   - Attention mask handling for padding and batches
5. Persistence and compatibility
   - Explicit compatibility rules between checkpoint versions
   - Optional sharding of large parameter vectors
6. Testability and verification
   - Unit tests for tokenizer, softmax stability, checkpoint roundtrip
   - Golden tests for deterministic runs




