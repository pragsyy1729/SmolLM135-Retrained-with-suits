# SmolLM2-135M Training from Scratch

This repository contains a from-scratch implementation of the SmolLM2-135M model architecture, trained on custom text data.

## Table of Contents
- [Model Architecture](#model-architecture)
- [Parameter Calculation](#parameter-calculation)
- [Training Data](#training-data)
- [Training Details](#training-details)
- [Speedups Used](#speedups-used)
- [Results](#results)
- [Usage](#usage)

---

## Model Architecture

SmolLM2-135M is a **Llama-based decoder-only transformer** model. Unlike GPT-2, it incorporates modern architectural improvements that have become standard in recent language models.

### Architecture Overview

```
Input Tokens
     │
     ▼
┌─────────────────┐
│ Token Embedding │  (No position embeddings - RoPE applied in attention)
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│         Transformer Block x30       │
│  ┌─────────────────────────────┐    │
│  │ RMSNorm                     │    │
│  │      ↓                      │    │
│  │ Grouped Query Attention     │    │
│  │ (9 query heads, 3 KV heads) │    │
│  │ + RoPE                      │    │
│  │      ↓                      │    │
│  │ Residual Connection         │    │
│  │      ↓                      │    │
│  │ RMSNorm                     │    │
│  │      ↓                      │    │
│  │ SwiGLU MLP                  │    │
│  │      ↓                      │    │
│  │ Residual Connection         │    │
│  └─────────────────────────────┘    │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────┐
│ Final RMSNorm   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ LM Head         │  (Tied with token embeddings)
└────────┬────────┘
         │
         ▼
    Output Logits
```

### Key Components

#### 1. RMSNorm (Root Mean Square Normalization)
Unlike LayerNorm, RMSNorm doesn't center activations (no mean subtraction), making it more computationally efficient.

```python
# Formula: x * weight / sqrt(mean(x²) + eps)
def forward(self, x):
    rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    return x * rms * self.weight
```

#### 2. Rotary Position Embedding (RoPE)
RoPE encodes position information by rotating query and key vectors. The dot product of rotated vectors naturally encodes relative positions.

- **Advantage**: No learned position embeddings needed
- **Advantage**: Better extrapolation to longer sequences
- **theta**: 10,000 (base frequency)

#### 3. Grouped Query Attention (GQA)
GQA reduces memory bandwidth by sharing key-value heads across multiple query heads.

| Component | Count |
|-----------|-------|
| Query Heads | 9 |
| Key-Value Heads | 3 |
| KV Groups | 3 (each KV head shared by 3 query heads) |
| Head Dimension | 64 |

```python
# Q: (B, T, 9 heads, 64)
# K, V: (B, T, 3 heads, 64) → repeated to match 9 query heads
```

#### 4. SwiGLU MLP
SwiGLU is a gated linear unit variant using SiLU activation, shown to improve model quality.

```python
# Formula: down_proj(silu(gate_proj(x)) * up_proj(x))
def forward(self, x):
    return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

### Configuration

| Parameter | Value |
|-----------|-------|
| `vocab_size` | 50,304 (GPT-2 compatible) |
| `hidden_size` | 576 |
| `intermediate_size` | 1,536 |
| `num_hidden_layers` | 30 |
| `num_attention_heads` | 9 |
| `num_key_value_heads` | 3 |
| `max_position_embeddings` | 2,048 |
| `rms_norm_eps` | 1e-5 |
| `rope_theta` | 10,000 |
| `hidden_act` | SiLU |
| `tie_word_embeddings` | True |

---

## Parameter Calculation

### Component-by-Component Breakdown

#### 1. Token Embeddings
```
vocab_size × hidden_size = 50,304 × 576 = 28,975,104 parameters
```

#### 2. Attention (per layer)
```
Q projection:  hidden_size × (num_heads × head_dim)     = 576 × 576 = 331,776
K projection:  hidden_size × (num_kv_heads × head_dim)  = 576 × 192 = 110,592
V projection:  hidden_size × (num_kv_heads × head_dim)  = 576 × 192 = 110,592
O projection:  (num_heads × head_dim) × hidden_size     = 576 × 576 = 331,776
─────────────────────────────────────────────────────────────────────────────
Total per layer: 884,736 parameters
Total for 30 layers: 884,736 × 30 = 26,542,080 parameters
```

#### 3. MLP (per layer)
```
gate_proj:  hidden_size × intermediate_size = 576 × 1,536 = 884,736
up_proj:    hidden_size × intermediate_size = 576 × 1,536 = 884,736
down_proj:  intermediate_size × hidden_size = 1,536 × 576 = 884,736
─────────────────────────────────────────────────────────────────────
Total per layer: 2,654,208 parameters
Total for 30 layers: 2,654,208 × 30 = 79,626,240 parameters
```

#### 4. RMSNorm (per layer + final)
```
input_layernorm:           hidden_size = 576
post_attention_layernorm:  hidden_size = 576
─────────────────────────────────────────────
Total per layer: 1,152 parameters
Total for 30 layers: 1,152 × 30 = 34,560 parameters
Final norm: 576 parameters
Total normalization: 35,136 parameters
```

#### 5. LM Head
```
Tied with token embeddings: 0 additional parameters
```

### Total Parameter Count

| Component | Parameters | Percentage |
|-----------|------------|------------|
| Embedding | 28,975,104 | 21.4% |
| Attention | 26,542,080 | 19.6% |
| MLP | 79,626,240 | 58.9% |
| Normalization | 35,136 | 0.03% |
| **Total** | **135,178,560** | **100%** |

> **Note**: Our implementation uses vocab_size=50,304 instead of the original 49,152, adding ~663K parameters to the embedding layer. Original SmolLM2-135M has ~134.5M parameters.

---

## Training Data

### Dataset Description
The model was trained on dialogue scripts from the television series **"Suits"**, a legal drama that follows characters working at a fictional New York law firm.

### Characteristics
- **Content Type**: Television dialogue scripts
- **Genre**: Legal drama
- **Language Style**: Professional legal terminology mixed with casual dialogue
- **Text Format**: Character names followed by dialogue

### Tokenization
- **Tokenizer**: GPT-2 BPE tokenizer (tiktoken)
- **Vocabulary Size**: 50,257 tokens (padded to 50,304 for GPU efficiency)

---

## Training Details

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Total Steps | 5,000 + 50 (resumed) |
| Batch Size | 16 (effective) |
| Micro Batch Size | 4 |
| Gradient Accumulation | 4 steps |
| Sequence Length | 1,024 tokens |
| Learning Rate | 6e-4 (max) |
| LR Schedule | Cosine with warmup |
| Warmup Steps | 500 |
| Weight Decay | 0.1 |
| Gradient Clipping | 1.0 |
| Optimizer | AdamW (fused) |
| Precision | bfloat16 |

### Checkpointing
- Checkpoints saved every **500 steps**
- Text generation with fixed prompts at each checkpoint
- Final checkpoint at step 5,050 (after resume demonstration)

### Fixed Evaluation Prompts
```
1. "Once upon a time"
2. "The meaning of life is"
3. "In a galaxy far away"
```

---

## Speedups Used

| Speedup | Implementation |
|---------|----------------|
| **Flash Attention** | `F.scaled_dot_product_attention(is_causal=True)` |
| **Mixed Precision** | `torch.autocast(dtype=torch.bfloat16)` |
| **torch.compile** | JIT compilation for CUDA |
| **TF32 Precision** | `torch.set_float32_matmul_precision('high')` |
| **Gradient Accumulation** | 4 micro-batches per step |
| **Fused AdamW** | `fused=True` for CUDA |
| **Power-of-2 Vocab** | 50,304 for efficient GPU memory access |

---

## Results

### Training Progress
- **Initial Loss**: ~10.8 (random initialization)
- **Final Loss**: Significantly reduced after 5,050 steps
- **Checkpoint Resume**: Successfully demonstrated loading from step 5,000 and continuing training

### Checkpoints Saved
```
checkpoints/
├── checkpoint_step_500.pt
├── checkpoint_step_1000.pt
├── checkpoint_step_1500.pt
├── checkpoint_step_2000.pt
├── checkpoint_step_2500.pt
├── checkpoint_step_3000.pt
├── checkpoint_step_3500.pt
├── checkpoint_step_4000.pt
├── checkpoint_step_4500.pt
├── checkpoint_step_5000.pt
└── checkpoint_step_5050.pt
```

---

## Usage

### Requirements
```bash
pip install torch tiktoken matplotlib
```

### Training
1. Upload `input.txt` (training data) to the working directory
2. Run the notebook cells sequentially
3. Checkpoints will be saved to `checkpoints/` directory

### Loading a Checkpoint
```python
checkpoint = torch.load('checkpoints/checkpoint_step_5000.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

### Text Generation
```python
generated = generate_text(
    model,
    prompt="Once upon a time",
    max_new_tokens=50,
    temperature=0.8,
    top_k=50
)
print(generated)
```

---

## References

- [SmolLM2 - HuggingFace](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)
- [RoPE: Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [GQA: Grouped Query Attention](https://arxiv.org/abs/2305.13245)
- [SwiGLU Activation](https://arxiv.org/abs/2002.05202)
- [RMSNorm](https://arxiv.org/abs/1910.07467)

---

## License

This project is for educational purposes.

---

