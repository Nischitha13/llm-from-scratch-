# MiniGPT — Building and Training an LLM from Scratch

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![JAX](https://img.shields.io/badge/JAX-0.6.2-orange?logo=google&logoColor=white)
![Flax](https://img.shields.io/badge/Flax_NNX-0.10.7-blueviolet)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)

A personal learning project to understand the internal architecture of large language models by building one from scratch. This is not intended for production use — the goal was to deeply understand every component: tokenization, embeddings, multi-head attention, causal masking, the training loop, and autoregressive inference, by implementing each piece manually using JAX and Flax.

> **Sample output:**
> Prompt: `"Once upon a time a big bear"`
> Output: `"Once upon a time a big bear. All a little boy was three years old. He was very happy and wanted to play with his friends. He was very happy and he saw a big..."`

---

## Model Architecture

```
Input text  →  Tokenizer (GPT-2 BPE, vocab=50,257)
            →  Token Embedding  (50257 × 192)
             + Position Embedding (128 × 192)
            →  Transformer Block × 6
                  └─ Multi-Head Attention (6 heads, d_k=32)
                       └─ Q, K, V projections (192 → 32 per head)
                       └─ Scaled dot-product attention
                       └─ Causal mask (no peeking at future tokens)
                       └─ Weighted value aggregation
                  └─ Residual connection
            →  Output Linear Layer (192 → 50257)
            →  Next token prediction
```

| Hyperparameter | Value |
|---|---|
| Parameters | ~20 million |
| Vocabulary size | 50,257 (GPT-2 BPE) |
| Max sequence length | 128 tokens |
| Embedding dimension | 192 |
| Attention heads | 6 |
| Transformer blocks | 6 |
| Training data | TinyStories dataset |
| Framework | JAX + Flax NNX |

---

## Project Structure

```
.
├── Dockerfile                          # Docker image definition
├── docker-compose.yml                  # Run Jupyter with one command
├── requirements.txt                    # All Python dependencies
├── helper.py                           # Model, data loading, and generation code
├── TinyStories-1000.txt                # Sample dataset (1000 stories)
├── training_loss.png                   # Loss curve from full training run
│
├── data_loading.ipynb                  # Stage 1 — Load and explore the data
├── Building_and_training_LLM.ipynb     # Stage 2 — Build the model architecture
├── train.ipynb                         # Stage 3 — Train the model
└── final.ipynb                         # Stage 4 — Load checkpoint, run Gradio demo
```

**Run the notebooks in order:** `data_loading` → `Building_and_training_LLM` → `train` → `final`

---

## Run Locally with Docker

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/building-and-training-llm.git
cd building-and-training-llm

# 2. Start Jupyter (builds the Docker image on first run)
docker-compose up

# 3. Open your browser
# Go to: http://localhost:8888
```

Open the notebooks in order and run all cells.

> **Note:** The trained model checkpoint is not included in this repo (143 MB — too large for GitHub).
> You need to train your own model. See the [Training on Google Colab](#training-on-google-colab-recommended) section below.

---

## Training on Google Colab (Recommended)

Training on CPU is extremely slow (~7 hours for 1000 stories). Use Google Colab's free T4 GPU instead (~5 minutes).

### Steps

**1. Open Google Colab:** [colab.research.google.com](https://colab.research.google.com)

**2. Set runtime to GPU:**
`Runtime → Change runtime type → T4 GPU → Save`

**3. Upload these two files** using the folder icon in the left sidebar:
- `helper.py`
- `TinyStories-1000.txt`

**4. Install dependencies:**
```python
!pip install flax optax grain tiktoken orbax-checkpoint -q
```

**5. Run training:**
```python
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax

from helper import MiniGPT, load_and_preprocess_data

num_epochs = 20

text_dl, batches_per_epoch = load_and_preprocess_data(
    file_path='TinyStories-1000.txt',
    batch_size=32,
    maxlen=128,
    max_stories=1000,
    num_epochs=num_epochs,
    shuffle=True,
    seed=42
)

model = MiniGPT()

def loss_fn(model, batch):
    inputs, targets = batch
    logits = model(inputs)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
    return loss, logits

total_steps = batches_per_epoch * num_epochs
warmup_steps = max(1, total_steps // 10)

lr_schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=3e-4,
    warmup_steps=warmup_steps, decay_steps=total_steps, end_value=1e-5
)

optimizer = nnx.Optimizer(model, optax.adamw(learning_rate=lr_schedule, weight_decay=0.01), wrt=nnx.Param)
metrics = nnx.MultiMetric(loss=nnx.metrics.Average('loss'))

@nnx.jit
def train_step(model, optimizer, metrics, batch):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch[1])
    optimizer.update(model, grads)

prep_target_batch = jax.vmap(
    lambda tokens: jnp.concatenate((tokens[1:], jnp.array([0]))))

metrics_history = {'train_loss': []}

for epoch in range(num_epochs):
    step = 0
    for batch in text_dl:
        input_batch = jnp.array(jnp.array(batch).T).astype(jnp.int32)
        target_batch = prep_target_batch(jnp.array(jnp.array(batch).T)).astype(jnp.int32)
        train_step(model, optimizer, metrics, (input_batch, target_batch))
        if (step + 1) % 10 == 0:
            for metric, value in metrics.compute().items():
                metrics_history[f'train_{metric}'].append(value)
            metrics.reset()
            print(f"Epoch {epoch+1}, Step {step+1}, Loss: {metrics_history['train_loss'][-1]:.4f}")
        step += 1
```

**6. Save and download the checkpoint:**
```python
from pathlib import Path
import orbax.checkpoint

checkpoint_path = Path("/content/small_checkpoint.orbax").absolute()
checkpointer = orbax.checkpoint.PyTreeCheckpointer()
checkpointer.save(str(checkpoint_path), nnx.state(model), force=True)

from google.colab import files
import shutil
shutil.make_archive("/content/checkpoint", "zip", "/content/small_checkpoint.orbax")
files.download("/content/checkpoint.zip")
```

**7. Copy the checkpoint into Docker:**
```bash
# On your local Mac terminal:
unzip ~/Downloads/checkpoint.zip -d ~/Downloads/small_checkpoint.orbax
docker ps   # get container ID
docker cp ~/Downloads/small_checkpoint.orbax <container_id>:/app/small_checkpoint.orbax
```

**8. Run `final.ipynb`** — the Gradio interface will launch at `http://localhost:7860`

---

## Training Loss

The image below shows loss from a full training run on 2,000,000 stories:

![Training Loss](training_loss.png)

---

## Results

| Metric | Value |
|---|---|
| Initial training loss | ~10.92 |
| Random baseline loss | 10.82 (= log(50,257)) |
| Training data | 1,000 stories × 20 epochs |
| Training time | ~5 min on Google Colab T4 GPU |

A randomly initialized model on a 50,257-token vocabulary produces loss ≈ log(50,257) = **10.82** — equivalent to guessing uniformly at random. The model quickly learns to beat random and generates coherent story continuations.

**Generated sample:**
```
Prompt : "Once upon a time a big bear"
Output : "Once upon a time a big bear. All a little boy was three years
          old. He was very happy and wanted to play with his friends..."
```

![Training Loss](training_loss.png)

---

## Key Engineering Decisions

**1. Causal (not bidirectional) attention mask**
Each token is only allowed to attend to itself and tokens that came before it — never future tokens. This is enforced by a lower-triangular mask applied before softmax. Without it, the model could "cheat" during training by reading the answer it's supposed to predict, and would fail completely at inference time when future tokens don't exist.

**2. Right-padding to match training and inference**
Sequences shorter than `maxlen=128` are padded with zeros on the right (not the left). Padding on the left would shift all positional embeddings, making position 0 mean something different at training vs inference time. Right-padding keeps the real tokens at their correct positions in both cases.

**3. JAX + Flax NNX over PyTorch**
JAX compiles the entire training step (`@nnx.jit`) to XLA, which fuses operations and eliminates Python overhead — critical for GPU utilization. The functional, stateless design also makes it straightforward to reason about what the model is doing at each step, which was the goal of building from scratch.

---

## Tech Stack

| Library | Purpose |
|---|---|
| [JAX](https://github.com/google/jax) | Accelerated numerical computing |
| [Flax NNX](https://flax.readthedocs.io/) | Neural network layers |
| [Optax](https://optax.readthedocs.io/) | Optimizers (AdamW, cosine schedule) |
| [Grain](https://github.com/google/grain) | Efficient data loading |
| [tiktoken](https://github.com/openai/tiktoken) | GPT-2 BPE tokenizer |
| [Orbax](https://orbax.readthedocs.io/) | Model checkpointing |
| [Gradio](https://www.gradio.app/) | Interactive web demo |
| [Docker](https://www.docker.com/) | Reproducible environment |
