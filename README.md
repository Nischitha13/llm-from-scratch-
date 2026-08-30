# MiniGPT — Building and Training an LLM from Scratch

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![JAX](https://img.shields.io/badge/JAX-0.6.2-orange?logo=google&logoColor=white)
![Flax](https://img.shields.io/badge/Flax_NNX-0.10.7-blueviolet)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)

## What this project is

This is a personal learning project built to understand how large language
models actually work under the hood, by building a small one from the
ground up instead of just using an existing library. It is **not** meant for
production use — the point was to hand-implement every core piece of a
GPT-style model (tokenization, embeddings, multi-head attention, causal
masking, the training loop, and text generation) so that each step is fully
understood rather than hidden behind a framework.

The result, "MiniGPT," is a ~20-million-parameter transformer trained on the
[TinyStories](https://arxiv.org/abs/2305.07759) dataset — a collection of
very simple, short children's stories designed specifically for training
tiny language models that can still produce coherent text.

> **Sample output:**
> Prompt: `"Once upon a time a big bear"`
> Output: `"Once upon a time a big bear. All a little boy was three years old. He was very happy and wanted to play with his friends. He was very happy and he saw a big..."`

---

## How it works, in plain English

1. **Tokenizer** — Raw text is split into sub-word tokens using GPT-2's BPE
   tokenizer (the same one OpenAI uses), giving a vocabulary of 50,257
   possible tokens.
2. **Embeddings** — Each token is converted into a 192-dimensional vector,
   and a second embedding encodes *where* in the sequence that token sits
   (its position), since the model has no built-in sense of order.
3. **Transformer blocks** — The embedded sequence passes through 6 stacked
   blocks. Each block uses multi-head self-attention so every token can
   "look at" other tokens and decide how relevant they are, then adds the
   result back to its input via a residual connection (which helps
   gradients flow during training).
4. **Causal mask** — Attention is restricted so a token can only look at
   itself and earlier tokens, never ones that come after it. This is what
   makes the model capable of *predicting the next word* instead of
   cheating by peeking at the answer.
5. **Output layer** — The final representation is projected back to a
   50,257-way probability distribution over the vocabulary, and the model
   picks (or samples) the next token from that distribution — repeating
   this one token at a time produces the generated text.

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

Each notebook builds on the previous one, so they are meant to be run in
order: `data_loading` → `Building_and_training_LLM` → `train` → `final`.

---

## Running it yourself

You have two options: run everything locally in Docker (easiest way to get
a working environment), or train on Google Colab's free GPU (much faster
than training on a CPU). Both are covered below.

### Option 1 — Run Locally with Docker

**Prerequisites:** [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/building-and-training-llm.git
cd building-and-training-llm

# 2. Start Jupyter (builds the Docker image on first run)
docker-compose up

# 3. Open your browser at:
# http://localhost:8888
```

Then open the notebooks in order and run all cells.

> **Note:** The trained model checkpoint isn't included in this repo (it's
> 143 MB, too large for GitHub). You'll need to train your own — see the
> Colab section below — before `final.ipynb` will have a model to load.

### Option 2 — Training on Google Colab (Recommended)

Training on a CPU is very slow (roughly 7 hours for just 1,000 stories).
Colab's free T4 GPU does the same job in about 5 minutes, so it's the
recommended path for actually training a model.

**1. Open Colab:** [colab.research.google.com](https://colab.research.google.com)

**2. Switch to a GPU runtime:**
`Runtime → Change runtime type → T4 GPU → Save`

**3. Upload two files** using the folder icon in the left sidebar:
- `helper.py`
- `TinyStories-1000.txt`

**4. Install the dependencies used for training:**
```python
!pip install flax optax grain tiktoken orbax-checkpoint -q
```

**5. Run the training loop:**
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

**7. Move the checkpoint into your Docker container:**
```bash
# On your local machine terminal:
unzip ~/Downloads/checkpoint.zip -d ~/Downloads/small_checkpoint.orbax
docker ps   # get container ID
docker cp ~/Downloads/small_checkpoint.orbax <container_id>:/app/small_checkpoint.orbax
```

**8. Run `final.ipynb`** — this loads the checkpoint and launches a Gradio
demo at `http://localhost:7860` where you can type a prompt and see the
model generate a story.

---

## Training Loss

The chart below shows the loss curve from a full training run on 2,000,000
stories — a much larger run than the 1,000-story quick example above.

![Training Loss](training_loss.png)

---

## Results

| Metric | Value |
|---|---|
| Initial training loss | ~10.92 |
| Random baseline loss | 10.82 (= log(50,257)) |
| Training data | 1,000 stories × 20 epochs |
| Training time | ~5 min on Google Colab T4 GPU |

Why compare against 10.82? A freshly initialized model that hasn't learned
anything yet should be no better than guessing uniformly at random across
all 50,257 possible tokens, and the loss for pure random guessing works out
to `log(50,257) ≈ 10.82`. The fact that the model starts right around that
number and then drops as training proceeds confirms it's actually learning
to predict text rather than something being broken. After training, it
reliably produces coherent story continuations.

**Generated sample:**
```
Prompt : "Once upon a time a big bear"
Output : "Once upon a time a big bear. All a little boy was three years
          old. He was very happy and wanted to play with his friends..."
```

---

## Key Engineering Decisions

**1. Causal (not bidirectional) attention mask**
Each token is only allowed to attend to itself and the tokens that came
before it — never future tokens. This is enforced with a lower-triangular
mask applied before the softmax. Without it, the model could "cheat" during
training by looking directly at the token it's supposed to predict, and it
would then fail completely at inference time, when future tokens don't
exist yet to peek at.

**2. Right-padding to keep training and inference consistent**
Sequences shorter than `maxlen=128` are padded with zeros on the right, not
the left. Padding on the left would shift every real token into different
position indices depending on how much padding was added, so position 0
would mean something different from one example to the next. Padding on
the right keeps every real token pinned to its true position, so the
positional embeddings mean the same thing during training and at
inference.

**3. JAX + Flax NNX instead of PyTorch**
JAX compiles the entire training step (`@nnx.jit`) down to XLA, which fuses
operations together and removes Python's per-step overhead — this matters
a lot for keeping the GPU busy. JAX's functional, stateless style also
makes it easier to trace exactly what happens to the data at each step,
which lined up well with the goal of understanding the model from first
principles rather than treating it as a black box.

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
