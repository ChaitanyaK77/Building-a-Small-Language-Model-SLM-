[![Notebook](https://img.shields.io/badge/jupyter-notebook-orange.svg)](Small_Language_Model_Building.ipynb)
[![PyPI](https://img.shields.io/pypi/v/torch.svg)](https://pypi.org/project/torch/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](#license)

# building a Small Language Model(SLM) from Scratch 

A step-by-step Jupyter Notebook demonstrating how to build and train a compact small language model (“SLM”) from scratch using the **TinyStories** dataset. Covers data preparation, BPE tokenization, efficient binary storage, GPU memory locking, Transformer architecture, training configuration, and sample text generation.

---

## 🚀 Highlights

- **End-to-end pipeline**  
  From raw text to a fully trained model—all within one notebook.
- **Efficient Tokenization**  
  Uses Hugging Face’s `tiktoken` (BPE) for subword encodings.
- **Disk-backed Dataset**  
  Saves token IDs in `.bin` files for fast reloads.
- **Memory Locking**  
  Demonstrates reserving GPU memory to avoid fragmentation.
- **Custom Transformer**  
  Minimal PyTorch model with multi-head attention and feed-forward blocks.
- **Training Loop**  
  Configurable optimizer, learning-rate schedules, gradient clipping, and logging.
- **Sample Outputs**  
  Generates TinyStories-style text to verify model behavior.

---

## 📖 Table of Contents

1. [Introduction](#introduction)  
2. [Dataset](#dataset)  
3. [Prerequisites](#prerequisites)  
4. [Setup & Installation](#setup--installation)  
5. [Notebook Walk-through](#notebook-walk-through)  
6. [Training Configuration](#training-configuration)  
7. [Sample Generation](#sample-generation)  
8. [Results & Next Steps](#results--next-steps)  
9. [Contributing](#contributing)  
10. [License](#license)

---

## Introduction

Building Large Language Models (LLMs) from scratch can be resource-intensive. This notebook shows how to create a **Small Language Model (SLM)** using a lightweight dataset, minimalist code, and standard hardware (e.g., a single GPU).

---

## Dataset

- **TinyStories** (~2 million stories) for training and ~20,000 for validation  
- Hosted on Hugging Face Datasets  
- Each “story” is a short, self-contained text ideal for low-compute experimentation

---

## Prerequisites

- Python 3.8+  
- GPU with CUDA (optional, but highly recommended)  
- [PyTorch](https://pytorch.org/)  
- [Hugging Face `datasets`](https://github.com/huggingface/datasets)  
- [`tiktoken`](https://github.com/openai/tiktoken)  
- `numpy`, `matplotlib`, `tqdm`

---

## Setup & Installation

```bash
# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install core dependencies
pip install torch torchvision \
            datasets \
            tiktoken \
            numpy \
            matplotlib \
            tqdm
```

---

## Notebook Walk-through

### Data Loading

```python
from datasets import load_dataset
ds = load_dataset("roneneldan/TinyStories")
```

### Tokenization

Implements Byte Pair Encoding via `tiktoken`.  
Converts text → token IDs → binary `.bin` files.

### Dataset Storage

Saves training/validation tokens on disk for fast reload.

### Memory Locking

Reserves GPU memory (`torch.cuda.set_per_process_memory_fraction` or similar) to prevent fragmentation.

### Model Definition

Lightweight Transformer with configurable layers, heads, and embedding size.

### Training Loop

- **Optimizer**: AdamW  
- **LR Schedulers**: LinearLR, CosineAnnealingLR, SequentialLR  
- Gradient clipping, periodic evaluation, and loss logging.

### Evaluation & Generation

- Samples new stories to verify qualitative performance.  
- Plots loss curves with Matplotlib.

---

## Training Configuration

```python
@dataclass
class GPTConfig:
    block_size: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float = 0.0
    bias: bool = True
```

**Example hyperparameters:**

| Parameter  | Value |
|------------|-------|
| block_size | 128   |
| vocab_size | 50,000|
| n_layer    | 4     |
| n_head     | 8     |
| n_embd     | 256   |
| dropout    | 0.1   |

---

## Sample Generation

After training, run:

```python
prompt = "Once upon a time"
generated = model.generate(prompt, max_new_tokens=100)
print(generated)
```

Expect TinyStories-style outputs (short, coherent sentences).

---

## Results & Next Steps

### Results

- Validation loss curve (see notebook plot).  
- Qualitative samples demonstrating grammar and coherence.

### Next Steps

- Scale up dataset (longer stories).  
- Experiment with deeper/wider architectures.  
- Integrate more sophisticated tokenizers (e.g., SentencePiece).

---

## Contributing

1. Fork this repository  
2. Create a new branch (`git checkout -b feature/xyz`)  
3. Commit your changes (`git commit -m 'Add xyz feature'`)  
4. Push to your branch (`git push origin feature/xyz`)  
5. Open a Pull Request  

All contributions—bug reports, documentation fixes, new features—are welcome!

---
## References
I have taken inspiration from the following resources 
1. Karpathy, A. (2023). [nanoGPT](https://github.com/karpathy/nanoGPT) [GitHub repository].  
2. Eldan, R., & Li, Y. (2023). [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759) arXiv preprint arXiv:2305.07759.

---
## License

Released under the MIT License.
