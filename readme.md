# Mental Health Chatbot - Mobile AI Assistant

A mobile-optimized AI chatbot fine-tuned on mental health topics (Bipolar, OCD, Anxiety) that runs entirely on-device using TensorFlow Lite. This project generates Q&A pairs from PDF books, fine-tunes Gemma 3-1B models, and converts them for mobile deployment.

##  Project Overview

This project creates specialized mental health chatbots that can run on mobile devices without requiring internet connectivity. It processes mental health literature, generates conversational Q&A pairs, and fine-tunes lightweight language models for on-device inference.

### Key Features
-  **Automated Q&A Generation**: Extracts knowledge from PDF books and generates conversational question-answer pairs
-  **Fine-tuned Models**: Custom models trained on mental health topics (Bipolar, OCD, Anxiety)
-  **Mobile-Ready**: Converted to TensorFlow Lite format for on-device inference
-  **Privacy-First**: Runs completely offline on mobile devices
-  **Optimized**: Dynamic int4 quantization for efficient mobile performance

##  Project Structure

```
├── Data/                      # Raw PDF books and datasets
├── qa_pair/                   # Generated Q&A pairs (JSON format)
├── results/                   # Fine-tuned models and converted outputs
├── task/                      # Final .task bundles for mobile deployment
├── .gitignore                 # Git ignore rules
├── book-to-qa.ipynb          # PDF → Q&A pair generation
├── convert_to_tflite.ipynb   # Model → TFLite conversion
├── llm_bundling.ipynb        # TFLite → .task bundling with tokenizer
├── main.ipynb                # Main execution notebook
├── MOE.ipynb                 # Mixture of Experts experimentation
├── run_chat.ipynb            # Test chat interface
└── train_and_merge.ipynb     # Model fine-tuning and LoRA merging
```

##  Getting Started

### Prerequisites

```bash
# Python 3.10+
pip install torch transformers
pip install peft datasets
pip install PyPDF2 tqdm
pip install ai-edge-torch
pip install mediapipe
```

### Required Resources
- **GPU**: Recommended for training (Kaggle/Colab GPU works)
- **RAM**: 16GB+ recommended
- **Storage**: ~5GB per fine-tuned model
- **HuggingFace Token**: Required for Gemma model access

##  Workflow

### Step 1: Generate Q&A Pairs from PDFs

**Notebook**: `book-to-qa.ipynb`

Extracts text from mental health PDFs and generates conversational Q&A pairs using Llama-3.2-1B-Instruct.

```python
# Configuration in book-to-qa.ipynb
books = ['bipolar', 'ocd', 'anxiety']
pairs_per_chunk = 20     # Q&A pairs per text chunk
max_chunks = 100         # Number of chunks to process
chunk_size = 400         # Words per chunk
```

**Output**: `{book}_pairs.json` in `/qa_pair/` folder

**Example Output**:
```json
[
  {
    "question": "What are the early warning signs of a bipolar episode?",
    "answer": "Early warning signs include changes in sleep patterns, increased energy levels..."
  }
]
```

### Step 2: Fine-tune Gemma Model

**Notebook**: `train_and_merge.ipynb`

Fine-tunes Google's Gemma-3-1B-Instruct model on the generated Q&A pairs using LoRA (Low-Rank Adaptation).

**Key Features**:
- 4-bit quantization for memory efficiency
- LoRA adapters (r=16, alpha=32)
- BFloat16 training optimized for Gemma
- Automatic merging of LoRA weights into base model

**Training Configuration**:
```python
book = "anxiety"  # or "bipolar", "ocd"
num_train_epochs = 3
batch_size = 4
learning_rate = 1e-4
```

**Outputs**:
1. `gemma-lora-{book}/` - LoRA adapter weights (~50-100MB)
2. `gemma-{book}-merged/` - Merged full model (~500MB)

### Step 3: Convert to TensorFlow Lite

**Notebook**: `convert_to_tflite.ipynb`

Converts the PyTorch model to TensorFlow Lite format using AI Edge Torch.

**Conversion Script**:
```bash
python3 convert_gemma3_to_tflite.py \
  --quantize="dynamic_int4_block128" \
  --checkpoint_path="/path/to/merged/model" \
  --output_path="/path/to/output/gemma3.tflite" \
  --prefill_seq_lens=1024 \
  --kv_cache_max_len=2048 \
  --mask_as_input=True
```

**Quantization**: Dynamic int4 block128 for optimal mobile performance

### Step 4: Bundle for Mobile

**Notebook**: `llm_bundling.ipynb`

Creates a `.task` bundle that includes the TFLite model and tokenizer for MediaPipe deployment.

```python
config = bundler.BundleConfig(
    tflite_model="gemma.tflite",
    tokenizer_model="tokenizer.model",
    start_token="<bos>",
    stop_tokens=["<eos>"],
    output_filename="gemma.task"
)
bundler.create_bundle(config)
```

**Output**: `{book}.task` - Ready for mobile deployment

##  Mobile Deployment

### Using Google AI Edge Gallery

1. **Transfer Files**: Copy `.task` files to your Android device
2. **Import Model**: Open AI Edge Gallery app → '+' button → Select `.task` file
3. **Test Chat**: Start conversing with your mental health assistant

### System Requirements
- Android 8.0+ (API level 26+)
- 2GB+ RAM
- 500MB+ storage per model

##  Testing

**Notebook**: `run_chat.ipynb`

Test the fine-tuned model before mobile conversion:

```python
# Load model
model = AutoModelForCausalLM.from_pretrained("./results/gemma-anxiety-merged")
tokenizer = AutoTokenizer.from_pretrained("./results/gemma-anxiety-merged")

# Chat
prompt = "What helps manage anxiety?"
response = generate_response(model, tokenizer, prompt)
```

##  Experiments

**Notebook**: `MOE.ipynb`

Experimental Mixture of Experts (MoE) architecture for combining multiple specialized models.

##  Model Specifications

| Model | Base | Parameters | Training | Output Size | Mobile Size |
|-------|------|------------|----------|-------------|-------------|
| Bipolar Chatbot | Gemma-3-1B | 1B | 3 epochs | ~500MB | ~150MB |
| OCD Chatbot | Gemma-3-1B | 1B | 3 epochs | ~500MB | ~150MB |
| Anxiety Chatbot | Gemma-3-1B | 1B | 3 epochs | ~500MB | ~150MB |

##  Technical Details

### Models Used
- **Q&A Generation**: `meta-llama/Llama-3.2-1B-Instruct`
- **Fine-tuning**: `google/gemma-3-1b-it`

### Training Techniques
- **LoRA**: Low-Rank Adaptation (r=16, alpha=32)
- **Quantization**: 4-bit NF4 during training, int4 for mobile
- **Optimization**: BFloat16 precision, paged AdamW optimizer

### Conversion Pipeline
```
PyTorch Model → AI Edge Torch → TFLite → MediaPipe Bundle → .task
```

##  Performance Metrics

- **Training Time**: ~2-3 hours per model (Kaggle GPU)
- **Inference Speed**: ~2-5 tokens/sec on mobile (varies by device)
- **Model Size**: ~150MB per .task bundle

##  Contributing

1. Add new mental health topics by processing additional PDFs
2. Improve Q&A generation prompts for better quality
3. Experiment with different quantization strategies
4. Optimize inference speed for mobile devices

##  License

This project uses:
- Llama 3.2 (Meta License)
- Gemma 3 (Google Terms of Use)
- Ensure compliance with model licenses for commercial use

##  Acknowledgments

- Meta AI for Llama 3.2
- Google for Gemma 3 and AI Edge Torch
- HuggingFace for Transformers and PEFT libraries
- MediaPipe for mobile deployment tools

