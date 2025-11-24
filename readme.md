# Mental Health Chatbot - Mobile AI Assistant ([Sample Video](https://drive.google.com/file/d/1jmju8upvCVEVkJ_QUbQax5TwPpLvHG-u/view?usp=sharing))

A mobile-optimized AI chatbot system with intelligent orchestration for mental health topics (Anxiety, Bipolar, Depression, OCD, Schizophrenia). The system uses a classifier-based orchestrator to route user queries to specialized fine-tuned models that run entirely on-device using TensorFlow Lite.

## 🎯 Project Overview

This project creates an intelligent mental health chatbot system that runs on mobile devices without internet connectivity. It features an orchestrator that intelligently routes user queries to specialized models, each fine-tuned on specific mental health conditions. The system processes mental health literature, generates conversational Q&A pairs, fine-tunes lightweight language models, and deploys them with smart routing capabilities.

### Key Features
- 🧠 **Intelligent Orchestration**: TF-IDF + Logistic Regression classifier routes queries to the most relevant specialized model
- 🤖 **Multi-Agent Architecture**: Separate fine-tuned models for Anxiety, Bipolar, Depression, OCD, and Schizophrenia
- 📚 **Automated Q&A Generation**: Extracts knowledge from PDF books and generates conversational question-answer pairs
- 🎓 **Fine-tuned Models**: Custom models trained on specific mental health topics
- 📱 **Mobile-Ready**: Converted to TensorFlow Lite format for on-device inference
- 🔒 **Privacy-First**: Runs completely offline on mobile devices
- ⚡ **Optimized**: Dynamic int4 quantization for efficient mobile performance

## 📂 Project Structure

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
├── mental_health_orchestrator.pkl  # Weights for PC
├── orchestrator.ipynb        # Orchestrator training and testing
├── MOE.ipynb                 # Mixture of Experts experimentation
├── run_chat.ipynb            # Test chat interface
├── train_and_merge.ipynb     # Model fine-tuning and LoRA merging
└── README.md                 # This file
```

**Resources:**
- [Results](https://drive.google.com/drive/folders/15dC9hV4wE5AbOgiIZ0fd-k9frnbY-2Cl?usp=sharing) - Results on Drive
- [Tflite](https://drive.google.com/drive/folders/13ezzGJjhfwZYFZrWFJ1tb50R2hORGTSr?usp=sharing) - Tflite is the intermediate step or you can say quantized version before deployment
- [Task](https://drive.google.com/drive/folders/1-Q738xri5ZWYKD8OtMuVBA7T9cfuKfvH?usp=sharing) - Format that allows deployment
- [Data](https://drive.google.com/drive/folders/1NcERdwz8T4eSXQKUQG59Ty9ymKV0DPN5?usp=sharing) - The books gemma3 trained on

## 🚀 Getting Started

### Prerequisites

```bash
# Python 3.10+
pip install torch transformers
pip install peft datasets
pip install PyPDF2 tqdm
pip install ai-edge-torch
pip install mediapipe
pip install scikit-learn
```

### Required Resources
- **GPU**: Recommended for training (Kaggle/Colab GPU works)
- **RAM**: 16GB+ recommended
- **Storage**: ~5GB per fine-tuned model
- **HuggingFace Token**: Required for Gemma model access

## 🔄 Workflow

### Step 0: Train the Orchestrator (NEW!)

**Notebook**: `orchestrator.ipynb`

The orchestrator is a lightweight classifier that analyzes user queries and routes them to the most appropriate specialized model.

**How it Works**:
1. **Feature Extraction**: Uses TF-IDF vectorization (5000 features, bigrams) on Q&A pairs
2. **Classification**: Multinomial Logistic Regression predicts the relevant mental health category
3. **Routing**: Directs the query to the corresponding fine-tuned model


**Training the Orchestrator**:
```python
from mental_health_orchestrator import MentalHealthOrchestrator

# Initialize and train
orchestrator = MentalHealthOrchestrator(data_dir="qa_pair")
orchestrator.train(test_size=0.2)

# Save for deployment
orchestrator.save_model("orchestrator_model.pkl")
```

**Using the Orchestrator**:
```python
# Load trained orchestrator
orchestrator = MentalHealthOrchestrator()
orchestrator.load_model("orchestrator_model.pkl")

# Route user query
user_query = "I've been feeling extremely anxious lately"
predicted_class, probabilities = orchestrator.predict(user_query)

print(f"Route to: {predicted_class}")
print(f"Confidence: {probabilities}")
# Output: Route to: anxiety
#         Confidence: {'anxiety': 0.85, 'depression': 0.08, ...}
```

**Model Classes**:
- `anxity` - Anxiety disorders and panic attacks (note: typo in training data)
- `bipolar` - Bipolar disorder and mood swings
- `depresion` - Depression and low mood (note: typo in training data)
- `ocd` - Obsessive-compulsive disorder
- `schiz` - Schizophrenia and psychotic symptoms

### Step 1: Generate Q&A Pairs from PDFs

**Notebook**: `book-to-qa.ipynb`

Extracts text from mental health PDFs and generates conversational Q&A pairs using Llama-3.2-1B-Instruct.

```python
# Configuration in book-to-qa.ipynb
books = ['anxiety', 'bipolar', 'depression', 'ocd', 'schiz']
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
book = "anxiety"  # or "bipolar", "depression", "ocd", "schiz"
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

## 📱 Mobile Deployment

### Architecture Overview

```
User Query → Orchestrator Classifier → Route to Specialized Model → Response
                    ↓
     [Anxiety | Bipolar | Depression | OCD | Schiz]
```

### Using Google AI Edge Gallery

1. **Transfer Files**: Copy all `.task` files to your Android device
2. **Import Models**: Open AI Edge Gallery app → '+' button → Import each specialized model
3. **Orchestration Layer**: Implement the orchestrator logic in your mobile app to route queries
4. **Test Chat**: Start conversing with your intelligent mental health assistant

### System Requirements
- Android 8.0+ (API level 26+)
- 2GB+ RAM
- ~750MB storage for all 5 models
- Optional: 50MB for orchestrator model

## 🧪 Testing

**Notebook**: `run_chat.ipynb`

Test the fine-tuned models and orchestrator before mobile conversion:

```python
# Load orchestrator
from mental_health_orchestrator import MentalHealthOrchestrator
orchestrator = MentalHealthOrchestrator()
orchestrator.load_model("orchestrator_model.pkl")

# Route query
query = "What helps manage anxiety?"
predicted_model, probs = orchestrator.predict(query)

# Load appropriate model
model = AutoModelForCausalLM.from_pretrained(f"./results/gemma-{predicted_model}-merged")
tokenizer = AutoTokenizer.from_pretrained(f"./results/gemma-{predicted_model}-merged")

# Generate response
response = generate_response(model, tokenizer, query)
```

## 🔬 Experiments

**Notebook**: `MOE.ipynb`

Experimental Mixture of Experts (MoE) architecture for combining multiple specialized models.

## 📊 Model Specifications

| Model | Base | Parameters | Training | Output Size | Mobile Size |
|-------|------|------------|----------|-------------|-------------|
| Anxiety Chatbot | Gemma-3-1B | 1B | 3 epochs | ~490MB | ~550MB |
| Bipolar Chatbot | Gemma-3-1B | 1B | 3 epochs | ~490MB | ~550MB |
| Depression Chatbot | Gemma-3-1B | 1B | 3 epochs | ~490MB | ~550MB |
| OCD Chatbot | Gemma-3-1B | 1B | 3 epochs | ~490MB | ~550MB |
| Schizophrenia Chatbot | Gemma-3-1B | 1B | 3 epochs | ~490MB | ~550MB |
| Orchestrator | TF-IDF + LogReg | ~5K features | N/A | ~50MB | ~50MB |

### Orchestrator Performance
- **Accuracy**: ~85-95% (depends on training data quality)
- **Inference**: <10ms on CPU
- **Classes**: 5 mental health categories
- **Features**: TF-IDF with bigrams, 5000 max features

## 🔧 Technical Details

### System Architecture
1. **Orchestrator Layer**: TF-IDF vectorization + Logistic Regression classifier
2. **Specialized Agents**: 5 fine-tuned Gemma-3-1B models
3. **Routing Logic**: Probability-based classification to select best agent
4. **Inference**: On-device TFLite execution

### Models Used
- **Q&A Generation**: `meta-llama/Llama-3.2-1B-Instruct`
- **Fine-tuning**: `google/gemma-3-1b-it`
- **Orchestrator**: Scikit-learn (TfidfVectorizer + LogisticRegression)

### Training Techniques
- **LoRA**: Low-Rank Adaptation (r=16, alpha=32)
- **Quantization**: 4-bit NF4 during training, int4 for mobile
- **Optimization**: BFloat16 precision, paged AdamW optimizer

### Conversion Pipeline
```
PyTorch Model → AI Edge Torch → TFLite → MediaPipe Bundle → .task
```

## ⚡ Performance Metrics

- **Training Time**: ~2-3 hours per model (Kaggle GPU)
- **Orchestrator Training**: ~5-10 minutes (CPU)
- **Routing Inference**: <10ms per query (CPU)
- **Model Inference**: ~2-5 tokens/sec on mobile (varies by device)
- **Model Size**: ~150MB per .task bundle
- **Total System**: ~750MB for all 5 models + 50MB orchestrator

## 🤝 Contributing

1. **Add new mental health topics** by processing additional PDFs and training new specialized models
2. **Improve orchestrator accuracy** by enhancing Q&A pair quality or experimenting with advanced classifiers
3. **Optimize routing logic** with confidence thresholds or multi-model ensemble approaches
4. **Improve Q&A generation** prompts for better training data quality
5. **Experiment with different quantization** strategies for smaller model sizes
6. **Optimize inference speed** for mobile devices with model distillation

## 📄 License

This project uses:
- Llama 3.2 (Meta License)
- Gemma 3 (Google Terms of Use)
- Ensure compliance with model licenses for commercial use

## 🙏 Acknowledgments

- Meta AI for Llama 3.2
- Google for Gemma 3 and AI Edge Torch
- HuggingFace for Transformers and PEFT libraries
- MediaPipe for mobile deployment tools
- Scikit-learn for orchestration classifier