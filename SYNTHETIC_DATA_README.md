# Synthetic Data Generator

Generates training data for Planuze LLM fine-tuning by processing PDFs and creating Q&A pairs using Ollama.

## 🎯 Features

- **PDF Processing**: Automatically reads all PDFs from `data/source_documents`
- **Intelligent Chunking**: Splits documents with overlap to maintain context
- **Dual Mode Generation**:
  - **ASK Mode**: Informative Q&A pairs from document content
  - **ACTION Mode**: System action examples with proper JSON formatting
- **Production Format**: Generates data in the exact format used in production (XML tags, @ prefix for IDs)
- **Multi-language**: Supports both Portuguese and English examples
- **Ollama Integration**: Uses local Ollama for generation (no API costs)

## 📋 Prerequisites

1. **Ollama Running**: Make sure Ollama is running locally
   ```bash
   ollama serve
   ```

2. **Model Downloaded**: Ensure you have the generator model
   ```bash
   ollama pull qwen2.5:14b
   # or your preferred model
   ```

3. **PDFs Ready**: Place your source documents in `data/source_documents/`

## ⚙️ Configuration

All settings are in `envs/.env.global`:

```bash
# Ollama Configuration
OLLAMA_HOST=http://localhost:11434
SYNTHETIC_GENERATOR_MODEL="qwen2.5:14b"

# Chunking Settings
SYNTHETIC_CHUNK_SIZE=3500        # Characters per chunk
SYNTHETIC_OVERLAP=500            # Overlap between chunks

# Paths
SYNTHETIC_SOURCE_DIR="data/source_documents"
SYNTHETIC_OUTPUT_FILE="data/raw/train_data_synthetic.jsonl"

# Prompts (automatically loaded from .env)
AI_PROMPT_ASK_INSTRUCTIONS="..."
AI_PROMPT_ACTION_INSTRUCTIONS="..."
```

## 🚀 Usage

### Quick Start

```bash
make synthetic
```

This will:
1. Read all PDFs from `data/source_documents/`
2. Chunk the content intelligently
3. Generate ASK mode examples from document content
4. Generate ACTION mode examples with realistic company data
5. Save to `data/raw/train_data_synthetic.jsonl`

### Manual Execution

```bash
# Activate virtual environment
source .venv/bin/activate

# Run generator
python -m src.application.generate_synthetic
```

### Customization

Edit `src/application/generate_synthetic.py` to:
- Adjust number of ACTION examples (default: 50)
- Modify example generation logic
- Add new action types
- Change sampling strategies

## 📊 Output Format

The generator creates data in **Alpaca format**:

```json
{
  "instruction": "<full_prompt_with_xml_tags>",
  "input": "",
  "output": "<expected_response>"
}
```

### ASK Mode Example

```json
{
  "instruction": "<system_instructions>...</system_instructions>\n<mode>ASK</mode>\n<context>...</context>\n<question>What is LGPD?</question>\n<language>pt</language>",
  "input": "",
  "output": "LGPD é a Lei Geral de Proteção de Dados..."
}
```

### ACTION Mode Example

```json
{
  "instruction": "<system_action_required>...</system_action_required>\n<mode>ACTION</mode>\n<context>...</context>\n<question>Change company name to NewCorp</question>\n<language>en</language>",
  "input": "",
  "output": "{\"action_id\":null,\"subject\":\"company\",\"action\":\"update\",\"message\":\"I will change your company name from **OldCorp** to **NewCorp**.\",\"payload\":{\"p\":{\"id\":\"cml_123\"},\"b\":{\"name\":\"NewCorp\"}}}"
}
```

## 🔍 Data Quality

The generator ensures:

- ✅ **Exact Production Format**: Matches `PROMPT_FORMAT_SPEC.md`
- ✅ **Semantic JSON**: Uses `@` prefix for metadata (IDs, timestamps)
- ✅ **Language Consistency**: Response language matches `<language>` tag
- ✅ **Context Relevance**: Questions answerable from provided context
- ✅ **Action Validity**: Proper JSON structure with all required fields

## 📈 Typical Output

From 19 PDFs with ~10 chunks each:
- **ASK Examples**: ~380-570 (2-3 per chunk)
- **ACTION Examples**: 50 (configurable)
- **Total**: ~430-620 training examples
- **Languages**: ~50% Portuguese, ~50% English

## 🛠️ Troubleshooting

### "No PDF files found"
- Check that PDFs are in `data/source_documents/`
- Verify `SYNTHETIC_SOURCE_DIR` in `.env.global`

### "Ollama API error"
- Ensure Ollama is running: `ollama serve`
- Check `OLLAMA_HOST` in `.env.global`
- Verify model is downloaded: `ollama list`

### "Failed to parse Q&A"
- The LLM might be returning non-JSON
- Try a different model (e.g., `llama3.1`)
- Adjust temperature in `call_ollama()` method

### Low quality examples
- Use a larger/better model (e.g., `qwen2.5:14b` instead of `7b`)
- Reduce `SYNTHETIC_CHUNK_SIZE` for more focused context
- Increase `temperature` for more variety

## 🔄 Next Steps

After generation:

1. **Review the data**:
   ```bash
   head -n 5 data/raw/train_data_synthetic.jsonl
   ```

2. **Prepare for training**:
   ```bash
   make mlx:prepare
   # or
   make cuda:prepare
   ```

3. **Train the model**:
   ```bash
   make mlx:train
   # or
   make cuda:train
   ```

## 📝 Notes

- Generation time depends on:
  - Number of PDFs
  - Chunk size
  - Ollama model speed
  - Number of ACTION examples
- Typical runtime: 10-30 minutes for 19 PDFs
- The generator uses `rich` for beautiful progress bars
- Data is automatically shuffled before saving

---

**See also**: `PROMPT_FORMAT_SPEC.md` for detailed prompt structure documentation
