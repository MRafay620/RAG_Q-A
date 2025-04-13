# Dr. X's Publications Analysis System

This repository contains the implementation of an NLP-based system to analyze Dr. X's research publications. The system can process various document formats, create a RAG Q&A system, translate documents, and generate summaries.

## Features

- **Text Extraction**: Extract text from .docx, .pdf, .csv, .xlsx, .xls, and .xlsm files
- **Text Chunking**: Break down publications into smaller, manageable parts using the cl100k_base tokenizer
- **Vector Database**: Create embeddings using the nomic-embed-text model and store them in ChromaDB
- **RAG Q&A System**: Answer questions about the publications using retrieved context
- **Translation**: Translate publications between languages using LLM
- **Summarization**: Generate concise summaries of publications with ROUGE evaluation
- **Performance Measurement**: Track tokens per second for all operations

## Translation Model Performance Comparison

We conducted a comparative analysis between two translation models: **llama3.2:latest** and **lauchacarro/qwen2.5-translator:latest**. Here are our findings:

### Performance Characteristics

#### llama3.2:latest

- **Initial Speed**: ~2200 tokens/second
- **Performance Pattern**: U-shaped curve across operations
- **Minimum Speed**: ~150 tokens/second at operations 1-2
- **Final Speed**: ~1250 tokens/second
- **Operation Range**: Spans from 0 to 3

#### qwen2.5-translator:latest

- **Initial Speed**: ~2000 tokens/second
- **Performance Pattern**: Linear decline
- **Final Speed**: ~175 tokens/second
- **Operation Range**: Limited to 0-1

### Usage Recommendations

- **Use llama3.2:latest when**: Processing longer documents where the recovery phase can offset intermediate slowdowns, or when working with complex translation tasks that benefit from multi-stage processing.
- **Use qwen2.5-translator:latest when**: Working with shorter documents where the initial high performance is advantageous, or when needing more predictable processing times.

## Setup and Installation

1. Clone this repository:

```bash
git clone https://github.com/MRafay620/RAD_Q&A.git
cd code
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure Ollama is installed and running on your system:

```bash
# Pull the required models
ollama pull llama3.2:latest
ollama pull nomic-embed-text:latest
# Optional alternative translation model
ollama pull lauchacarro/qwen2.5-translator:latest
```

## Usage

### Process Documents

Process all supported documents in a directory:

```bash
python main.py process --dir Files
```

### Query the System

Ask a question about the processed documents:

```bash
python main.py query --question "What was Dr. X researching?"
```

### Translate Documents

Translate a document from one language to another:

```bash
python main.py translate --file Files\Ocean_ecogeochemistry_A_review.pdf --source "Spanish" --target "English" --model "llama3.2:latest"
```

### Summarize Documents

Generate a summary of a document:

```bash
python main.py summarize --file Files\Ocean_ecogeochemistry_A_review.pdf --ratio 0.3
```

## System Architecture

1. **TextExtractor**: Extracts text from various file formats including tables
2. **TextChunker**: Breaks down texts into manageable chunks using the cl100k_base tokenizer
3. **VectorDatabase**: Creates and manages embeddings with ChromaDB and nomic-embed-text
4. **LanguageModel**: Interfaces with local LLMs for various NLP tasks
5. **DocumentAnalyzer**: Coordinates the entire workflow and analysis process

## Creative Enhancements

### Advanced Chunking

The system implements recursive character splitting with customizable separators, which preserves semantic coherence within chunks better than simple character splits.

### Enhanced Translations

The system uses a specially crafted prompt that instructs the LLM to maintain the original structure, formatting, and technical terminology, making translations more accurate for technical documents.

### Context-Aware Q&A

The RAG system maintains conversation history (previous questions and answers) to provide more coherent responses in multi-turn conversations.

### Cross-Encoder Document Re-ranking

The system implements a two-stage retrieval process: first retrieving candidate documents using embedding similarity, then re-ranking them using a cross-encoder model for more accurate relevance assessment.

### Table Extraction

The system extracts text from tables in document formats and presents it in a readable format, preserving the relationship between table cells.

## Models Used

- **LLM**: llama3.2:latest - A powerful open-source model capable of handling various NLP tasks with high quality
- **Alternative Translation Model**: lauchacarro/qwen2.5-translator:latest - Provides different performance characteristics for translation tasks
- **Embedding Model**: nomic-embed-text:latest - A local embedding model that provides high-quality semantic embeddings
- **Cross-Encoder Model**: cross-encoder/ms-marco-MiniLM-L-6-v2 - Used for document re-ranking to improve retrieval quality

## Performance Evaluation

The system tracks tokens per second for all operations:

- Embedding generation
- Document retrieval
- Answer generation
- Translation
- Summarization

This allows for direct performance comparison and optimization, as demonstrated in our translation model analysis.

## Limitations

- Requires Ollama to be installed and configured on the local machine
- Only uses local models, which may have lower quality than state-of-the-art cloud-based models
- Table extraction preserves content but not exact visual layout
- Translation quality depends on the capabilities of the underlying LLM
- Different translation models show varying performance characteristics across operations

## Future Improvements

- Implement more advanced chunking strategies based on semantic units
- Add support for more file formats (e.g., HTML, XML, etc.)
- Implement hybrid search combining embedding similarity with keyword matching
- Add automatic language detection for translation
- Create a web interface for easier interaction with the system
- Further optimize model selection based on document characteristics and task requirements
