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


## Setup and Installation

Choose one of the following options to set up the system.

### Option 1: Setup and Installation (Using Docker)

This option uses Docker to ensure a consistent and reproducible environment.

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (required for GPU support)


#### System Requirements
- **Operating System**: Linux (tested on Ubuntu 22.04 or later), macOS, or Windows with WSL2
- **Optional**: NVIDIA GPU with CUDA support for GPU-accelerated inference
- **Storage**: At least 10GB of free disk space for models and data


#### Steps
1. **Clone the repository**:
   ```bash
   git clone https://github.com/MRafay620/RAD_Q&A.git
   cd RAD_Q&A
   ```

2. **Review Docker configuration**:
   - Check the `docker-compose.yml` file to customize port mappings or environment variables if needed.

3. **Build and start containers**:
   ```bash
   docker compose up -d
   ```
   This will:
   - Pull the `ollama/ollama:latest` image from Docker Hub.
   - Build the main application image (`osos_raffay:latest`) using the provided `Dockerfile`.
   - Start both containers (`ollama` and `main_app`) in a Docker network.

4. **Access the application**:
   - Open the Streamlit interface in your web browser at:
     ```
     http://localhost:8051
     ```

5. **Verify container status** (optional):
   ```bash
   docker compose ps
   ```

#### Notes
- **GPU Support**: If an NVIDIA GPU and the NVIDIA Container Toolkit are available, the `ollama` container will use GPU acceleration. Otherwise, it defaults to CPU mode.
- **Persisting Data**: Models and data are stored in a named volume (`ollama_data`) to persist across container restarts.
- **Stop Containers**:
   ```bash
   docker compose down
   ```
- **Troubleshooting**:
   - If `localhost:8051` is unavailable, modify the port mapping in `docker-compose.yml` (e.g., change to `"8080:8051"`).
   - Check logs with:
     ```bash
     docker compose logs
     ```

### Option 2: Setup and Installation (Simple, Non-Docker)

This option provides a straightforward setup for running the system directly on your local machine.

#### System Requirements
- **Operating System**: Linux, macOS, or Windows
- **Python**: Version 3.8 or higher
- **Storage**: At least 10GB of free disk space for models and data

#### Prerequisites
- Python 3.8+ installed
- [Ollama](https://ollama.ai/) installed and running locally

#### Steps
1. **Clone the repository**:
   ```bash
   git clone https://github.com/MRafay620/RAD_Q&A.git
   cd RAD_Q&A/app
   ```

2. **Create and activate a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Ollama models**:
   - Ensure Ollama is running on your system.
   - Pull the required models:
     ```bash
     ollama pull llama3.2:latest
     ollama pull nomic-embed-text:latest
     # Optional translation model
     ollama pull lauchacarro/qwen2.5-translator:latest
     ```

5. **Run the application**:
   - Start the Streamlit interface:
     ```bash
     streamlit run app.py
     ```
   - Access it in your web browser at:
     ```
     http://localhost:8051
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
python main.py translate --file Files\Ocean_ecogeochemistry_A_review.pdf --source "English" --target "German" --model "llama3.2:latest"
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

# Performance Comparison

Below is a performance comparison between two models:

## llama3.2 vs qwen2.5-translator

![llama3.2 performance](https://github.com/user-attachments/assets/0dd64411-11ca-4801-a0f6-e50e85833181)
*llama3.2:latest*

vs

![qwen2.5-translator performance](https://github.com/user-attachments/assets/c1ef1b69-985c-4818-b5c8-948f0a160c18)
*qwen2.5-translator:latest*

The charts above demonstrate the relative performance characteristics of each model. For detailed analysis of these results, please refer to the [evaluation section](#evaluation).

# App Result:

![Image](https://github.com/user-attachments/assets/3a2015f8-2c24-4f77-854b-8d55a7fa1322)
![Image](https://github.com/user-attachments/assets/a4d8cdeb-ea1f-4cbf-902b-5e1991d0da13)
![Image](https://github.com/user-attachments/assets/d10e544e-e6e2-4228-bd81-9aa9393e47d2)
![Image](https://github.com/user-attachments/assets/2eec8989-f437-46b1-8186-528d1df95340)
![Image](https://github.com/user-attachments/assets/ee5460ad-6cb7-4d70-a4c2-26026ebd08f2)
![Image](https://github.com/user-attachments/assets/a27adf47-a8e7-4f47-8db6-0b6471f423a1)
![Image](https://github.com/user-attachments/assets/54fa3bd2-50df-4658-92a5-f23e8dde4230)
![Image](https://github.com/user-attachments/assets/7a4ca0ec-0535-4e84-b822-f11416178ec3)

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
