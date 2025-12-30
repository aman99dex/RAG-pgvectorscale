# Pgvectorscale RAG Solution Setup

A high-performance Retrieval-Augmented Generation (RAG) solution built with PostgreSQL, pgvectorscale (Timescale Vector), and Python. This project demonstrates how to leverage vector embeddings for semantic search and combine it with large language models to generate contextually relevant answers from a knowledge base.

## What It Does

This project implements a complete RAG pipeline for an e-commerce FAQ system:

- **Vector Storage**: Uses Timescale's pgvectorscale extension on PostgreSQL to store and index high-dimensional vector embeddings
- **Semantic Search**: Performs similarity search on vectorized FAQ data using cosine distance
- **Answer Synthesis**: Leverages LLMs (OpenAI GPT, Anthropic Claude, or local models) to generate coherent answers based on retrieved context
- **Hybrid Search**: Supports advanced retrieval techniques including time-based filtering and approximate nearest neighbor (ANN) search

The system takes user questions, finds semantically similar FAQ entries from the knowledge base, and synthesizes helpful responses using the retrieved context.

## Features

- **Vector Database**: PostgreSQL with pgvectorscale for efficient vector storage and search
- **Multiple LLM Providers**: Support for OpenAI, Anthropic, and local LLM endpoints
- **Embedding Generation**: Uses OpenAI's text-embedding-3-small model for creating vector representations
- **Structured Output**: Uses Instructor library for type-safe LLM responses
- **Docker Setup**: Easy deployment with Docker Compose
- **Configurable**: Environment-based configuration with Pydantic settings
- **Logging**: Comprehensive logging for monitoring and debugging

## Prerequisites

- Docker and Docker Compose
- Python 3.7+
- OpenAI API key (for embeddings and optional LLM usage)
- Optional: Anthropic API key or local LLM endpoint

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd pgvectorscale-rag-solution-setup
   ```

2. **Set up environment variables**:
   ```bash
   cp app/example.env .env
   ```
   Edit `.env` and add your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   # Optional: ANTHROPIC_API_KEY=your_anthropic_key
   # Optional: LLAMA_BASE_URL=your_local_llm_url
   ```

3. **Start the database**:
   ```bash
   cd docker
   docker compose up -d
   ```

4. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

The application uses Pydantic-based configuration loaded from environment variables. Key settings include:

- **Database**: Timescale service URL (automatically configured for local Docker setup)
- **LLM Settings**: Model selection, temperature, max tokens, retries
- **Vector Store**: Table name, embedding dimensions, time partitioning
- **Embeddings**: OpenAI embedding model (default: text-embedding-3-small)

See `app/config/settings.py` for complete configuration options.

## Usage

### 1. Populate the Vector Database

Run the vector insertion script to embed and store the FAQ data:

```bash
python app/insert_vectors.py
```

This will:
- Load the FAQ dataset from `data/faq_dataset.csv`
- Generate embeddings using OpenAI's API
- Store vectors in the Timescale database with time partitioning

### 2. Perform Similarity Search

Use the similarity search script to query the knowledge base:

```bash
python app/similarity_search.py
```

The script demonstrates:
- Query embedding generation
- Vector similarity search using cosine distance
- Retrieval of relevant context
- Answer synthesis using configured LLM

### 3. Customize and Extend

- **Add new data**: Update `data/faq_dataset.csv` and re-run insertion
- **Change LLM provider**: Modify settings in `app/config/settings.py`
- **Implement hybrid search**: Extend `VectorStore` class with additional filtering
- **Add indexing**: Create ANN indexes for better performance on large datasets

## Project Structure

```
├── app/
│   ├── config/
│   │   └── settings.py          # Application configuration
│   ├── database/
│   │   └── vector_store.py      # Vector operations and DB client
│   ├── services/
│   │   ├── llm_factory.py       # LLM provider abstraction
│   │   └── synthesizer.py       # Answer synthesis logic
│   ├── insert_vectors.py        # Data ingestion script
│   ├── similarity_search.py     # Query and retrieval script
│   └── example.env              # Environment template
├── data/
│   └── faq_dataset.csv          # Sample FAQ data
├── docker/
│   └── docker-compose.yml       # Database setup
├── requirements.txt             # Python dependencies
├── LICENCE                     # License file
└── README.md                   # This file
```

## Key Components

### VectorStore Class
Handles all vector operations:
- Embedding generation with OpenAI
- Database connections via Timescale Vector client
- Similarity search with configurable limits
- Time-based partitioning for efficient queries

### LLM Factory
Provides unified interface for different LLM providers:
- OpenAI GPT models
- Anthropic Claude models
- Local/self-hosted models via OpenAI-compatible APIs

### Synthesizer
Generates structured responses:
- Thought process tracking
- Context-aware answer synthesis
- Confidence assessment for answer completeness

## Performance Optimization

For production use with large datasets (>10k vectors):

1. **Create ANN Indexes**:
   ```sql
   CREATE INDEX ON embeddings USING timescale_vector (embedding);
   ```

2. **Tune PostgreSQL**:
   - Adjust `shared_buffers` and `work_mem`
   - Configure connection pooling

3. **Batch Processing**: Modify insertion scripts for bulk operations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

See LICENCE file for details.

## Resources

- [Timescale Vector Documentation](https://github.com/timescale/pgvectorscale)
- [pgvector Extension](https://github.com/pgvector/pgvector)
- [Instructor Library](https://github.com/jxnl/instructor)
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
