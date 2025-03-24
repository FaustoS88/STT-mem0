# 🎙️ STT with Memory - AI Companion

A conversational AI system combining speech recognition using fast-whisper, long-term memory with mem0 + postgresSQL and contextual awareness through two implementation versions.

## Features

- **Speech-to-Text**: High-quality transcription using Faster-Whisper
- **Long-term Memory**: Vector storage with Mem0 and PostgreSQL/pgvector
- **Multiple LLM Support**: OpenRouter or DeepSeek integration
- **Dual Interaction Modes**: Voice input or text-based interaction
- **Contextual Responses**: AI responses informed by conversation history

## Implementation Versions

### STTMem0.py (Basic)
- Individual message storage in Mem0
- Simple context retrieval
- Direct text interaction
- Minimal resource usage

### STTMem0V2.py (Enhanced)
- Hybrid memory architecture:
  - Short-term conversation context window
  - Long-term Mem0 storage
  - Conversation chunks preserved as memories
- Additional features:
  - Conversation history viewing (`h`)
  - Memory count check (`m`)
  - JSON conversation export
  - Improved context awareness

## Technical Implementation

### Core Components
```python
# Memory Configuration (both versions)
self.memory = Memory.from_config({
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-large",
            "api_key": OPENAI_API_KEY
        }
    },
    "vector_store": {
        "provider": "pgvector",
        "config": {
            "user": "postgres",
            "password": "postgres",
            "host": "127.0.0.1",
            "port": "54329",
            "dbname": "postgres"
        }
    }
})

# V2 Enhanced Memory Handling
def add_to_memory(self, text, speaker="user"):
    # Store individual message
    self.memory.add(messages=[{"role": speaker, "content": text}], user_id="default")
    
    # Every 4 messages, store conversation chunk
    if len(self.conversation_history) >= 4 and len(self.conversation_history) % 4 == 0:
        recent_convo = self.conversation_history[-4:]
        convo_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_convo])
        self.memory.add(
            messages=[{"role": "system", "content": f"Conversation exchange: {convo_text}"}],
            user_id="default",
            metadata={"type": "conversation_chunk"}
        )
```

## ⚙️ Setup & Configuration

1. **Database Setup**:
```bash
docker run --name mem0-pgvector \
  -e POSTGRES_PASSWORD=postgres \
  -p 54329:5432 \
  -v ~/mem0_postgres_data:/var/lib/postgresql/data \
  -d pgvector/pgvector:pg16
```

2. **Environment Variables** (`.env`):
```ini
# Required
OPENAI_API_KEY=your_key
DATABASE_URL=postgresql://postgres:postgres@localhost:54329/postgres

# LLM Provider (choose one)
DEEPSEEK_API_KEY=your_key  # OR
OPENROUTER_API_KEY=your_key
LLM_PROVIDER=deepseek|openrouter # they both use the same Deepseek v3 0324 new model version released today! (03/24/25)
```

3. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

## Running the System

### Basic Version
```bash
python STTMem0.py
```

### Enhanced Version (Recommended)
```bash
python STTMem0V2.py
```

## Controls & Interaction
- Press Enter to start recording
- Press Enter again to stop recording
- Type 't' to switch to text input mode
- Type 'q' to quit the application
V2 Additional Controls :

- 'm' - Check memory count
- 'h' - View conversation history
- 's' - Save conversation to JSON

## Memory Architecture

### V1 Structure
- Individual messages stored as separate memories
- Simple vector search for context

### V2 Improvements
1. Short-term context window (last 5 interactions)
2. Conversation chunk storage (every 4 messages)
3. Metadata tagging for different memory types
4. Integrated memory inspection tools

## 📊 Performance Notes

| Aspect          | STTMem0.py | STTMem0V2.py |
|-----------------|------------|--------------|
| Memory Usage    | Low        | Moderate     |
| Context Quality | good       | High         |
| Response Time   | Fast       | Slightly slower |
| Storage Efficiency|Excellent | Good         |

## Troubleshooting

1. **Database Issues**:
   - Verify PostgreSQL is running: `docker ps`
   - Check connection string in `.env`

2. **LLM Errors**:
   - Confirm API keys are valid
   - If you see Incorrect API key provided , export your key: export OPENAI_API_KEY=your_key

## Memory Management**:
- Use `memory_manager.py` to inspect/clear memories
- Check table exists: `psql -h 127.0.0.1 -p 54329 -U postgres -d postgres -c "\dt"`

## System Requirements
- Python 3.9+
- Docker (for PostgreSQL/pgvector)
- FFmpeg (installed automatically with dependencies)
- Microphone for speech input
- 4GB+ RAM recommended

## License
MIT License - See [LICENSE](LICENSE) for details

## Credits
Built with:

- Mem0 - Memory system
- Faster-Whisper - Speech recognition
- DeepSeek / OpenRouter - LLM providers
