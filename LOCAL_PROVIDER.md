# Local AI Provider Setup Guide

This guide explains how to run Billy B-Assistant completely offline using local AI models.

## Overview

The Local Provider uses:
- **Whisper** (faster-whisper) - Speech-to-text transcription
- **Ollama** - Local LLM for conversation (Llama 3, Mistral, etc.)
- **Piper TTS** - Text-to-speech synthesis (coming soon)

## Prerequisites

### 1. Install Ollama

**Windows:**
```bash
# Download from https://ollama.com/download
# Or use winget
winget install Ollama.Ollama
```

**Mac/Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. Pull a Model

```bash
# Recommended: Llama 3.2 (smaller, faster)
ollama pull llama3.2:latest

# Or use a larger model for better quality
ollama pull llama3.1:8b
```

### 3. Install Python Dependencies

```bash
pip install faster-whisper
```

## Configuration

### Enable Local Provider

Add to your `.env` file:

```ini
# Use local provider instead of OpenAI
REALTIME_AI_PROVIDER=local

# Optional: Configure Ollama settings
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2:latest

# Optional: Whisper model size (tiny, base, small, medium, large)
WHISPER_MODEL=base
```

### Or Just Remove OpenAI Key

If you remove `OPENAI_API_KEY` from your `.env`, Billy will automatically use the local provider.

## Usage

1. **Start Ollama** (if not running):
   ```bash
   ollama serve
   ```

2. **Run Billy**:
   ```bash
   python main.py
   ```

3. Billy will now use:
   - Local Whisper for transcription
   - Local Ollama for conversation
   - (TTS coming soon)

## Model Recommendations

| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| `llama3.2:1b` | 1.3GB | Very Fast | Good | Testing, low-end hardware |
| `llama3.2:latest` (3B) | 2GB | Fast | Very Good | Recommended for most users |
| `llama3.1:8b` | 4.7GB | Medium | Excellent | Best quality |
| `mistral:latest` | 4.1GB | Medium | Excellent | Alternative to Llama |
| `phi3:mini` | 2.3GB | Fast | Good | Microsoft's small model |

## Performance Tips

1. **Use smaller models** - 3B models are fast and good enough for conversation
2. **GPU acceleration** - Ollama automatically uses GPU if available
3. **Adjust Whisper model** - Use `tiny` or `base` for speed, `small` for quality
4. **Quantization** - Ollama models are pre-quantized for efficiency

## Troubleshooting

### Ollama Not Connecting
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama
ollama serve
```

### Whisper Too Slow
Use a smaller model in `.env`:
```ini
WHISPER_MODEL=tiny  # Fastest
```

### Out of Memory
Use a smaller Ollama model:
```bash
ollama pull llama3.2:1b
```

## Next Steps

- [ ] Implement Piper TTS integration for local speech synthesis
- [ ] Add function calling support for local models
- [ ] Support for streaming responses
- [ ] GPU acceleration options

## Current Limitations

- TTS not yet implemented (returns silence placeholder)
- No streaming responses yet (full response generation)
- Function calling needs testing with local models

## Contributing

Help us improve the local provider! Priority areas:
1. Piper TTS integration
2. Streaming response support
3. Better function calling for local models
4. Performance optimizations
