# 🔍 Detector Game

An AI-powered detective game where you interrogate suspects to uncover the criminal. Built with **FastAPI**, **LangGraph**, and modern LLM providers.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-purple.svg)

## 🎮 Game Overview

In Detector Game, you play as a detective investigating a crime scene. The AI generates a unique mystery scenario with multiple suspects, each with their own personality, alibi, and secrets. Your job is to:

1. **Analyze the case** - Read through the crime summary and clues
2. **Interrogate suspects** - Ask questions to uncover contradictions and secrets
3. **Track suspicion levels** - Watch as the AI analyzes each suspect's responses
4. **Make your accusation** - When you're confident, accuse the suspect you believe is guilty

## ✨ Features

- 🎲 **Dynamic Scenario Generation** - Every game creates a unique crime story with different suspects
- 🤖 **Multiple LLM Support** - Choose between OpenAI GPT, Google Gemini, or Qwen models
- 💬 **Realistic Interrogations** - Suspects respond based on their personality and guilt level
- 📊 **Suspicion Tracking** - AI analyzes responses and updates suspicion scores in real-time
- 🌐 **Simple Web UI** - Clean HTML/CSS/JS frontend for easy gameplay

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- An API key from one of the supported LLM providers:
  - OpenAI (`OPENAI_API_KEY`)
  - Google Gemini (`GOOGLE_API_KEY`)
  - Qwen/DashScope (`QWEN_API_KEY`)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/detector-game.git
   cd detector-game/src
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.sample .env
   # Edit .env and add your API key(s)
   ```

### Running the Game

Start the development server:

```bash
uvicorn backend.api:app --reload
```

Then open your browser and navigate to: **http://localhost:8000**

## 🏗️ Project Structure

```
src/
├── backend/
│   ├── api.py           # FastAPI endpoints
│   ├── graph.py         # LangGraph game state machine
│   └── llm_strategy.py  # LLM providers (OpenAI, Gemini, Qwen)
├── static/
│   ├── index.html       # Web UI
│   ├── style.css        # Styling
│   └── app.js           # Frontend logic
├── test/                # Test files
├── requirements.txt     # Python dependencies
└── .env.sample          # Environment template
```

## 🔧 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve the web UI |
| `/api/new_game` | POST | Start a new game with generated scenario |
| `/api/ask` | POST | Ask a question to a suspect |
| `/api/accuse` | POST | Accuse a suspect as the criminal |

### Example: Start a New Game

```bash
curl -X POST http://localhost:8000/api/new_game \
  -H "Content-Type: application/json" \
  -d '{"num_suspects": 4}'
```

### Example: Ask a Question

```bash
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{
    "game_id": "your-game-id",
    "suspect_id": "suspect-1",
    "question": "Where were you on the night of the murder?"
  }'
```

## 🤖 Supported LLM Providers

The game uses a **Strategy Pattern** to support multiple LLM providers:

| Provider | Model | Environment Variable |
|----------|-------|---------------------|
| OpenAI | gpt-4o-mini | `OPENAI_API_KEY` |
| Google | gemini-2.0-flash | `GOOGLE_API_KEY` |
| Qwen | qwen-plus | `QWEN_API_KEY` |

The system automatically selects an available provider based on which API key is configured.

## 🛠️ Technology Stack

- **Backend**: FastAPI, Python
- **AI Framework**: LangGraph, LangChain
- **LLM Providers**: OpenAI, Google Gemini, Qwen
- **Frontend**: Vanilla HTML/CSS/JavaScript

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🤝 Contributing

Contributions are welcome! Feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

**Built with ❤️ using FastAPI + LangGraph**
