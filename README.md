# 🧠 AI Chat Application with OpenAI, Grok (xAI), and Vector Search

This is a lightweight FastAPI-powered chat API that supports semantic context retrieval using HuggingFace embeddings and cosine similarity. It supports both **OpenAI GPT-3.5** and **Grok (xAI)** as pluggable LLM providers.

---

## 🚀 Features

- 🔍 Contextual retrieval using `thenlper/gte-small` from HuggingFace
- 🧠 Cosine similarity-based vector search (no FAISS required)
- 💬 OpenAI GPT-3.5 chat integration
- 🤖 Grok (xAI) chat provider support
- ⚡ FastAPI backend
- 🛡️ `.env` support for secrets

---

## 📁 Project Structure

AI_Application/
├── app.py # FastAPI app entrypoint
├── chat_provider.py # OpenAI & Grok logic
├── vector_store.py # HuggingFace embedding & search logic
├── build_vector_store.py # One-time embedding generation
├── vector_store/
│ ├── embeddings.npy
│ └── id_to_text.pkl
├── .env # Your secret keys
└── requirements.txt

bash
Copy
Edit

---

## 🔧 Installation

```bash
# 1. Clone the repo
git clone https://github.com/your-username/AI_Application.git
cd AI_Application

# 2. Create & activate virtual environment
python3.10 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Set your API keys in .env file
echo "OPENAI_API_KEY=sk-..." >> .env
echo "GROK_API_KEY=your_xai_api_key" >> .env
🧠 Preprocess Vector Store
Before using the chat API, build the embedding index:

bash
Copy
Edit
python build_vector_store.py
🚀 Start the API Server
bash
Copy
Edit
uvicorn app:app --reload
The API will be available at:
📍 http://localhost:8000
Interactive docs:
📍 http://localhost:8000/docs

💬 Example Requests (using curl)
✅ OpenAI Chat
bash
Copy
Edit
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "what does chahat love?"}'
🤖 Grok (xAI) Chat
bash
Copy
Edit
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "what does chahat love?", "use_grok": true}'
📦 Requirements
Python 3.10+

requirements.txt
makefile
Copy
Edit
fastapi
uvicorn
httpx
openai
python-dotenv
numpy==1.23.5
sentence-transformers==2.2.2
torch==2.1.2
nltk>=3.8
scipy>=1.10
scikit-learn>=1.2.2
huggingface_hub==0.14.1
👨‍💻 Author
Made with ❤️ by Chahat Gupta
💼 LinkedIn