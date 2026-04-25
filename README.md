# Q&A Chatbot Project

A collection of Streamlit-based AI chatbot demos built with LangChain and hosted locally or on Streamlit Cloud.

## What’s Included

- `app.py` - Q&A chatbot powered by OpenAI.
- `main.py` - Q&A chatbot powered by OpenRouter.
- `RAG Document Q&A/` - PDF-based retrieval Q&A app.
- `RAG Q&A Conversation/` - conversational RAG app with PDF uploads and chat history.
- `Text_summarization/` - text summarization demo.
- `Search Engine Tools/` - search-oriented tool demo.
- `Chat SQL/` - SQL-oriented chatbot demo.

## Features

- Simple Streamlit UI
- LLM-based question answering
- Sidebar controls for model settings in supported apps
- PDF upload and retrieval-based answering in RAG apps
- Chat history support in conversational RAG
- Easy deployment to Streamlit Cloud

## Requirements

Use the virtual environment already present in the workspace:

```powershell
.\cleanenv\Scripts\activate
```

Then install dependencies for the app you want to run.

## Run the Main Apps

### OpenAI chatbot

```powershell
streamlit run app.py
```

### OpenRouter chatbot

```powershell
streamlit run main.py
```

### RAG Document Q&A

```powershell
cd "RAG Document Q&A"
streamlit run app.py
```

### RAG Q&A Conversation

```powershell
cd "RAG Q&A Conversation"
streamlit run app.py
```

## Environment Variables

Create a `.env` file if needed and add the correct keys for the app you are using.

### For `app.py`

```env
LANGCHAIN_API_KEY=your_langsmith_key
OPENAI_API_KEY=your_openai_key
```

### For `main.py`

```env
LANGCHAIN_API_KEY=your_langsmith_key
OPENROUTER_API_KEY=your_openrouter_key
```

### For RAG apps

```env
GROQ_API_KEY=your_groq_key
HF_TOKEN=your_huggingface_token
LANGCHAIN_API_KEY=your_langsmith_key
```

## Streamlit Cloud Deployment

1. Push the project to GitHub.
2. Open Streamlit Community Cloud.
3. Select the repo and choose the correct entry file.
4. Add the required secrets in the app settings.
5. Deploy Live Link - https://conversational-rag-system-with-pdf-upload-bge5evasyyqykgmbdkgp.streamlit.app/

## Notes

- If a PDF is scanned or image-based, text extraction may fail unless OCR is added.
- For best results, use PDFs with selectable text.
- If you change PDFs in a RAG app, re-create the vector database/index.

## Recommended Next Steps

- Add OCR fallback for scanned PDFs.
- Add a clear chat/reset button to RAG apps.
- Add proper dependency pinning per subproject for smoother deployment.
