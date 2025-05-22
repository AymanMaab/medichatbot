# 🧠 MediChatBot – Mistral-Powered AI Medical Assistant

MediChatBot is an intelligent, interactive medical chatbot powered by the **Mistral-7B-Instruct** model through **TogetherAI**. It provides accurate, context-aware responses to user questions by combining large language model (LLM) reasoning with retrieval-augmented generation (RAG) from custom medical data.

---

## 🚀 Features

- 🩺 **Mistral 7B LLM** via TogetherAI
- 📚 **Document-aware QA** using vector search (FAISS)
- 💬 **Streamlit Chat UI** with conversation history
- 🔎 **Contextual search** using `sentence-transformers`
- 🧠 **RAG** with custom prompt formatting
- 🧾 **Expandable Source Citations** for transparency
- 🔄 **Clear Cache Button** for development/debugging

---

## 🛠️ Tech Stack

| Tool                | Purpose                                      |
|---------------------|----------------------------------------------|
| [Mistral-7B-Instruct](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) | Language model for answering questions       |
| TogetherAI          | Inference provider for LLMs                 |
| Streamlit           | Web UI for chatbot interface                |
| FAISS               | Vector similarity search                    |
| Sentence Transformers | Embedding model (`all-MiniLM-L6-v2`)        |
| LangChain           | RAG pipeline and prompt formatting          |

---

## 📁 Project Structure

```

medichatbot/
│
├── medibot.py                 # Streamlit app with Mistral QA chain
├── create\_memory\_for\_llm.py   # Embeds PDF documents into FAISS
├── connect\_memory\_with\_llm.py # LLM and token setup
├── vectorstore/db\_faiss/      # Stored vector DB (auto-generated)
├── data/                      # Your medical PDFs go here
├── Pipfile                    # Pipenv dependency file
└── README.md                  # Project documentation

````

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/AymanMaab/medichatbot.git
cd medichatbot
````

### 2. Install dependencies

Make sure you have `pipenv` installed:

```bash
pipenv install
```

### 3. Activate the virtual environment

```bash
pipenv shell
```

### 4. Set your Hugging Face or TogetherAI token

Create a `.env` file or edit `connect_memory_with_llm.py`:

```env
HF_TOKEN=your_huggingface_or_togetherAI_token
```

### 5. Add your documents

Place your medical PDF files into the `/data` directory and run:

```bash
python create_memory_for_llm.py
```

This generates a vectorstore for RAG-based retrieval.

### 6. Launch the app

```bash
streamlit run medibot.py
```

---

## 🖥️ UI Overview

* A simple chat interface powered by `st.chat_input()` and `st.chat_message()`.
* Conversation history is saved in session state.
* Answers come directly from the Mistral model, with relevant document context.
* Expandable “View Sources” section shows where each answer came from.

---

## 🧑‍⚕️ Use Cases

* Symptom checking (non-diagnostic)
* Medical FAQs and patient education
* Information retrieval from internal medical documents
* Front-end to a healthcare knowledge system

---

## ⚠️ Disclaimer

> This project is for **educational purposes only**. It is not a substitute for professional medical advice, diagnosis, or treatment.

---

## ⭐ Contribute & Support

Pull requests and issues are welcome!

If you find this project useful, give it a ⭐ and share it!

---

