# 📚 PaperPilot — Research Paper Intelligence Workspace

> Understand research papers faster with AI-powered question answering, summaries, and evidence-based analysis.

PaperPilot is an AI-powered research paper analyzer that helps users understand academic papers without manually reading and searching through every page.

Upload a research paper in PDF format, ask questions in natural language, generate a structured summary, and inspect the supporting evidence retrieved from the paper.

---

## ✨ Why PaperPilot?

Research papers can be difficult to understand because they contain:

- Complex technical language
- Long methodology sections
- Difficult results and findings
- Important information spread across multiple pages
- Limitations and future work that are easy to miss

PaperPilot simplifies this process by allowing users to interact with a research paper like a knowledgeable assistant.

Instead of searching through the entire document manually, users can ask:

> What problem does this paper solve?

> What methodology was used?

> What dataset was used?

> What are the main results?

> What are the limitations?

> What future work is suggested?

---

## 🚀 Features

### 📄 PDF Upload

Upload a research paper in PDF format directly through the application.

### 🔎 Intelligent Retrieval

The system extracts text from the PDF, divides it into meaningful chunks, creates embeddings, and indexes them using FAISS.

### 💬 Ask Questions

Ask questions about the uploaded paper using normal language.

The system retrieves relevant sections and generates a direct answer based on the paper.

### 📝 Automatic Paper Summary

Generate a structured summary covering important research sections such as:

- Research problem
- Objectives
- Methodology
- Dataset
- Main results
- Limitations
- Future work

### 📚 Supporting Evidence

View the retrieved sections that support the generated answer.

This improves transparency and helps users verify the response against the original paper.

### 📥 Downloadable Report

Download the generated research analysis as a Markdown report for future reference.

### 🎨 Professional User Interface

PaperPilot uses a clean, academic-style interface with a white and burgundy theme designed for readability and ease of use.

---

## 🧠 How It Works

```text
Upload Research Paper
        ↓
Extract Text from PDF
        ↓
Split Text into Chunks
        ↓
Generate Text Embeddings
        ↓
Create FAISS Vector Index
        ↓
Retrieve Relevant Sections
        ↓
Analyze Retrieved Evidence
        ↓
Generate Answer or Summary
        ↓
Display Results and Supporting Evidence
```

---

## 🏗️ System Architecture

PaperPilot follows a retrieval-augmented generation architecture.

```text
                ┌─────────────────────┐
                │   Research Paper    │
                │        PDF          │
                └──────────┬──────────┘
                           ↓
                ┌─────────────────────┐
                │   PDF Text          │
                │   Extraction        │
                └──────────┬──────────┘
                           ↓
                ┌─────────────────────┐
                │   Text Chunking     │
                └──────────┬──────────┘
                           ↓
                ┌─────────────────────┐
                │   Sentence         │
                │   Embeddings        │
                └──────────┬──────────┘
                           ↓
                ┌─────────────────────┐
                │   FAISS Vector      │
                │   Index             │
                └──────────┬──────────┘
                           ↓
                ┌─────────────────────┐
                │   Relevant Evidence │
                │   Retrieval         │
                └──────────┬──────────┘
                           ↓
                ┌─────────────────────┐
                │   Groq LLM Analysis │
                └──────────┬──────────┘
                           ↓
                ┌─────────────────────┐
                │   Answer / Summary  │
                └─────────────────────┘
```

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Core programming language |
| Streamlit | Web application interface |
| PyPDF | Extracting text from PDF files |
| Sentence Transformers | Creating text embeddings |
| FAISS | Vector similarity search |
| PyTorch | Supporting embedding model execution |
| Groq API | Generating answers and summaries |
| GitHub | Version control and project hosting |
| Streamlit Community Cloud | Application deployment |

---

## 📁 Project Structure

```text
multi-agent-rag/
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/snigdhaxk/multi-agent-rag.git
```

### 2. Open the project folder

```bash
cd multi-agent-rag
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

#### Windows PowerShell

```powershell
venv\Scripts\activate
```

#### macOS/Linux

```bash
source venv/bin/activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Configure the Groq API Key

PaperPilot uses the Groq API to generate answers and summaries.

Create an API key from the Groq console and configure it as an environment variable.

### Windows PowerShell

```powershell
$env:GROQ_API_KEY="your_groq_api_key"
```

### macOS/Linux

```bash
export GROQ_API_KEY="your_groq_api_key"
```

Do not upload your API key to GitHub.

---

## ▶️ Run the Application

Start the Streamlit application using:

```bash
python -m streamlit run app.py --server.fileWatcherType none
```

The application will open in your browser.

---

## ☁️ Deployment

PaperPilot can be deployed using Streamlit Community Cloud.

Deployment settings:

```text
Repository: snigdhaxk/multi-agent-rag
Branch: main
Main file path: app.py
```

Add the following secret in the Streamlit Cloud application settings:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

The API key should be added through **Secrets**, not directly inside the source code.

---

## 🔐 Privacy and Security

- API keys are stored using environment variables or deployment secrets.
- API keys should never be committed to GitHub.
- Uploaded papers are processed by the application for analysis.
- Users should avoid uploading confidential or sensitive documents.

---

## 🎯 Example Questions

After uploading a research paper, users can ask:

```text
What problem does this paper solve?
```

```text
What methodology was used?
```

```text
What dataset was used?
```

```text
What are the main findings?
```

```text
What are the limitations of this study?
```

```text
What future work is suggested?
```

```text
How is this research different from existing approaches?
```

---

## 🔮 Future Improvements

Possible future improvements include:

- Support for multiple PDF documents
- Comparison between research papers
- Citation and reference analysis
- Improved table and figure extraction
- Chat history and saved sessions
- User authentication
- More advanced multi-agent orchestration
- Exporting reports as PDF or Word documents
- Support for additional large language models

---

## 👩‍💻 Author

Developed by **Snigdha Kundana**.

This project was created to explore AI-powered document analysis, retrieval-augmented generation, vector search, and research assistance.

---

## ⭐ If You Find This Project Useful

Consider giving the repository a star ⭐

Your feedback and suggestions are welcome!
