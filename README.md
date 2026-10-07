# StudyMate AI

**Learn Better From What You Have**

StudyMate AI is an adaptive AI-powered study assistant that transforms a student's own PDF notes into an interactive study workspace.

Instead of relying on generic AI responses, StudyMate retrieves relevant information from uploaded study material and uses it to generate grounded explanations, summaries, exam preparation material, quizzes, performance insights, and personalized revision plans.

## Features

### Ask Your Notes
Ask questions directly from uploaded PDF material. StudyMate retrieves relevant sections of the document and generates answers grounded in the notes.

### Smart Summary
Automatically converts lengthy study material into structured revision notes containing:
- Main topics
- Important definitions
- Core concepts
- Important points
- Last-minute revision points

### Exam Preparation
Generates:
- Short-answer questions for approximately 2–5 marks
- Long-answer questions for approximately 8–13 marks
- Concept and application-based questions

StudyMate can also generate structured exam answers based on the selected mark range.

### Adaptive Quiz
Generates multiple-choice questions directly from the uploaded notes and evaluates the student's understanding.

### Knowledge Gap Detection
Analyzes quiz performance topic-by-topic to classify concepts as:
- Strong
- Developing
- Weak

### Weak-Area Practice
Automatically creates a new targeted quiz focusing on topics where the student performed poorly.

### Performance Tracking
Tracks:
- Average topic performance
- Latest scores
- Number of attempts
- Mastered topics
- Weak topics

### Last-Minute Revision
Creates personalized revision plans for:
- 15 minutes
- 30 minutes
- 1 hour

When quiz performance is available, weaker topics are prioritized automatically.

### Evidence Mode
Displays the actual sections and page numbers retrieved from the PDF before an answer is generated, making the AI response more transparent.

## How StudyMate Works

```text
PDF Notes
    |
    v
Text Extraction
    |
    v
Text Chunking
    |
    v
Sentence Embeddings
    |
    v
Similarity Search
    |
    v
Relevant Note Sections
    |
    v
Gemini AI
    |
    v
Grounded Study Response
```

StudyMate combines semantic retrieval with generative AI so responses are based on the student's own material.

## Tech Stack

- Python
- Streamlit
- Google Gemini API
- Sentence Transformers
- all-MiniLM-L6-v2
- PyMuPDF
- NumPy
- Retrieval-Augmented Generation (RAG)

## Project Structure

```text
StudyMate-AI/
|
|-- app.py
|-- requirements.txt
|-- README.md
|-- .gitignore
|
`-- .streamlit/
    `-- secrets.toml   # Not committed to GitHub
```

## Running the Project Locally

Clone the repository:

```bash
git clone <your-repository-url>
cd StudyMate-AI
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Create:

```text
.streamlit/secrets.toml
```

Add your Gemini API key:

```toml
GEMINI_API_KEY = "your-api-key"
```

Run StudyMate:

```bash
python -m streamlit run app.py
```

## Privacy and Security

API keys and local environment files are excluded from version control through `.gitignore`.

Uploaded study material is processed by the application for its study features and is not included in the GitHub repository.

## Future Improvements

Potential future improvements include:
- Persistent user performance storage
- Multi-document study sessions
- Improved semantic reranking
- More advanced topic analytics
- Exportable revision material

## Author

**Aarushi Konda**

B.Tech Computer Science Engineering  
Specialization in Artificial Intelligence & Machine Learning

---

StudyMate AI — **Learn Better From What You Have**