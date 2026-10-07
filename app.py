# ============================================================
# STUDYMATE AI
# Learn Better From What You Have
# ============================================================

import streamlit as st
import pymupdf
import hashlib
import json
import re
import time
import os
from datetime import datetime

import numpy as np
from sentence_transformers import SentenceTransformer
from google import genai


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="StudyMate AI",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# GLOBAL THEME
# ============================================================

st.markdown(
    """
    <style>

    :root {
        --navy: #173842;
        --dark-teal: #315D63;
        --teal: #86AEB0;
        --teal-light: #AFCBCD;
        --pale-blue: #D8E7E5;
        --cream: #F1EEE4;
        --cream-two: #E8E5DA;
        --white-cream: #F7F4EC;
        --muted: #557077;
    }

    html,
    body,
    [data-testid="stAppViewContainer"],
    .stApp {
        background: #86AEB0 !important;
        color: #173842 !important;
    }

    .block-container {
        max-width: 1180px !important;
        padding-top: 2.3rem !important;
        padding-left: 3rem !important;
        padding-right: 3rem !important;
        padding-bottom: 5rem !important;
    }

    /* --------------------------------------------------------
       STREAMLIT HEADER
       -------------------------------------------------------- */

    [data-testid="stHeader"] {
        background: transparent !important;
    }

    [data-testid="stToolbar"] {
        background: transparent !important;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    /* --------------------------------------------------------
       SIDEBAR
       -------------------------------------------------------- */

    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div {
        background: #F1EEE4 !important;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(23, 56, 66, 0.12) !important;
    }

    [data-testid="stSidebar"] {
    color: #173842 !important;
}

    [data-testid="stSidebar"] hr {
        border-color: rgba(23, 56, 66, 0.12) !important;
    }

    /* --------------------------------------------------------
       TYPOGRAPHY
       -------------------------------------------------------- */

    h1 {
        color: #173842 !important;
        font-size: 2.55rem !important;
        line-height: 1.1 !important;
        font-weight: 700 !important;
        letter-spacing: -1.2px !important;
    }

    h2 {
        color: #173842 !important;
        font-size: 1.65rem !important;
        font-weight: 650 !important;
        letter-spacing: -0.5px !important;
    }

    h3 {
        color: #173842 !important;
        font-size: 1.15rem !important;
        font-weight: 650 !important;
    }

    p,
    li,
    label {
        color: #234B53 !important;
        line-height: 1.6 !important;
    }

    [data-testid="stCaptionContainer"] {
        color: #315D63 !important;
    }

    /* --------------------------------------------------------
       BUTTONS
       -------------------------------------------------------- */

    div.stButton > button,
    div.stFormSubmitButton > button {
        background: #173842 !important;
        color: #F7F4EC !important;
        border: none !important;
        border-radius: 8px !important;
        min-height: 43px !important;
        padding: 0.55rem 1.25rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 12px rgba(23, 56, 66, 0.10) !important;
        transition:
            background 0.15s ease,
            transform 0.15s ease,
            box-shadow 0.15s ease !important;
    }

    div.stButton > button *,
    div.stFormSubmitButton > button * {
        color: #F7F4EC !important;
    }

    div.stButton > button:hover,
    div.stFormSubmitButton > button:hover {
        background: #315D63 !important;
        color: white !important;
        border: none !important;
        transform: translateY(-1px);
        box-shadow: 0 7px 17px rgba(23, 56, 66, 0.15) !important;
    }

    /* --------------------------------------------------------
       TEXT INPUTS
       -------------------------------------------------------- */

    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea {
        background: #F7F4EC !important;
        color: #173842 !important;
        border: 1px solid rgba(23, 56, 66, 0.22) !important;
        border-radius: 8px !important;
        caret-color: #173842 !important;
    }

    [data-testid="stTextInput"] input::placeholder,
    [data-testid="stTextArea"] textarea::placeholder {
        color: #718388 !important;
    }

    [data-testid="stTextInput"] label,
    [data-testid="stTextArea"] label {
        color: #173842 !important;
    }

    /* --------------------------------------------------------
       SELECT BOX
       -------------------------------------------------------- */

    [data-baseweb="select"] > div {
        background: #F7F4EC !important;
        color: #173842 !important;
        border-color: rgba(23, 56, 66, 0.20) !important;
        border-radius: 8px !important;
    }

    [data-baseweb="select"] span {
        color: #173842 !important;
    }

    /* --------------------------------------------------------
       METRIC CARDS
       -------------------------------------------------------- */

    [data-testid="stMetric"] {
        background: #F1EEE4 !important;
        border: 1px solid rgba(23, 56, 66, 0.10) !important;
        border-radius: 10px !important;
        padding: 1.15rem 1.3rem !important;
        box-shadow: 0 5px 15px rgba(23, 56, 66, 0.06) !important;
    }

    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] * {
        color: #557077 !important;
    }

    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] * {
        color: #173842 !important;
        font-weight: 700 !important;
    }

    /* --------------------------------------------------------
       FILE UPLOADER
       -------------------------------------------------------- */

    [data-testid="stFileUploaderDropzone"] {
        background: #F1EEE4 !important;
        border: 1px dashed rgba(23, 56, 66, 0.30) !important;
        border-radius: 10px !important;
    }

    [data-testid="stFileUploaderDropzone"] * {
        color: #173842 !important;
    }

    [data-testid="stFileUploaderDropzone"] button {
        background: #173842 !important;
        color: #F7F4EC !important;
    }
    
[data-testid="stFileUploaderFile"] {
    background: #20212A !important;
}

[data-testid="stFileUploaderFile"] * {
    color: #F7F4EC !important;
}
/* --------------------------------------------------------
   UPLOADED PDF CARD
   -------------------------------------------------------- */

[data-testid="stFileUploaderFile"] {
    background: #E8E5DA !important;
    border: 1px solid rgba(23, 56, 66, 0.15) !important;
    border-radius: 10px !important;
}

[data-testid="stFileUploaderFile"] div,
[data-testid="stFileUploaderFile"] span,
[data-testid="stFileUploaderFile"] p {
    color: #173842 !important;
}

[data-testid="stFileUploaderFile"] small {
    color: #557077 !important;
}

[data-testid="stFileUploaderFile"] svg {
    color: #173842 !important;
    fill: #173842 !important;
}

/* --------------------------------------------------------
   FIX UPLOAD BUTTON
   -------------------------------------------------------- */

[data-testid="stFileUploaderDropzone"] button {
    background: #173842 !important;
    color: #F7F4EC !important;
}

[data-testid="stFileUploaderDropzone"] button p,
[data-testid="stFileUploaderDropzone"] button span,
[data-testid="stFileUploaderDropzone"] button div {
    color: #F7F4EC !important;
}

[data-testid="stFileUploaderDropzone"] button svg {
    color: #F7F4EC !important;
    fill: #F7F4EC !important;
}

    /* --------------------------------------------------------
       RADIO / QUIZ OPTIONS
       -------------------------------------------------------- */

    [data-testid="stRadio"] label,
    [data-testid="stRadio"] label p {
        color: #173842 !important;
    }

    [data-testid="stRadio"] div[role="radiogroup"] label {
        background: #F1EEE4 !important;
        border: 1px solid rgba(23, 56, 66, 0.10) !important;
        border-radius: 8px !important;
        padding: 0.62rem 0.75rem !important;
        margin-bottom: 0.28rem !important;
    }

    /* Sidebar navigation should stay simple */

    [data-testid="stSidebar"]
    [data-testid="stRadio"]
    div[role="radiogroup"] label {
        background: transparent !important;
        border: none !important;
        padding: 0.42rem 0.5rem !important;
        margin: 0 !important;
    }

    [data-testid="stSidebar"]
    [data-testid="stRadio"]
    div[role="radiogroup"] label:hover {
        background: rgba(134, 174, 176, 0.18) !important;
    }

    /* --------------------------------------------------------
       ALERTS
       -------------------------------------------------------- */

    [data-testid="stAlert"] {
        background: #F1EEE4 !important;
        color: #173842 !important;
        border: 1px solid rgba(23, 56, 66, 0.10) !important;
        border-radius: 9px !important;
    }

    [data-testid="stAlert"] * {
        color: #173842 !important;
    }

    div[data-testid="stAlert"][kind="info"] {
        background: #DCE9E7 !important;
    }

    div[data-testid="stAlert"][kind="success"] {
        background: #DDE8D9 !important;
    }

    div[data-testid="stAlert"][kind="warning"] {
        background: #EEE6D0 !important;
    }

    div[data-testid="stAlert"][kind="error"] {
        background: #ECDAD5 !important;
    }

    /* --------------------------------------------------------
       EXPANDERS
       -------------------------------------------------------- */

    [data-testid="stExpander"] {
        background: #F1EEE4 !important;
        border: 1px solid rgba(23, 56, 66, 0.11) !important;
        border-radius: 9px !important;
        overflow: hidden !important;
    }

    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary *,
    [data-testid="stExpander"] p {
        color: #173842 !important;
    }

    /* --------------------------------------------------------
       FORMS
       -------------------------------------------------------- */

    [data-testid="stForm"] {
        background: rgba(241, 238, 228, 0.55) !important;
        border: 1px solid rgba(23, 56, 66, 0.10) !important;
        border-radius: 12px !important;
        padding: 1.2rem !important;
    }

    /* --------------------------------------------------------
       PROGRESS BAR
       -------------------------------------------------------- */

    [data-testid="stProgress"] > div {
        background: rgba(241, 238, 228, 0.65) !important;
        border-radius: 20px !important;
    }

    [data-testid="stProgress"] > div > div > div {
        background: #315D63 !important;
        border-radius: 20px !important;
    }

    hr {
        border: none !important;
        border-top: 1px solid rgba(23, 56, 66, 0.15) !important;
        margin: 1.8rem 0 !important;
    }

    code {
        background: #E8E5DA !important;
        color: #173842 !important;
        border-radius: 4px !important;
    }

    /* --------------------------------------------------------
       MOBILE
       -------------------------------------------------------- */

    @media (max-width: 800px) {

        .block-container {
            padding-left: 1.1rem !important;
            padding-right: 1.1rem !important;
            padding-top: 1.5rem !important;
        }

        h1 {
            font-size: 2rem !important;
        }

        h2 {
            font-size: 1.4rem !important;
        }
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULTS = {
    "user_name": "",
    "current_file_id": None,

    "quiz": None,
    "quiz_submitted": False,
    "quiz_result": None,

    "adaptive_quiz": None,
    "adaptive_submitted": False,
    "adaptive_result": None,

    "performance_history": {},

    "last_question": "",
    "last_answer": "",
    "last_sources": [],

    "summary": "",
    "important_questions": "",
    "generated_answers": "",
    "revision": "",
}


for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# GEMINI
# ============================================================

def get_gemini_client():

    try:

        api_key = st.secrets["GEMINI_API_KEY"]

        return genai.Client(
            api_key=api_key
        )

    except Exception:

        return None


def ask_gemini(prompt):

    client = get_gemini_client()

    if client is None:
        return (
            "Gemini API key was not found. "
            "Add GEMINI_API_KEY to "
            ".streamlit/secrets.toml."
        )

    models = [
        "gemini-3.7-flash",
        "gemini-3.5-flash-lite",
    ]

    last_error = None

    for model_name in models:

        try:

            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )

            if response and response.text:
                return response.text.strip()

        except Exception as error:
            last_error = error
            continue

    return (
        "The AI service is temporarily unavailable. "
        "Please try again in a moment.\n\n"
        f"Technical details: {last_error}"
    )

# ============================================================
# EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


# ============================================================
# PDF FUNCTIONS
# ============================================================

@st.cache_data(show_spinner=False)
def extract_pdf(pdf_bytes):

    document = pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    pages = []

    for page_number, page in enumerate(document):

        text = page.get_text("text")

        pages.append(
            {
                "page": page_number + 1,
                "text": text.strip()
            }
        )

    document.close()

    full_text = "\n\n".join(
        page["text"]
        for page in pages
        if page["text"]
    )

    return pages, full_text


@st.cache_data(show_spinner=False)
def make_chunks(
    pages,
    chunk_size=150,
    overlap=30
):

    chunks = []

    step = chunk_size - overlap

    for page in pages:

        words = page["text"].split()

        if not words:
            continue

        for start in range(
            0,
            len(words),
            step
        ):

            piece = words[
                start:start + chunk_size
            ]

            if not piece:
                continue

            chunks.append(
                {
                    "page": page["page"],
                    "text": " ".join(piece)
                }
            )

            if start + chunk_size >= len(words):
                break

    return chunks


@st.cache_data(show_spinner=False)
def create_embeddings(chunk_texts):

    model = load_embedding_model()

    embeddings = model.encode(
        chunk_texts,
        normalize_embeddings=True
    )

    return np.asarray(embeddings)


def retrieve_chunks(
    question,
    chunks,
    embeddings,
    top_k=4
):

    if not chunks:
        return []

    model = load_embedding_model()

    question_embedding = model.encode(
        [question],
        normalize_embeddings=True
    )[0]

    scores = (
        embeddings @ question_embedding
    )

    top_indices = np.argsort(
        scores
    )[::-1][:top_k]

    results = []

    for index in top_indices:

        results.append(
            {
                "page": chunks[index]["page"],
                "text": chunks[index]["text"],
                "score": float(scores[index])
            }
        )

    return results


# ============================================================
# HELPERS
# ============================================================

def clean_json_response(text):

    text = text.strip()

    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    return text.strip()


def get_file_id(file_bytes):

    return hashlib.md5(
        file_bytes
    ).hexdigest()


def reset_document_state():

    keys = [
        "quiz",
        "quiz_submitted",
        "quiz_result",
        "adaptive_quiz",
        "adaptive_submitted",
        "adaptive_result",
        "last_question",
        "last_answer",
        "last_sources",
        "summary",
        "important_questions",
        "generated_answers",
        "revision",
    ]

    for key in keys:

        if key in DEFAULTS:

            st.session_state[key] = (
                DEFAULTS[key]
            )


def normalize_topic(topic):

    topic = str(
        topic
    ).strip().lower()

    rules = {

        "linked list": "Linked Lists",
        "linked lists": "Linked Lists",
        "singly linked": "Linked Lists",
        "doubly linked": "Linked Lists",

        "stack": "Stacks",
        "stacks": "Stacks",

        "queue": "Queues",
        "queues": "Queues",

        "array": "Arrays",
        "arrays": "Arrays",

        "pointer": "Pointers",
        "pointers": "Pointers",

        "tree": "Trees",
        "trees": "Trees",
        "binary tree": "Trees",

        "graph": "Graphs",
        "graphs": "Graphs",

        "sorting": "Sorting",
        "sort": "Sorting",

        "searching": "Searching",
        "search": "Searching",

        "recursion": "Recursion",

        "string": "Strings",
        "strings": "Strings",

        "structure": "Structures",
        "structures": "Structures",

        "matrix": "Matrices",
        "matrices": "Matrices",
    }

    for keyword, normalized in rules.items():

        if keyword in topic:
            return normalized

    if not topic:
        return "General"

    return topic.title()


def parse_quiz(text):

    try:

        cleaned = clean_json_response(
            text
        )

        data = json.loads(
            cleaned
        )

        if isinstance(data, dict):

            if "questions" in data:

                data = data["questions"]

            elif "quiz" in data:

                data = data["quiz"]

        if not isinstance(data, list):

            return None

        if len(data) != 5:

            return None

        validated = []

        for item in data:

            if not isinstance(
                item,
                dict
            ):
                return None

            required = [
                "topic",
                "question",
                "options",
                "answer",
                "explanation"
            ]

            if not all(
                key in item
                for key in required
            ):
                return None

            options = item["options"]

            if (
                not isinstance(options, list)
                or len(options) != 4
            ):
                return None

            options = [
                str(option).strip()
                for option in options
            ]

            answer = str(
                item["answer"]
            ).strip()

            if answer not in options:
                return None

            validated.append(
                {
                    "topic": normalize_topic(
                        item["topic"]
                    ),
                    "question": str(
                        item["question"]
                    ).strip(),
                    "options": options,
                    "answer": answer,
                    "explanation": str(
                        item["explanation"]
                    ).strip(),
                }
            )

        return validated

    except Exception:

        return None


def record_performance(
    file_id,
    topic_results
):

    if (
        file_id
        not in st.session_state.performance_history
    ):

        st.session_state.performance_history[
            file_id
        ] = {}

    file_history = (
        st.session_state.performance_history[
            file_id
        ]
    )

    for topic, data in topic_results.items():

        if topic not in file_history:

            file_history[topic] = []

        file_history[topic].append(
            data["percentage"]
        )


def calculate_topic_results(
    quiz,
    answers
):

    topic_stats = {}

    for index, question in enumerate(
        quiz
    ):

        topic = normalize_topic(
            question["topic"]
        )

        if topic not in topic_stats:

            topic_stats[topic] = {
                "correct": 0,
                "total": 0
            }

        topic_stats[topic][
            "total"
        ] += 1

        if (
            answers[index]
            == question["answer"]
        ):

            topic_stats[topic][
                "correct"
            ] += 1

    for topic in topic_stats:

        correct = (
            topic_stats[topic]["correct"]
        )

        total = (
            topic_stats[topic]["total"]
        )

        topic_stats[topic][
            "percentage"
        ] = round(
            (correct / total) * 100
        )

    return topic_stats


def get_performance_summary(
    file_id
):

    history = (
        st.session_state.performance_history.get(
            file_id,
            {}
        )
    )

    result = {}

    for topic, scores in history.items():

        if scores:

            average = round(
                sum(scores) / len(scores)
            )

            result[topic] = {
                "average": average,
                "attempts": len(scores),
                "latest": scores[-1]
            }

    return result


def render_quiz_review(
    quiz,
    answers
):

    for index, question in enumerate(
        quiz
    ):

        selected = answers[index]

        correct = question["answer"]

        with st.expander(
            f"Question {index + 1}: "
            f"{question['topic']}"
        ):

            st.write(
                question["question"]
            )

            st.write(
                f"**Your answer:** "
                f"{selected}"
            )

            st.write(
                f"**Correct answer:** "
                f"{correct}"
            )

            if selected == correct:

                st.success("Correct")

            else:

                st.warning(
                    "Needs review"
                )

            st.write(
                "**Explanation:** "
                + question["explanation"]
            )


# ============================================================
# ONBOARDING / COVER PAGE
# ============================================================

if not st.session_state.user_name:

    # --------------------------------------------------------
    # COVER PAGE CSS
    # CSS only — no visible HTML elements
    # --------------------------------------------------------

    st.markdown(
        """
        <style>

        /* Hide sidebar on onboarding page */
        [data-testid="stSidebar"] {
            display: none !important;
        }

        /* Main cover spacing */
        .block-container {
            max-width: 1250px !important;
            padding-top: 2.5rem !important;
            padding-bottom: 3rem !important;
        }

        /* Remove excessive spacing */
        [data-testid="stVerticalBlock"] {
            gap: 0.8rem;
        }

        /* Main cover title */
        h1 {
            color: #173842 !important;
            font-size: 3.3rem !important;
            line-height: 1.05 !important;
            letter-spacing: -2px !important;
            font-weight: 750 !important;
            max-width: 620px;
        }

        /* Cover subheadings */
        h2 {
            color: #173842 !important;
            letter-spacing: -0.5px !important;
        }

        /* Paragraph text */
        p {
            color: #315D63 !important;
        }

        /* Input */
        [data-testid="stTextInput"] input {
            background: #F7F4EC !important;
            color: #173842 !important;
            border: 1px solid rgba(23, 56, 66, 0.18) !important;
            border-radius: 10px !important;
            min-height: 50px !important;
            padding-left: 1rem !important;
        }

        [data-testid="stTextInput"] input::placeholder {
            color: #718388 !important;
        }

        /* Button */
        div.stButton > button {
            min-height: 50px !important;
            border-radius: 10px !important;
            font-size: 0.95rem !important;
            font-weight: 650 !important;
        }

        /* Image */
        [data-testid="stImage"] img {
            width: 100% !important;
            border-radius: 18px !important;
        }

        /* Small divider */
        hr {
            border: none !important;
            border-top: 1px solid rgba(23, 56, 66, 0.14) !important;
            margin: 1rem 0 !important;
        }

        /* Mobile */
        @media (max-width: 800px) {

            .block-container {
                padding-top: 1.3rem !important;
                padding-left: 1.1rem !important;
                padding-right: 1.1rem !important;
            }

            h1 {
                font-size: 2.35rem !important;
                line-height: 1.08 !important;
                letter-spacing: -1.3px !important;
            }

            h2 {
                font-size: 1.35rem !important;
            }

            [data-testid="stImage"] img {
                max-height: 420px;
                object-fit: contain;
            }
        }

        </style>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # MAIN COVER
    # --------------------------------------------------------

    left, right = st.columns(
        [1.08, 0.92],
        gap="large",
        vertical_alignment="center"
    )


    # ========================================================
    # LEFT SIDE
    # ========================================================

    with left:

        # BRAND
        brand1, brand2 = st.columns(
            [0.08, 0.92],
            vertical_alignment="center"
        )

        with brand1:
            st.markdown("### ✦")

        with brand2:
            st.markdown("### STUDYMATE AI")


        st.write("")

        st.caption(
            "YOUR PERSONAL STUDY WORKSPACE"
        )


        st.title(
            "Learn Better From What You Have"
        )


        st.write(
            "Turn your own notes into clear answers, "
            "smart summaries, exam preparation, adaptive "
            "quizzes and focused revision — all in one "
            "study space."
        )


        st.write("")


        # ----------------------------------------------------
        # FEATURE ROW
        # ----------------------------------------------------

        f1, f2 = st.columns(2)

        with f1:

            st.write("**Ask Notes**")
            st.caption(
                "Get answers grounded in your PDF."
            )

            st.write("**Exam Prep**")
            st.caption(
                "Prepare questions and exam-ready answers."
            )


        with f2:

            st.write("**Smart Summary**")
            st.caption(
                "Turn long notes into focused revision."
            )

            st.write("**Adaptive Quiz**")
            st.caption(
                "Find weak areas and practise them."
            )


        st.divider()


        # ----------------------------------------------------
        # NAME ENTRY
        # ----------------------------------------------------

        st.subheader(
            "Welcome to your study space"
        )


        st.write(
            "Start with your name, then upload your notes "
            "and let StudyMate build your study tools "
            "around them."
        )


        name = st.text_input(
            "Your name",
            placeholder="Enter your name",
            label_visibility="collapsed"
        )


        if st.button(
            "Enter StudyMate  →",
            use_container_width=True
        ):

            if name.strip():

                st.session_state.user_name = (
                    name.strip()
                )

                st.rerun()

            else:

                st.warning(
                    "Please enter your name first."
                )


    # ========================================================
    # RIGHT SIDE
    # ========================================================

    with right:

        hero_path = os.path.join(
            "assets",
            "studymate-hero.png"
        )


        # ----------------------------------------------------
        # IF IMAGE EXISTS
        # ----------------------------------------------------

        if os.path.exists(hero_path):

            st.image(
                hero_path,
                use_container_width=True
            )


        # ----------------------------------------------------
        # FALLBACK IF IMAGE ISN'T ADDED YET
        # ----------------------------------------------------

        else:

            st.write("")
            st.write("")

            st.subheader(
                "Your notes, transformed."
            )

            st.write(
                "Upload once. Study in different ways."
            )

            st.write("")

            card1, card2 = st.columns(2)

            with card1:

                st.info(
                    "**SUMMARY**\n\n"
                    "Important concepts condensed "
                    "for revision."
                )


            with card2:

                st.info(
                    "**QUIZ**\n\n"
                    "Test yourself and identify "
                    "knowledge gaps."
                )


            card3, card4 = st.columns(2)

            with card3:

                st.info(
                    "**ASK NOTES**\n\n"
                    "Ask questions using your own "
                    "material."
                )


            with card4:

                st.info(
                    "**REVISION**\n\n"
                    "Focus your remaining study "
                    "time where it matters."
                )


            st.write("")


            st.caption(
                "STUDYMATE AI  •  YOUR MATERIAL  •  YOUR PROGRESS"
            )


    # Do not continue to main app until name is entered
    st.stop()

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    # --------------------------------------------------------
    # USER
    # --------------------------------------------------------

    st.title("StudyMate")

    st.caption("AI study workspace")

    st.write(
        f"**{st.session_state.user_name}**"
    )

    if st.button(
        "Change name",
        use_container_width=True
    ):
        st.session_state.user_name = ""
        st.rerun()


    # --------------------------------------------------------
    # PDF UPLOAD — MOVED TO TOP
    # --------------------------------------------------------

    st.markdown("---")

    st.caption("YOUR NOTES")

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        label_visibility="collapsed"
    )


    # --------------------------------------------------------
    # NAVIGATION
    # --------------------------------------------------------

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Ask Notes",
            "Smart Summary",
            "Exam Prep",
            "Quiz",
            "Progress",
            "Revision"
        ],
        label_visibility="collapsed"
    )

# ============================================================
# NO PDF YET
# ============================================================

if uploaded_file is None:

    hour = datetime.now().hour

    if hour < 12:

        greeting = "Good morning"

    elif hour < 17:

        greeting = "Good afternoon"

    else:

        greeting = "Good evening"


    st.caption(
        "STUDYMATE AI"
    )


    st.title(
        f"{greeting}, "
        f"{st.session_state.user_name}"
    )


    st.write(
        "Turn your own notes into a focused "
        "study workspace."
    )


    st.markdown("---")


    left, right = st.columns(
        [1.25, 0.75],
        gap="large"
    )


    with left:

        st.subheader(
            "Study with your material, not around it"
        )

        st.write(
            "Upload a PDF from the sidebar. "
            "StudyMate will read your notes and create "
            "tools that stay connected to the material "
            "you are actually studying."
        )

        st.info(
            "Start by uploading your class notes, "
            "textbook chapter or lecture PDF "
            "from the sidebar."
        )


    with right:

        st.subheader(
            "Your study space"
        )

        st.write(
            "**Ask Notes**"
        )

        st.caption(
            "Ask questions using your PDF as evidence."
        )

        st.write(
            "**Exam Prep**"
        )

        st.caption(
            "Generate likely questions and structured answers."
        )

        st.write(
            "**Adaptive Quiz**"
        )

        st.caption(
            "Find weak areas and practise them again."
        )


    st.markdown("---")


    st.subheader(
        "Study tools"
    )


    c1, c2, c3 = st.columns(3)


    with c1:

        st.info(
            "**Ask your notes**\n\n"
            "Get grounded explanations "
            "with source pages."
        )


    with c2:

        st.info(
            "**Build revision material**\n\n"
            "Generate summaries, questions "
            "and answers."
        )


    with c3:

        st.info(
            "**Track weak areas**\n\n"
            "Use quiz performance to "
            "guide revision."
        )


    st.stop()


# ============================================================
# PROCESS PDF
# ============================================================

pdf_bytes = (
    uploaded_file.getvalue()
)


file_id = get_file_id(
    pdf_bytes
)


if (
    st.session_state.current_file_id
    != file_id
):

    st.session_state.current_file_id = (
        file_id
    )

    reset_document_state()


with st.spinner(
    "Preparing your study material..."
):

    pages, full_text = extract_pdf(
        pdf_bytes
    )

    chunks = make_chunks(
        pages
    )

    chunk_texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = create_embeddings(
        chunk_texts
    )


if not full_text.strip():

    st.error(
        "I couldn't extract readable text from this PDF. "
        "It may contain scanned images instead of "
        "selectable text."
    )

    st.stop()


# Limit large prompts for now
note_context = full_text[:30000]


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    st.caption("STUDYMATE AI")

    st.title(
        f"Welcome back, {st.session_state.user_name}"
    )

    st.write(
        "Your notes are ready. Let's make this study session count."
    )

    st.markdown("---")


    # --------------------------------------------------------
    # QUICK STATS
    # --------------------------------------------------------

    history = get_performance_summary(
        file_id
    )

    attempts = sum(
        values["attempts"]
        for values in history.values()
    )


    m1, m2, m3 = st.columns(3)


    with m1:
        st.metric(
            "Pages",
            len(pages)
        )


    with m2:
        st.metric(
            "Study chunks",
            len(chunks)
        )


    with m3:
        st.metric(
            "Quiz attempts",
            attempts
        )


    st.markdown("---")


    # --------------------------------------------------------
    # CURRENT MATERIAL
    # --------------------------------------------------------

    st.caption("CURRENT MATERIAL")

    st.subheader(
        uploaded_file.name
    )


    material1, material2, material3 = st.columns(
        [1, 1, 2]
    )


    with material1:
        st.write(
            f"**{len(pages)} pages**"
        )


    with material2:
        st.write(
            f"**{len(chunks)} study chunks**"
        )


    with material3:
        st.success(
            "Ready to study"
        )


    st.write(
        "StudyMate has processed your PDF and prepared it "
        "for grounded question answering, summaries, exam "
        "preparation and adaptive practice."
    )


    st.markdown("---")


    # --------------------------------------------------------
    # STUDY TOOLS
    # --------------------------------------------------------

    st.caption("STUDY TOOLS")

    st.subheader(
        "What would you like to do?"
    )


    tool1, tool2 = st.columns(
        2,
        gap="large"
    )


    with tool1:

        st.info(
            "**ASK NOTES**\n\n"
            "Ask questions and get answers grounded "
            "in your uploaded PDF."
        )

        st.info(
            "**EXAM PREP**\n\n"
            "Generate important questions and "
            "structured exam-ready answers."
        )


    with tool2:

        st.info(
            "**SMART SUMMARY**\n\n"
            "Condense your notes into important "
            "concepts and revision points."
        )

        st.info(
            "**ADAPTIVE QUIZ**\n\n"
            "Test your understanding and discover "
            "topics that need more practice."
        )


    st.markdown("---")


    # --------------------------------------------------------
    # STUDY FLOW
    # --------------------------------------------------------

    st.caption("YOUR STUDY FLOW")

    st.subheader(
        "A simple way to use StudyMate"
    )


    flow1, flow2 = st.columns(
        [0.65, 1.35],
        gap="large"
    )


    with flow1:

        st.write("**01  Understand**")
        st.caption("Ask questions from your notes.")

        st.write("**02  Condense**")
        st.caption("Create a Smart Summary.")

        st.write("**03  Prepare**")
        st.caption("Generate exam-focused material.")


    with flow2:

        st.write("**04  Test**")
        st.caption(
            "Take a quiz and identify knowledge gaps."
        )

        st.write("**05  Improve**")
        st.caption(
            "Review Progress and practise weak areas."
        )

        st.write("**06  Revise**")
        st.caption(
            "Finish with a focused last-minute revision plan."
        )


# ============================================================
# ASK NOTES
# ============================================================

elif page == "Ask Notes":

    # --------------------------------------------------------
    # PAGE HEADER
    # --------------------------------------------------------

    st.caption("ASK YOUR NOTES")

    st.title("What do you want to understand?")

    st.write(
        "Ask anything from your uploaded material. "
        "StudyMate searches your PDF and answers using "
        "the most relevant sections."
    )

    st.markdown("---")


    # --------------------------------------------------------
    # QUESTION AREA
    # --------------------------------------------------------

    st.subheader("Ask a question")

    st.caption(
        f"Currently studying: {uploaded_file.name}"
    )

    question = st.text_area(
        "Your question",
        placeholder=(
            "For example: Explain how a linked list works "
            "and mention its main operations."
        ),
        height=120
    )

    ask_button = st.button(
        "Ask StudyMate",
        use_container_width=True,
        type="primary"
    )


    # --------------------------------------------------------
    # RETRIEVE + GENERATE ANSWER
    # --------------------------------------------------------

    if ask_button:

        if not question.strip():

            st.warning(
                "Enter a question first."
            )

        else:

            with st.spinner(
                "Searching your notes and preparing an answer..."
            ):

                sources = retrieve_chunks(
                    question,
                    chunks,
                    embeddings,
                    top_k=4
                )

                context = "\n\n".join(
                    [
                        (
                            f"[Page {source['page']}]\n"
                            f"{source['text']}"
                        )
                        for source in sources
                    ]
                )

                prompt = f"""
You are StudyMate, an academic study assistant.

Answer the student's question using ONLY the supplied notes.

Rules:
- Stay grounded in the notes.
- Do not invent information.
- If the notes do not contain enough information, clearly say so.
- Explain in student-friendly language.
- Structure longer answers using headings or bullet points.
- Mention relevant page numbers when useful.

STUDENT QUESTION:
{question}

RELEVANT NOTES:
{context}
"""

                answer = ask_gemini(prompt)

                st.session_state.last_question = question
                st.session_state.last_answer = answer
                st.session_state.last_sources = sources


    # --------------------------------------------------------
    # ANSWER AREA
    # --------------------------------------------------------

    if st.session_state.last_answer:

        st.markdown("---")

        st.caption("STUDYMATE ANSWER")

        st.subheader(
            st.session_state.last_question
        )

        st.write(
            st.session_state.last_answer
        )


        # ----------------------------------------------------
        # SOURCE PAGES
        # ----------------------------------------------------

        pages_used = sorted(
            set(
                source["page"]
                for source
                in st.session_state.last_sources
            )
        )

        if pages_used:

            source_text = ", ".join(
                f"Page {page}"
                for page in pages_used
            )

            st.info(
                f"Sources used: {source_text}"
            )


        # ----------------------------------------------------
        # EVIDENCE MODE
        # ----------------------------------------------------

        with st.expander(
            "View evidence from your PDF"
        ):

            st.caption(
                "These are the sections of your PDF "
                "StudyMate retrieved before generating the answer."
            )

            for number, source in enumerate(
                st.session_state.last_sources,
                start=1
            ):

                st.markdown(
                    f"### Evidence {number}"
                )

                st.caption(
                    f"Page {source['page']}  •  "
                    f"Similarity {source['score']:.3f}"
                )

                st.write(
                    source["text"]
                )

                if number != len(
                    st.session_state.last_sources
                ):

                    st.markdown("---")

# ============================================================
# SMART SUMMARY
# ============================================================

elif page == "Smart Summary":

    # PAGE HEADER
    st.caption("SMART SUMMARY")
    st.title("Turn your notes into revision material")
    st.write(
        "Generate a structured summary from your uploaded PDF "
        "with the important topics, definitions, concepts, and "
        "last-minute revision points in one place."
    )
    st.markdown("---")

    # MATERIAL INFO
    st.subheader("Create your summary")
    st.caption(f"Currently studying: {uploaded_file.name}")

    st.info(
        "StudyMate will organize your notes into:\n\n"
        "• Main topics\n\n"
        "• Important definitions\n\n"
        "• Core concepts\n\n"
        "• Important points\n\n"
        "• Last-minute revision points"
    )

    generate_summary = st.button(
        "Generate Smart Summary",
        use_container_width=True,
        type="primary"
    )

    # GENERATE SUMMARY
    if generate_summary:

        prompt = f"""
You are StudyMate, an academic study assistant.

Create a clear study summary using ONLY the notes below.

Organize the summary into:

1. Main Topics
2. Important Definitions
3. Core Concepts
4. Important Points
5. Last-Minute Revision Points

Requirements:
- Preserve the terminology used in the notes.
- Be concise but useful for an engineering student.
- Do not add unsupported information.
- Use clear headings and bullet points.
- Make the summary easy to revise before an exam.

NOTES:
{note_context}
"""

        with st.spinner(
            "Reading your notes and creating your revision summary..."
        ):
            st.session_state.summary = ask_gemini(prompt)

    # DISPLAY SUMMARY
    if st.session_state.summary:

        st.markdown("---")

        st.caption("YOUR REVISION SUMMARY")
        st.subheader("Smart Summary")

        st.markdown(
            st.session_state.summary
        )

# ============================================================
# EXAM PREP
# ============================================================

elif page == "Exam Prep":

    # PAGE HEADER
    st.caption("EXAM PREP")
    st.title("Prepare for the questions that matter")
    st.write(
        "Turn your uploaded notes into exam-focused questions "
        "and structured answers designed for different mark ranges."
    )
    st.markdown("---")

    tab1, tab2 = st.tabs(
        [
            "Important Questions",
            "Generate Answers"
        ]
    )

    # ========================================================
    # IMPORTANT QUESTIONS
    # ========================================================

    with tab1:

        st.subheader("Important Questions")
        st.caption(f"Currently studying: {uploaded_file.name}")

        st.write(
            "StudyMate will scan your material and create a balanced "
            "set of questions covering important topics from your notes."
        )

        st.info(
            "Your question set will include:\n\n"
            "• 8 short-answer questions — approximately 2–5 marks\n\n"
            "• 6 long-answer questions — approximately 8–13 marks\n\n"
            "• 5 concept or application questions"
        )

        generate_questions = st.button(
            "Generate Important Questions",
            use_container_width=True,
            type="primary"
        )

        if generate_questions:

            prompt = f"""
You are StudyMate, an academic study assistant.

Using ONLY the notes below, generate exam-focused questions.

Create exactly:

SHORT ANSWER QUESTIONS
- 8 questions suitable for approximately 2–5 marks.

LONG ANSWER QUESTIONS
- 6 questions suitable for approximately 8–13 marks.

CONCEPT / APPLICATION QUESTIONS
- 5 questions that test understanding or application.

Rules:
- Questions must be answerable from the notes.
- Cover the major topics.
- Avoid duplicates.
- Do not provide answers yet.
- Keep the questions clear and suitable for an engineering exam.

NOTES:
{note_context}
"""

            with st.spinner(
                "Analyzing your notes and finding important questions..."
            ):
                st.session_state.important_questions = ask_gemini(
                    prompt
                )

        if st.session_state.important_questions:

            st.markdown("---")
            st.caption("YOUR QUESTION SET")
            st.subheader("Important Questions")

            st.markdown(
                st.session_state.important_questions
            )

    # ========================================================
    # GENERATE ANSWERS
    # ========================================================

    with tab2:

        st.subheader("Generate an Exam Answer")
        st.caption(f"Currently studying: {uploaded_file.name}")

        st.write(
            "Enter an exam question and choose the answer style. "
            "StudyMate will create an answer using your uploaded notes."
        )

        answer_type = st.selectbox(
            "Answer style",
            [
                "Short Answer — 2 to 5 marks",
                "Long Answer — 8 to 13 marks",
                "Concept / Reasoning Answer"
            ]
        )

        exam_question = st.text_area(
            "Exam question",
            placeholder=(
                "For example: Explain the different operations "
                "performed on a linked list."
            ),
            height=120
        )

        generate_answer = st.button(
            "Generate Answer",
            use_container_width=True,
            type="primary"
        )

        if generate_answer:

            if not exam_question.strip():

                st.warning(
                    "Enter an exam question first."
                )

            else:

                if answer_type.startswith("Short"):

                    instructions = """
Write a concise 2–5 mark answer.
Give the definition/main idea first.
Include the key points required for marks.
"""

                elif answer_type.startswith("Long"):

                    instructions = """
Write a detailed 8–13 mark answer.
Use an introduction, clear headings,
important points and a short conclusion where suitable.
"""

                else:

                    instructions = """
Write a concept-focused answer.
Explain the reasoning clearly and connect the ideas step by step.
"""

                prompt = f"""
You are StudyMate, an academic study assistant.

Answer the exam question using ONLY the supplied notes.

QUESTION:
{exam_question}

ANSWER REQUIREMENTS:
{instructions}

Rules:
- Do not invent facts that are absent from the notes.
- Preserve important technical terminology.
- Make the answer easy to reproduce in an exam.
- Structure the answer clearly according to the selected mark range.

NOTES:
{note_context}
"""

                with st.spinner(
                    "Reading your notes and preparing your exam answer..."
                ):

                    st.session_state.generated_answers = ask_gemini(
                        prompt
                    )

        if st.session_state.generated_answers:

            st.markdown("---")
            st.caption("YOUR EXAM ANSWER")
            st.subheader(exam_question)

            st.markdown(
                st.session_state.generated_answers
            )

# ============================================================
# QUIZ
# ============================================================

elif page == "Quiz":

    st.caption(
        "ACTIVE RECALL"
    )


    st.title(
        "Adaptive Quiz"
    )


    st.write(
        "Test your understanding. StudyMate uses "
        "your results to identify topics that may "
        "need more revision."
    )


    # --------------------------------------------------------
    # GENERATE MAIN QUIZ
    # --------------------------------------------------------

    if st.session_state.quiz is None:

        if st.button(
            "Generate Quiz",
            use_container_width=True
        ):

            prompt = f"""
Create exactly 5 multiple-choice questions using ONLY the notes below.

Return ONLY valid JSON.

The response must be a JSON array containing exactly 5 objects.

Each object must have exactly these fields:

"topic"
"question"
"options"
"answer"
"explanation"

Rules:
- options must contain exactly 4 strings.
- answer must exactly match one of the 4 options.
- Each question should test understanding, not trivial wording.
- Use meaningful topic names.
- Do not use markdown.
- Do not wrap the JSON in code fences.

Example:

[
  {{
    "topic": "Linked Lists",
    "question": "Question here",
    "options": [
      "Option A",
      "Option B",
      "Option C",
      "Option D"
    ],
    "answer": "Option A",
    "explanation": "Explanation here"
  }}
]

NOTES:
{note_context}
"""


            with st.spinner(
                "Building your quiz..."
            ):

                raw_quiz = ask_gemini(
                    prompt
                )

                parsed = parse_quiz(
                    raw_quiz
                )


            if parsed:

                st.session_state.quiz = (
                    parsed
                )

                st.session_state.quiz_submitted = (
                    False
                )

                st.session_state.quiz_result = (
                    None
                )

                st.rerun()

            else:

                st.error(
                    "The AI returned an invalid quiz format. "
                    "Please click Generate Quiz again."
                )


    # --------------------------------------------------------
    # DISPLAY MAIN QUIZ
    # --------------------------------------------------------

    else:

        quiz = st.session_state.quiz


        if not st.session_state.quiz_submitted:

            with st.form(
                "main_quiz_form"
            ):

                answers = []


                for index, question in enumerate(
                    quiz
                ):

                    st.markdown(
                        f"### {index + 1}. "
                        f"{question['question']}"
                    )


                    answer = st.radio(
                        "Choose an answer",
                        question["options"],
                        index=None,
                        key=(
                            f"quiz_"
                            f"{file_id}_"
                            f"{index}"
                        ),
                        label_visibility="collapsed"
                    )


                    answers.append(
                        answer
                    )


                submitted = (
                    st.form_submit_button(
                        "Submit Quiz",
                        use_container_width=True
                    )
                )


            if submitted:

                if any(
                    answer is None
                    for answer in answers
                ):

                    st.warning(
                        "Answer all five questions "
                        "before submitting."
                    )

                else:

                    score = sum(
                        1
                        for index, question
                        in enumerate(quiz)
                        if answers[index]
                        == question["answer"]
                    )


                    percentage = round(
                        score
                        / len(quiz)
                        * 100
                    )


                    topic_results = (
                        calculate_topic_results(
                            quiz,
                            answers
                        )
                    )


                    record_performance(
                        file_id,
                        topic_results
                    )


                    st.session_state.quiz_result = {
                        "score": score,
                        "percentage": percentage,
                        "answers": answers,
                        "topics": topic_results
                    }


                    st.session_state.quiz_submitted = (
                        True
                    )


                    st.rerun()


        # ----------------------------------------------------
        # MAIN QUIZ RESULT
        # ----------------------------------------------------

        else:

            result = (
                st.session_state.quiz_result
            )


            c1, c2 = st.columns(2)


            with c1:

                st.metric(
                    "Score",
                    f"{result['score']}/"
                    f"{len(quiz)}"
                )


            with c2:

                st.metric(
                    "Percentage",
                    f"{result['percentage']}%"
                )


            if (
                result["percentage"]
                >= 80
            ):

                st.success(
                    "Strong performance. "
                    "Most of these concepts "
                    "are looking solid."
                )


            elif (
                result["percentage"]
                >= 60
            ):

                st.info(
                    "Good progress. Review the "
                    "weaker topics before moving on."
                )


            else:

                st.warning(
                    "Some concepts need another pass. "
                    "Use the topic breakdown below."
                )


            # ------------------------------------------------
            # KNOWLEDGE GAP DETECTOR
            # ------------------------------------------------

            st.subheader(
                "Knowledge Gap Detector"
            )


            for topic, values in (
                result["topics"].items()
            ):

                percentage = (
                    values["percentage"]
                )


                st.write(
                    f"**{topic}: "
                    f"{percentage}%**"
                )


                st.progress(
                    min(
                        max(
                            percentage / 100,
                            0
                        ),
                        1
                    )
                )


                if percentage < 60:

                    st.caption(
                        "Weak area"
                    )

                elif percentage < 80:

                    st.caption(
                        "Developing"
                    )

                else:

                    st.caption(
                        "Strong"
                    )


            # ------------------------------------------------
            # REVIEW
            # ------------------------------------------------

            st.subheader(
                "Question Review"
            )


            render_quiz_review(
                quiz,
                result["answers"]
            )


            c1, c2 = st.columns(2)


            with c1:

                if st.button(
                    "Generate New Quiz",
                    use_container_width=True
                ):

                    st.session_state.quiz = (
                        None
                    )

                    st.session_state.quiz_result = (
                        None
                    )

                    st.session_state.quiz_submitted = (
                        False
                    )

                    st.session_state.adaptive_quiz = (
                        None
                    )

                    st.rerun()


            with c2:

                weak_topics = [
                    topic
                    for topic, values
                    in result["topics"].items()
                    if values["percentage"] < 60
                ]


                adaptive_button = st.button(
                    "Practice Weak Areas",
                    use_container_width=True,
                    disabled=not weak_topics
                )


            # ------------------------------------------------
            # GENERATE WEAK AREA QUIZ
            # ------------------------------------------------

            if adaptive_button:

                previous_questions = "\n".join(
                    question["question"]
                    for question in quiz
                )


                weak_text = ", ".join(
                    weak_topics
                )


                prompt = f"""
Create exactly 5 NEW multiple-choice questions.

Focus primarily on these weak topics:

{weak_text}

Use ONLY the supplied notes.

Do NOT repeat these previous questions:

{previous_questions}

Return ONLY valid JSON as an array of exactly 5 objects.

Each object must contain:

"topic"
"question"
"options"
"answer"
"explanation"

Rules:
- exactly 4 options per question
- answer must exactly match one option
- no markdown
- no code fences
- questions should strengthen weak concepts

NOTES:
{note_context}
"""


                with st.spinner(
                    "Creating targeted practice..."
                ):

                    raw = ask_gemini(
                        prompt
                    )

                    parsed = parse_quiz(
                        raw
                    )


                if parsed:

                    st.session_state.adaptive_quiz = (
                        parsed
                    )

                    st.session_state.adaptive_submitted = (
                        False
                    )

                    st.session_state.adaptive_result = (
                        None
                    )

                    st.rerun()

                else:

                    st.error(
                        "The targeted quiz could not "
                        "be generated correctly. Try again."
                    )


        # ----------------------------------------------------
        # ADAPTIVE PRACTICE QUIZ
        # ----------------------------------------------------

        if st.session_state.adaptive_quiz:

            st.markdown("---")


            st.subheader(
                "Weak-Area Practice"
            )


            adaptive_quiz = (
                st.session_state.adaptive_quiz
            )


            if not st.session_state.adaptive_submitted:

                with st.form(
                    "adaptive_quiz_form"
                ):

                    adaptive_answers = []


                    for index, question in enumerate(
                        adaptive_quiz
                    ):

                        st.markdown(
                            f"### {index + 1}. "
                            f"{question['question']}"
                        )


                        answer = st.radio(
                            "Choose an answer",
                            question["options"],
                            index=None,
                            key=(
                                f"adaptive_"
                                f"{file_id}_"
                                f"{index}"
                            ),
                            label_visibility="collapsed"
                        )


                        adaptive_answers.append(
                            answer
                        )


                    submitted = (
                        st.form_submit_button(
                            "Submit Practice Quiz",
                            use_container_width=True
                        )
                    )


                if submitted:

                    if any(
                        answer is None
                        for answer
                        in adaptive_answers
                    ):

                        st.warning(
                            "Answer all five questions "
                            "before submitting."
                        )

                    else:

                        score = sum(
                            1
                            for index, question
                            in enumerate(
                                adaptive_quiz
                            )
                            if adaptive_answers[index]
                            == question["answer"]
                        )


                        percentage = round(
                            score
                            / len(adaptive_quiz)
                            * 100
                        )


                        topic_results = (
                            calculate_topic_results(
                                adaptive_quiz,
                                adaptive_answers
                            )
                        )


                        record_performance(
                            file_id,
                            topic_results
                        )


                        st.session_state.adaptive_result = {
                            "score": score,
                            "percentage": percentage,
                            "answers": adaptive_answers,
                            "topics": topic_results
                        }


                        st.session_state.adaptive_submitted = (
                            True
                        )


                        st.rerun()


            # ------------------------------------------------
            # ADAPTIVE RESULTS
            # ------------------------------------------------

            else:

                result = (
                    st.session_state.adaptive_result
                )


                c1, c2 = st.columns(2)


                with c1:

                    st.metric(
                        "Practice Score",
                        f"{result['score']}/"
                        f"{len(adaptive_quiz)}"
                    )


                with c2:

                    st.metric(
                        "Percentage",
                        f"{result['percentage']}%"
                    )


                render_quiz_review(
                    adaptive_quiz,
                    result["answers"]
                )


                if st.button(
                    "Finish Weak-Area Practice",
                    use_container_width=True
                ):

                    st.session_state.adaptive_quiz = (
                        None
                    )

                    st.session_state.adaptive_result = (
                        None
                    )

                    st.session_state.adaptive_submitted = (
                        False
                    )

                    st.rerun()


# ============================================================
# PROGRESS
# ============================================================

elif page == "Progress":

    # PAGE HEADER
    st.caption("LEARNING ANALYTICS")
    st.title("See how your understanding is improving")
    st.write(
        "Track your quiz performance across topics and identify "
        "what you've mastered, what is improving, and what still "
        "needs revision."
    )
    st.markdown("---")
    st.caption(f"Performance for: {uploaded_file.name}")


    performance = (
        get_performance_summary(
            file_id
        )
    )


    if not performance:

        st.info(
            "No quiz performance has been recorded "
            "for this PDF yet. Complete a quiz first."
        )

    else:

        averages = [
            values["average"]
            for values
            in performance.values()
        ]


        overall = round(
            sum(averages)
            / len(averages)
        )


        mastered = sum(
            1
            for values
            in performance.values()
            if values["average"] >= 80
        )


        weak_count = sum(
            1
            for values
            in performance.values()
            if values["average"] < 60
        )


        c1, c2, c3 = st.columns(3)


        with c1:

            st.metric(
                "Average",
                f"{overall}%"
            )


        with c2:

            st.metric(
                "Mastered Topics",
                mastered
            )


        with c3:

            st.metric(
                "Weak Topics",
                weak_count
            )


        st.markdown("---")


        for topic, values in sorted(
            performance.items()
        ):

            score = values["average"]


            st.subheader(
                topic
            )


            c1, c2, c3 = st.columns(3)


            with c1:

                st.metric(
                    "Average",
                    f"{score}%"
                )


            with c2:

                st.metric(
                    "Latest",
                    f"{values['latest']}%"
                )


            with c3:

                st.metric(
                    "Attempts",
                    values["attempts"]
                )


            st.progress(
                min(
                    max(
                        score / 100,
                        0
                    ),
                    1
                )
            )


            if score >= 80:

                st.success(
                    "Mastered"
                )

            elif score >= 60:

                st.info(
                    "Improving"
                )

            else:

                st.warning(
                    "Needs more revision"
                )


# ============================================================
# LAST-MINUTE REVISION
# ============================================================
elif page == "Revision":

    # PAGE HEADER
    st.caption("FOCUSED REVISION")
    st.title("Make the most of the time you have")
    st.write(
        "Create a focused last-minute revision plan based on "
        "your available study time and your quiz performance."
    )
    st.markdown("---")

    # REVISION SETUP
    st.subheader("Build your revision plan")
    st.caption(f"Currently studying: {uploaded_file.name}")

    revision_time = st.selectbox(
        "How much time do you have?",
        [
            "15 minutes",
            "30 minutes",
            "1 hour"
        ]
    )

    performance = get_performance_summary(
        file_id
    )

    if performance:

        performance_text = "\n".join(
            [
                (
                    f"- {topic}: "
                    f"{values['average']}% average, "
                    f"{values['latest']}% latest"
                )
                for topic, values
                in performance.items()
            ]
        )

        weak_topics = [
            topic
            for topic, values
            in performance.items()
            if values["average"] < 60
        ]

        if weak_topics:
            st.info(
                "StudyMate will prioritize your weaker areas: "
                + ", ".join(weak_topics)
            )
        else:
            st.success(
                "No major weak areas detected. Your revision plan "
                "will focus on reinforcing the most important concepts."
            )

    else:

        performance_text = (
            "No quiz performance is available yet."
        )

        st.info(
            "No quiz history is available yet. StudyMate will "
            "prioritize the most important concepts from your notes."
        )

    create_revision = st.button(
        "Create Revision Plan",
        use_container_width=True,
        type="primary"
    )

    # GENERATE REVISION PLAN
    if create_revision:

        prompt = f"""
You are StudyMate, an academic study assistant.

Create a focused last-minute revision guide for a student.

AVAILABLE TIME:
{revision_time}

PERFORMANCE:
{performance_text}

Use ONLY the notes below for academic content.

Prioritize weaker topics when performance data exists.

Structure the response as:

1. Revision Priority
2. Time Plan
3. Must-Know Concepts
4. Definitions / Facts to Remember
5. Quick Self-Test Questions
6. Final 2-Minute Checklist

Requirements:
- Make the amount of content realistic for {revision_time}.
- Be concise.
- Do not invent material absent from the notes.
- If there is no performance history, prioritize the most central concepts.
- Make the plan practical to follow within the available time.

NOTES:
{note_context}
"""

        with st.spinner(
            "Building your focused revision plan..."
        ):

            st.session_state.revision = ask_gemini(
                prompt
            )

    # DISPLAY REVISION PLAN
    if st.session_state.revision:

        st.markdown("---")
        st.caption("YOUR REVISION PLAN")
        st.subheader(f"{revision_time} Revision Plan")

        st.markdown(
            st.session_state.revision
        )