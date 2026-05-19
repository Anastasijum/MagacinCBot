import os
import glob
import PyPDF2
import anthropic
from flask import Flask, request, jsonify, send_from_directory
import chromadb
from dotenv import load_dotenv
import uuid

load_dotenv()

# PDF folder - relative path, PDFs go in /app/PDF/ on the server
PDF_DIR = os.environ.get("PDF_DIR", os.path.join(os.path.dirname(__file__), "PDF"))

def pdf_files():
    return glob.glob(f"{PDF_DIR}/**/*.pdf", recursive=True)

def documents(paths):
    doc = []
    for path in paths:
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    doc.append({
                        "id": f"{os.path.basename(path)}_{i}",
                        "text": text
                    })
    return doc

api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY environment variable is not set")

app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(24).hex())

client = anthropic.Anthropic(api_key=api_key)
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("magacin")

sessions = {}

def inicijalizuj_bazu():
    docs = documents(pdf_files())
    if not docs:
        print("UPOZORENJE: Nisu pronadjeni PDF fajlovi u:", PDF_DIR)
        app.config['RAG_ENABLED'] = False
        app.config['DOCUMENTS'] = []
        return

    app.config['DOCUMENTS'] = docs

    try:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        app.config['EMBEDDER'] = embedder
        app.config['RAG_ENABLED'] = True

        texts = [d["text"] for d in docs]
        embeddings_ = embedder.encode(texts).tolist()
        collection.add(
            ids=[d["id"] for d in docs],
            documents=[d["text"] for d in docs],
            embeddings=embeddings_
        )
        print(f"RAG mod: AKTIVAN — ucitano {len(docs)} stranica iz PDF-ova")
    except ImportError:
        app.config["RAG_ENABLED"] = False
        print("RAG mod: NEAKTIVAN (sentence_transformers nije instaliran)")

def pretrazi_kontekst(upit):
    docs = app.config.get('DOCUMENTS', [])
    if not app.config.get("RAG_ENABLED"):
        return "\n".join(d["text"] for d in docs)
    embedder = app.config.get("EMBEDDER")
    emb_upit = embedder.encode([upit]).tolist()
    results = collection.query(query_embeddings=emb_upit, n_results=3)
    return "\n\n".join(results["documents"][0])

@app.route("/")
def index():
    return send_from_directory('templates', 'index.html')

@app.route('/chat', methods=['POST'])
def rag():
    data = request.json
    session_id = data.get("session_id")
    mess = data.get("message", '').strip()

    if not mess:
        return jsonify({"error": "Poruka je prazna"}), 400

    if session_id not in sessions:
        sessions[session_id] = []

    kontekst = pretrazi_kontekst(mess)
    prompt = f"""
        Koristi sledece informacije o kulturnom centru da odgovoris:

        ---INFORMACIJE---
        {kontekst}

        Pitanje korisnika: {mess}
    """
    history = sessions[session_id].copy()
    history.append({"role": "user", "content": prompt})

    try:
        answer = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system="""Ti si ljubazni i profesionalni asistent Kulturnog centra Magacin.
            Odgovaraj na jeziku na kom je postavljeno pitanje.
            Ako nešto ne znaš, predloži korisniku da kontaktira institut direktno.
            Ne izmišljaj informacije koje nisu date.
            """,
            messages=history
        )

        ans = answer.content[0].text

        sessions[session_id].append({"role": "user", "content": mess})
        sessions[session_id].append({"role": "assistant", "content": ans})

        if len(sessions[session_id]) > 20:
            sessions[session_id] = sessions[session_id][-20:]

        return jsonify({"response": ans, "session_id": session_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/new-session', methods=['POST'])
def new_session():
    session_id = str(uuid.uuid4())
    sessions[session_id] = []
    return jsonify({"session_id": session_id})

@app.route('/health')
def health():
    return jsonify({"status": "ok", "rag_enabled": app.config.get('RAG_ENABLED', False)})

if __name__ == '__main__':
    inicijalizuj_bazu()
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
