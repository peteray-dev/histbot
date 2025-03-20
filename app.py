from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embedding
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
from store_index import PineconeHandler
import config

import requests
# import pdfkit
from bs4 import BeautifulSoup
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

# print(CHATBOT_NAME)
load_dotenv()
app = Flask(__name__)

UPLOAD_FOLDER = 'Data/upload'  # Directory for uploaded PDFs
SCRAPE_FOLDER = 'Data/URL'     # Directory for scraped PDFs
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SCRAPE_FOLDER, exist_ok=True)


PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

embeddings = download_hugging_face_embedding()

docsearch = None
@app.route('/generate_chatbot', methods=['POST'])
def generate_chatbot():
    global docsearch
    """Generate a chatbot based on uploaded documents and store in a new namespace"""
    chatbot_name = config.get_chatbot_name().replace(" ", "_").lower()
    namespace = f"user_id_{chatbot_name}"  # Generate a namespace
    index_name = "histphilbot"
    
    print(f"🚀 Generating chatbot for namespace: {namespace}")
    
    pinecone_handler = PineconeHandler(index_name=index_name)

    if pinecone_handler.index_exists():
        print(f"⚡ Namespace '{namespace}' already exists. Using existing data.")
        docsearch = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embeddings,
            namespace=namespace
        )
    else:
        print(f"🚀 Creating new namespace '{namespace}' and uploading documents...")
        pinecone_handler.create_index()
        docsearch = pinecone_handler.upsert_documents()

    return jsonify({"message": f"Chatbot '{chatbot_name}' generated successfully!"}), 200


# Moved docsearch inside function to prevent auto-execution on startup
retriever = None
memory = None
rag_chain = None


@app.route('/start_chatbot', methods=['POST'])
def start_chatbot():
    """Start the chatbot with the latest namespace"""
    global retriever, memory, rag_chain, docsearch

    chatbot_name = config.get_chatbot_name().replace(" ", "_").lower()
    namespace = f"user_id_{chatbot_name}"
    
    print(f"⚡ Starting chatbot for namespace: {namespace}")

    # docsearch = PineconeVectorStore.from_existing_index(
    #     index_name="histphilbot",
    #     embedding=embeddings,
    #     namespace=namespace
    # )
    if docsearch is None:
        print("❌ Error: Chatbot not generated yet. Please generate chatbot first.")
        return jsonify({"error": "Chatbot not generated. Click 'Generate Chatbot' first!"}), 400


    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    llm = OpenAI(temperature=0.1, max_tokens=500)
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{question}")])

    memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history", return_messages=True)

    rag_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=retriever, memory=memory, combine_docs_chain_kwargs={"prompt": prompt}
    )

    return jsonify({"message": "Chatbot started successfully!"}), 200



# Embed each chunk and upsert the embeddings into your Pinecone index.
# docsearch = PineconeVectorStore.from_existing_index(
#     index_name=index_name,
#     embedding=embeddings
# )

# print(docsearch)

# docsearch = generate_chatbot()
# retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})



# llm = OpenAI(temperature=0.1, max_tokens=500)
# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system_prompt),
#         ("human", "{question}"),
#     ]
# )

# question_answer_chain = create_stuff_documents_chain(llm, prompt)
# rag_chain = create_retrieval_chain(retriever, question_answer_chain)
# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, input_key="question", output_key="answer")

# memory = ConversationSummaryMemory(
#     llm=llm,  # Use the LLM for summarization
#     memory_key="chat_history",
#     return_messages=True
# )

# Create conversational RAG chain
# rag_chain = ConversationalRetrievalChain.from_llm(
#     llm=llm,
#     retriever=retriever,
#     memory=memory,
#     combine_docs_chain_kwargs={"prompt": prompt}
# )


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"question": msg, 
                                 "chat_history": memory.load_memory_variables({})["chat_history"]})
    print("Response : ", response["answer"])
    return str(response["answer"])



# Allowed file types for upload
ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ------------------ FILE UPLOAD HANDLING ------------------ #
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        # Ensure the directory exists before saving
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        file.save(filepath)
        return jsonify({"message": f"File uploaded successfully: {filename}"}), 200
    else:
        return jsonify({"error": "Invalid file format. Only PDFs are allowed."}), 400

# ------------------ WEB SCRAPING HANDLING ------------------ #
@app.route('/scrape', methods=['POST'])
def scrape_and_save():
    data = request.get_json()
    url = data.get("url")

    print("Scraping URL:", url)

    if not url:
        return jsonify({"error": "No URL provided"}), 400

    try:
        # Fetch website content
        response = requests.get(url)
        response.raise_for_status()

        # Parse the HTML
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract the page title
        title = soup.title.string.strip() if soup.title else "Untitled Document"

        # Start organizing the content
        content_sections = [f"{title}\n", "=" * len(title), "\n"]

        # Extract all meaningful content while maintaining structure
        for tag in soup.find_all(["h1", "h2", "h3", "p", "ul", "ol", "li", "pre", "code", "table", "a"]):
            if tag.name.startswith("h"):  # Headings
                content_sections.append(f"\n{tag.get_text().strip()}\n")
                content_sections.append("-" * len(tag.get_text().strip()) + "\n")
            elif tag.name == "p":  # Paragraphs
                paragraph = tag.get_text().strip()
                if paragraph:
                    content_sections.append(paragraph + "\n")
            elif tag.name in ["ul", "ol"]:  # Lists
                content_sections.append("")  # Line break before lists
                for li in tag.find_all("li"):
                    content_sections.append(f"- {li.get_text().strip()}")
                content_sections.append("")  # Line break after lists
            elif tag.name == "pre":  # Code blocks
                content_sections.append("\nCode Example:\n")
                content_sections.append("```\n" + tag.get_text().strip() + "\n```\n")
            elif tag.name == "code":  # Inline code
                content_sections.append(f"`{tag.get_text().strip()}`")
            elif tag.name == "a":  # Links
                link_text = tag.get_text().strip()
                link_url = tag.get("href")
                if link_url and not link_text.startswith("#"):  # Ignore empty or internal links
                    content_sections.append(f"\nReference: {link_text} - {link_url}\n")
            elif tag.name == "table":  # Tables
                content_sections.append("\nTable Data:\n")
                rows = tag.find_all("tr")
                for row in rows:
                    cols = [col.get_text().strip() for col in row.find_all(["td", "th"])]
                    content_sections.append(" | ".join(cols))
                content_sections.append("")  # Line break after table

        # Convert structured content into formatted text
        structured_text = "\n".join(content_sections)

        # Generate filename
        filename = url.replace("https://", "").replace("http://", "").replace("/", "_") + ".txt"
        filepath = os.path.join(SCRAPE_FOLDER, filename)

        # Save structured text as a .txt file
        with open(filepath, "w", encoding="utf-8") as file:
            file.write(structured_text)

        return jsonify({"message": f"Website content saved as structured TXT in {filepath}"}), 200

    except Exception as e:
        print("Scraping Error:", str(e))
        return jsonify({"error": str(e)}), 500
    
# Store chatbot name
@app.route('/set_chatbot_name', methods=['POST'])
def set_chatbot_name():
    global CHATBOT_NAME
    data = request.get_json()
    chatbot_name = data.get("chatbot_name")

    if not chatbot_name:
        return jsonify({"error": "No chatbot name provided"}), 400

    # Store the chatbot name globally
    # config.CHATBOT_NAME = chatbot_name
    # Generate Pinecone namespace using chatbot name (formatted properly)
    # pinecone_namespace = chatbot_name.replace(" ", "_").lower()

    config.set_chatbot_name(chatbot_name)

    return jsonify({
        "message": f"Chatbot name set to {config.get_chatbot_name()}.",
        # "pinecone_namespace": pinecone_namespace
    }), 200
    
@app.route('/get_chatbot_name', methods=['GET'])
def get_chatbot_name():
    return jsonify({"chatbot_name": config.get_chatbot_name()}), 200


    
if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8000, debug= True)