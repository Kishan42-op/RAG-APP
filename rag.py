from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import bs4
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader,TextLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from fastapi import UploadFile, File
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
import sentence_transformers
from pymongo import MongoClient

# Load environment variables
load_dotenv()

username = os.getenv("MONGO_USERNAME")
password = os.getenv("MONGO_PASSWORD")

# MongoDB configuration
MONGO_URI = "mongodb+srv://kishankachhadiya:RatI027pNWmdliE3@cluster0.pmwy3.mongodb.net/"  # Update this with your MongoDB connection URI

client = MongoClient(MONGO_URI)
db = client["UserHistory"]
qa_collection = db["QA"]

GROQ_API_KEY = os.getenv('GROQ_API_KEY')

llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")



# Embedding model
HuggingFace_embedding = HuggingFaceBgeEmbeddings(
                                    model_name="BAAI/bge-base-en-v1.5",
                                    model_kwargs={'device': 'cpu'},
                                    encode_kwargs={'normalize_embeddings': True}
                                    )

# Define the vector store
def initialize_vectorstore():
    try:
        return FAISS.load_local("vectordb",embeddings=HuggingFace_embedding, allow_dangerous_deserialization=True)
    except Exception as e:
        raise RuntimeError("Failed to initialize vector store: " + str(e))

vectorstore = initialize_vectorstore()

if not isinstance(vectorstore, FAISS):
    raise RuntimeError("Vector store initialization failed.")

retriever = vectorstore.as_retriever()

# Store for chat histories
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

### Contextualize question ###
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

### Answer question ###
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Define the RAG chain
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,  # The previously defined retrieval and question-answering chain
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# Define FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class QuestionRequest(BaseModel):
    session_id: str
    question: str

class AnswerResponse(BaseModel):
    answer: str

@app.post("/upload")
def upload_files(files: List[UploadFile] = File(...)):
    try:
        for file in files:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            with open(file_path, "wb") as f:
                f.write(file.file.read())

            # Load and process file
            if file.filename.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            else:
                raise HTTPException(status_code=400, detail="Unsupported file type")

            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)
            vectorstore = FAISS.from_documents(documents=splits, embedding=HuggingFace_embedding)

        # Save the updated vector store
        vectorstore.save_local("vectordb")

        return {"message": "Files uploaded and processed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    try:

        response = conversational_rag_chain.invoke(
            {"input": request.question},
            config={"configurable": {"session_id": request.session_id}},
        )
        # Get the answer from the RAG response
        answer = response["answer"]

        # Check if the session already exists in the MongoDB collection
        session_data = qa_collection.find_one({"session_id": request.session_id})
        
        if session_data:
            # Update the session's question-answer pairs
            qa_collection.update_one(
                {"session_id": request.session_id},
                {"$set": {f"questions.{request.question}": answer}}
            )
        else:
            # Insert a new session document
            qa_collection.insert_one({
                "session_id": request.session_id,
                "questions": {request.question: answer}
            })

        # Return the response to the client
        return {"answer": answer}

    except Exception as e:
        # Handle exceptions and return a 500 error
        raise HTTPException(status_code=500, detail=str(e))
