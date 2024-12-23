from dotenv import load_dotenv
import os
import gradio as gr
from llama_index.core import StorageContext, load_index_from_storage, VectorStoreIndex, SimpleDirectoryReader, ChatPromptTemplate, Settings
from llama_index.llms.huggingface import HuggingFaceInferenceAPI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import firebase_admin
from firebase_admin import db, credentials
import datetime
import uuid
import random

# Load environment variables
load_dotenv()

# Authenticate to Firebase
# cred = credentials.Certificate("redfernstech-fd8fe-firebase-adminsdk-g9vcn-0537b4efd6.json")
# firebase_admin.initialize_app(cred, {"databaseURL": "https://redfernstech-fd8fe-default-rtdb.firebaseio.com/"})

# Configure the Llama index settings
Settings.llm = HuggingFaceInferenceAPI(
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    tokenizer_name="meta-llama/Meta-Llama-3-8B-Instruct",
    context_window=3000,
    token=os.getenv("HF_TOKEN"),
    max_new_tokens=512,
    generate_kwargs={"temperature": 0.1},
)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# Define the directory for persistent storage and data
PERSIST_DIR = "db"
PDF_DIRECTORY = 'data'  # Directory containing PDFs

# Ensure directories exist
os.makedirs(PDF_DIRECTORY, exist_ok=True)
os.makedirs(PERSIST_DIR, exist_ok=True)

# Variable to store current chat conversation
current_chat_history = []

def data_ingestion_from_directory():
    # Use SimpleDirectoryReader on the directory containing the PDF files
    documents = SimpleDirectoryReader(PDF_DIRECTORY).load_data()
    storage_context = StorageContext.from_defaults()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)

def handle_query(query):
    chat_text_qa_msgs = [
    (
        "user",
        """
        You are Clara, the friendly and knowledgeable chatbot of LBRCE (lbrce.ac.in). Your goal is to assist students, parents, and other users by providing accurate, professional, and helpful answers to queries related to the college. Always maintain a warm and engaging tone, starting responses with phrases like "As of my knowledge," "Currently, I understand that," "Based on the latest information," or "At present," to make the interaction feel more personalized.

        When users ask about admission possibilities, such as: 
        - "My rank is 15273 in EAPCET under EWS category, can I get into CSE department?"
        - "What is the closing rank for CSE in EAPCET 2023?"
        - "Can I apply for IT if my rank is 45000 under BC-B category?"
        
        Ensure that all critical details are present in the user's query, specifically:
        - **Rank**: The rank obtained in the entrance exam.
        - **Exam**: The specific exam (e.g., EAPCET, GATE, ICET).
        - **Category**: The reservation or special category (e.g., EWS, BC-A, SC).
        - **Desired Department**: The specific department or course they wish to join (e.g., CSE, AI&DS).

        If any of these details are missing, respond with a friendly prompt requesting the missing information. For example:
        - "To provide a more accurate answer, could you please specify your reservation category?"
        - "It seems like you mentioned your rank but not the desired department. Could you share that as well?"
        - "I noticed that your query lacks the exam name. Kindly let me know which exam you are referring to (e.g., EAPCET, ICET)."

        When responding, avoid technical jargon or phrases like "as per the provided document" or "according to the data" and instead make your answers conversational and easy to understand. Be concise but thorough, ensuring that the user leaves with a complete understanding of the information they requested.

        Additionally, Clara should be able to handle general queries such as:
        - Details about various undergraduate and postgraduate programs offered.
        - Information on special category reservations, including EWS, CAP, NCC, PHO, PHV, PHH, and SG.
        - Information about facilities, faculty, placements, and scholarships.
        - Details of the application process, important dates, and eligibility criteria.
        - Campus life, hostel facilities, and extracurricular opportunities.

        When providing information, ensure that:
        - For ranks and admission-related queries, respond with a clear indication of the eligibility or chance of admission based on the provided rank and category.
        - For general queries, summarize the information in a user-friendly way without overwhelming the user.
        - Clarify any complex details by breaking them into simpler explanations.

        Remember, the user's experience should feel as if they are speaking to a knowledgeable guide who understands their needs and provides clear, direct answers with empathy and professionalism.
        
        {context_str}
        Question:
        {query_str}
        """
    )
]

    text_qa_template = ChatPromptTemplate.from_messages(chat_text_qa_msgs)

    # Load index from storage
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

    # Use chat history to enhance response
    context_str = ""
    for past_query, response in reversed(current_chat_history):
        if past_query.strip():
            context_str += f"User asked: '{past_query}'\nBot answered: '{response}'\n"

    query_engine = index.as_query_engine(text_qa_template=text_qa_template, context_str=context_str)
    answer = query_engine.query(query)

    if hasattr(answer, 'response'):
        response = answer.response
    elif isinstance(answer, dict) and 'response' in answer:
        response = answer['response']
    else:
        response = "Sorry, I couldn't find an answer."

    # Update current chat history
    current_chat_history.append((query, response))

    return response

# Example usage: Process PDF ingestion from directory
print("Processing PDF ingestion from directory:", PDF_DIRECTORY)
data_ingestion_from_directory()

# Define the Gradio interface function
def chat_interface(message):
    response = handle_query(message)  # Get the chatbot response
    return response

# Launch Gradio interface
gr.Interface(fn=chat_interface, 
             inputs="text", 
             outputs="text", 
             title="Clara - LBRCE Chatbot", 
             description="Ask questions related to LBRCE college.") \
.launch()
