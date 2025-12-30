from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import logging
import json
from datetime import datetime

from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory
from typing import Dict, List, Any
# Flask- main class we use to create web app
# request- holds all the info about an incoming req
# jsonify- converts python dict to json response for sending data over web

# creating a flask web sever instance
app=Flask(__name__,template_folder='../templates',static_folder='../static')
CORS(app) # This line enables CORS for all routes
# __name__ => spcl variable that represents name of curr module

store: Dict[str, ChatMessageHistory] = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    """Retrieves or creates a ChatMessageHistory for a given session ID."""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]
# to store chat history for that session in serverless environment(clears after timeout)

@app.route('/')
def index():
    return render_template('index.html')

# Configure the Gemini API client
# we use an environment variable for the API key
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set")



# System prompt loader
def load_system_prompt():
    try:
        with open("system_prompt.txt", "r", encoding="utf-8") as file:
            return file.read()
    except Exception:
        return "You are a helpful AI assistant."

system_prompt = load_system_prompt()


@app.route('/chat',methods=['POST'])
# decorator for diff url
# /chat => specific path for our chat endpoint
# methods=['POST']: very imp, It specifies that this route should only respond to a POST request. A POST request is used to send data to the server (like a user's message), as opposed to a GET request which is used to retrieve data.
def chat():
    data = request.json
    user_message = data.get('message')
    session_id = "mohit_chatbot_session"

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    try:
        # Load knowledge base (if available)
        docs = []
        if os.path.exists("knowledge_base.txt"):
            loader = TextLoader("knowledge_base.txt")
            docs = loader.load()

        # Generation chain (initialize Gemini LLM)
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=gemini_api_key,
            temperature=0.7
        )

        # The chat prompt is the core of our new system. It includes the system prompt, history, and user input.
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt + "\n\nRelevant context:\n{context}"),("placeholder", "{history}"),
            ("user", "{input}"),
        ])

        # We'll create a chain that combines the retrieved documents and the chat prompt
        # The AI will decide when to use the context and when to ignore it.
        combine_docs_chain = create_stuff_documents_chain(llm, chat_prompt)

        # Add History Management
        # Wrap the chain to automatically manage chat history
        with_history = RunnableWithMessageHistory(
            combine_docs_chain,
            get_session_history,
            # The history key maps to the placeholder in the ChatPromptTemplate
            input_messages_key="input",
            history_messages_key="history",
        )

        # Invoke the chain to get a response
        # The history manager automatically saves the current message and loads previous ones.
        response = with_history.invoke(
            {"input": user_message, "context": docs},
            config={"configurable": {"session_id": session_id}}
        )

        #log the conversation
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_message": user_message,
            "chatbot_response": response
        }
        print(json.dumps(log_entry))
        
        return jsonify({"response": str(response)})

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": "Failed to get a response from the AI."}), 500

    # jsonify fn converts this dict into a json obj
