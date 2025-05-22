import os
from huggingface_hub import InferenceClient
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from connect_memory_with_llm import HF_TOKEN, TogetherAILLM
import warnings

warnings.filterwarnings('ignore')
DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    """Load and cache the vector database."""
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        st.error(f"Error loading vectorstore: {e}")
        return None

def set_custom_prompt(custom_prompt_template):
    """Set the custom prompt template for the LLM."""
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )
    return prompt

@st.cache_resource
def load_llm():
    """Load and cache the LLM using custom TogetherAILLM class."""
    try:
        token = os.environ.get("HF_TOKEN") or HF_TOKEN
        if not token:
            raise ValueError("HF_TOKEN not found in environment variables or imports")
        
        client = InferenceClient(
            provider="together",
            api_key=token,
        )
        
        llm = TogetherAILLM(
            client=client,
            model="mistralai/Mistral-7B-Instruct-v0.3",
            temperature=0.5,
            max_tokens=512,
        )
        return llm
    except Exception as e:
        raise ValueError(f"Failed to load LLM: {e}")

@st.cache_resource
def initialize_qa_chain():
    """Initialize and cache the QA chain."""
    try:
        vectorstore = get_vectorstore()
        if vectorstore is None:
            return None
            
        llm = load_llm()
        
        custom_prompt_template = """Use the pieces of the information provided to answer the question.
If you don't know the answer, say "I don't know, don't try to make up an answer".
Don't provide anything out of the information provided.

Context: {context}
Question: {question}

Start the answer directly. No small talk."""
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": set_custom_prompt(custom_prompt_template)},
        )
        return qa_chain
    except Exception as e:
        st.error(f"Error initializing QA chain: {e}")
        return None

def main():
    st.title("Ask MediChatbot!")
    
    # Add clear cache button for debugging
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("Clear Cache"):
            st.cache_resource.clear()
            st.session_state.clear()
            st.rerun()
    
    # Initialize QA chain
    qa_chain = initialize_qa_chain()
    
    if qa_chain is None:
        st.error("Failed to initialize the chatbot. Please check your configuration.")
        st.stop()

    # Initialize session state for messages
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display conversation history
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    # User input
    user_input = st.chat_input("Enter your medical question:")

    # Process user input
    if user_input:
        # Debug: Show what input was received
        st.write(f"Debug: Processing input: '{user_input}'")
        
        # Add user message to session state
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate response
        with st.chat_message("assistant"):
            try:
                with st.spinner("Processing your question..."):
                    # Make sure we're using the current user input
                    response = qa_chain.invoke({"query": user_input})
                    
                    result = response['result']
                    source_documents = response.get('source_documents', [])
                    
                    # Display the main answer
                    st.markdown(result)
                    
                    # Display sources in expandable section
                    if source_documents:
                        with st.expander("View Sources"):
                            for i, doc in enumerate(source_documents):
                                st.write(f"**Source {i+1}:**")
                                content_preview = doc.page_content[:200]
                                if len(doc.page_content) > 200:
                                    content_preview += "..."
                                st.write(content_preview)
                                st.divider()
                    
                    # Add to session state
                    st.session_state.messages.append({"role": "assistant", "content": result})

            except Exception as e:
                error_message = f"An error occurred: {str(e)}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})

if __name__ == "__main__":
    main()