import os
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_core.language_models.llms import LLM
from typing import Any, List, Mapping, Optional

# Load environment variables from .env file
load_dotenv()

# Step 1: Define a custom LLM class for Together AI
class TogetherAILLM(LLM):
    client: Any
    model: str
    temperature: float = 0.5
    max_tokens: int = 512
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop=stop
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error calling TogetherAI API: {e}")
            raise
    
    @property
    def _llm_type(self) -> str:
        return "together-ai"
        
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"model": self.model}

# Step 2: Setup LLM with TogetherAI
HF_TOKEN = os.environ.get("HF_TOKEN")  # This should be your Together AI API key

# Verify HF_TOKEN is set
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set. Please set it in the .env file.")

def load_llm():
    """
    Load the LLM using custom TogetherAILLM class.
    """
    try:
        # Create a client that uses Together AI as the provider
        client = InferenceClient(
            provider="together",
            api_key=HF_TOKEN,
        )
        
        # Create a LangChain wrapper around the client
        llm = TogetherAILLM(
            client=client,
            model="mistralai/Mistral-7B-Instruct-v0.3",
            temperature=0.5,
            max_tokens=512,
        )
        return llm
    except Exception as e:
        raise ValueError(f"Failed to load LLM: {e}")

# Step 3: Setup prompt template
custom_prompt_template = """
    Use the pieces of the information provided to answer the question.
    If you don't know the answer, say "I don't know, don't try to make up an answer".
    Don't provide anything out of the information provided.

    Context: {context}
    Question: {question}

    Start the answer directly. No small talk.
"""

def set_custom_prompt(custom_prompt_template):
    """
    Set the custom prompt template for the LLM.
    """
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )
    return prompt

# Step 4: Load Database
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}  # Set to "cuda" if using GPU
)
try:
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
except Exception as e:
    raise ValueError(f"Failed to load FAISS vectorstore from {DB_FAISS_PATH}: {e}")

# Step 5: Create QA Chain
try:
    qa_chain = RetrievalQA.from_chain_type(
        llm=load_llm(),
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": set_custom_prompt(custom_prompt_template)},
    )
except Exception as e:
    raise ValueError(f"Failed to create QA chain: {e}")

# Step 6: Process user query
try:
    user_query = input("Enter your question: ")
    response = qa_chain.invoke({"query": user_query})
    print("Answer:", response['result'])
    print("Source Documents:", [doc.page_content for doc in response['source_documents']])  # Simplified output
except Exception as e:
    print(f"Error processing query: {e}")
    import traceback
    traceback.print_exc()  # Print full stack trace for better debugging