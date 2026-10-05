import gradio as gr
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import AzureChatOpenAI
from constants import AZURE_ENDPOINT,API_VERSION,DEPLOYMENT_NAME,OPENAI_API_KEY,OPEN_AI_TYPE,TEMPERATURE,INDEX_FOLDER


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

llm =  AzureChatOpenAI(
                    azure_endpoint=AZURE_ENDPOINT,
                    openai_api_version=API_VERSION,
                    deployment_name=DEPLOYMENT_NAME,
                    openai_api_key=OPENAI_API_KEY,
                    openai_api_type=OPEN_AI_TYPE,
                    temperature=TEMPERATURE
            )   


def find_match(query):
    try:

        print(f"Loading FAISS index from: {INDEX_FOLDER}")
        index = FAISS.load_local(INDEX_FOLDER,embeddings,"index",allow_dangerous_deserialization=True)
        results = index.similarity_search(query, k=5)
        if not results:
            return {"error": "No relevant results found for the query."}
        return results
    except FileNotFoundError:
        return {"error": f"FAISS index folder '{INDEX_FOLDER}' not found."}
    except Exception as e:
        return {"error": f"Error during FAISS search: {str(e)}"}
    

def rag_answer(question: str) -> str:
    if question.lower().strip() in ['exit', 'quit', 'bye', 'goodbye']:
        return "👋 Thank you for chatting! Have a great day!"
    context = find_match(question)
    if not context:
        return "I don't know."
    prompt = f"""
    You are a friendly and helpful chatbot.
    Your behavior rules:
    - If the user greets you (e.g., "hi", "hello", "hey"), respond with a polite and friendly greeting.
    - Answer the user's question using ONLY the information provided in the Context below.
    - Do NOT use any outside knowledge, assumptions, or reasoning beyond the Context.
    - Answer elaborately and in detail.
    - If the question cannot be answered using the Context, respond with "I don't know."
    Keep your responses clear, concise, and friendly.
    Context:
    {context}
    User Question:
    {question}
    Assistant Response:
    """
    print(f"Prompt: {prompt}")
    response = llm.invoke(prompt)
    return response.content.strip()
def chat(user_input, history):
    if history is None:
        history = []
    answer = rag_answer(user_input)
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": answer})
    return history
with gr.Blocks(css="""
    body, .gradio-container {
        background: linear-gradient(135deg, #f5f7fa, #e4ecf7);
        font-family: 'Segoe UI', sans-serif;
    }
 
    .chat-container {
        max-width: 900px;
        margin: auto;
        background: white;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0px 10px 30px rgba(0,0,0,0.08);
    }
              
 
    .gr-chatbot {
          border: 2px solid #000000 !important;    
        border-radius: 12px;
        background-color: #fafafa;
    }
 
    .gr-textbox textarea {
        border-radius: 12px !important;
        font-size: 15px;
    }
 
    .gr-button {
        border-radius: 12px !important;
        font-weight: 600;
    }
 
    h1 {
        color: #1f2937;
        font-size: 34px;
    }
 
    .subtitle {
        color: #4b5563;
        font-size: 15px;
        margin-bottom: 15px;
    }
""") as demo:
 
    gr.Markdown(
        """
        <div style="text-align:center; margin-bottom:20px;">
            <h1>🎥 YouTube RAG Chatbot</h1>
            <p class="subtitle">
                Ask questions from your indexed YouTube videos.<br>
                Type <b>exit</b>, <b>quit</b>, or <b>bye</b> to end the conversation.
            </p>
        </div>
        """
    )
 
    with gr.Column(elem_classes="chat-container"):
        chatbot = gr.Chatbot(
            height=420,
            # bubble_full_width=False
        )
 
        msg = gr.Textbox(
            label="Your Query",
            placeholder="Ask something from the YouTube video…",
           
        )
 
        msg.submit(chat, inputs=[msg, chatbot], outputs=chatbot)
 
demo.launch()