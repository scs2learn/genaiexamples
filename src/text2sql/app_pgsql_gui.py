from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from llama_index.core import Settings, SQLDatabase
from llama_index.core.indices.struct_store import SQLTableRetrieverQueryEngine
from llama_index.core.objects import (
    SQLTableNodeMapping,
    ObjectIndex,
    SQLTableSchema,
)
from llama_index.core import VectorStoreIndex

from sqlalchemy import create_engine
from llama_index.core.callbacks import (
    CallbackManager,
    LlamaDebugHandler,
)

import os
from dotenv import load_dotenv
import gradio as gr

def get_query_engine():
    model = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4")
    embedding = OpenAIEmbedding(api_key=os.getenv("OPENAI_API_KEY"), model="text-embedding-3-small")

    Settings.llm = model
    Settings.embed_model = embedding

    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager([llama_debug])

    # postgresql+psycopg2://postgres:suraj7177@localhost:5432/postgres

    engine = create_engine(f"postgresql+psycopg2://postgres:suraj7177@localhost:5432/northwind")

    nw_tables = [
        ("categories", "This table contains all the categories of Northwind."),
        ("suppliers", "This table contains all the Suppliers and vendors of Northwind."),
        ("products", "This table contains all the products of Northwind."),
        ("customers", "This table contains the users list who have purchased products from Northwind. contact_name field contains the customer name"),
        ("orders", "This table contains the orders list placed by customers of Northwind."),
        ("employees", "This table contains employee details of Northwind traders."),
        ("shippers", "This table contans the details of the shippers who ship the products from the traders to the end-customers.")
    ]

    sql_database = SQLDatabase(engine, include_tables=[table_name for table_name, _ in nw_tables])

    table_node_mapping = SQLTableNodeMapping(sql_database)

    table_schema_objs = []
    for table_name, table_description in nw_tables:
        table_schema_objs.append(
            SQLTableSchema(table_name=table_name, context_str=table_description))
        
    obj_index = ObjectIndex.from_objects(
        table_schema_objs,
        table_node_mapping,
        VectorStoreIndex,
        callback_manager=callback_manager,
    )

    query_engine = SQLTableRetrieverQueryEngine(
        sql_database,
        obj_index.as_retriever(similarity_top_k=len(nw_tables)),
    )

    return query_engine

def main():
    query_engine = get_query_engine()

    def process_user_input(user_input, chat_history):
        chat_history = []
    
        if not user_input:
            response = "Please enter a question"
        else:
            response = str(query_engine.query(user_input))

        chat_history.append((user_input, response))

        return chat_history, chat_history

    # Create the Gradio interface
    with gr.Blocks(title="Sample Application") as demo:
        with gr.Tab("Chatbot"):
            with gr.Row():
                chat_display = gr.Chatbot()
                message_ph = gr.Textbox(placeholder="Type your question")
            with gr.Row():
                state = gr.State()
                submit = gr.Button("Get answer to your question")

            submit.click(process_user_input, inputs=[message_ph, state], outputs=[chat_display, state])

    # Launch the application
    demo.launch()

if __name__ == "__main__":
    main()