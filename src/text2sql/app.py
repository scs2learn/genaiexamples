from llama_index.core import Settings, SQLDatabase
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
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

def get_query_engine():
    hf_llm = HuggingFaceInferenceAPI(
        model="mistralai/Mistral-7B-Instruct-v0.3",
        token=os.getenv("HF_TOKEN")
    )

    hf_embedding = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    Settings.llm = hf_llm
    Settings.embed_model = hf_embedding

    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager([llama_debug])

    engine = create_engine(f"postgresql+psycopg2://<username>:<password>@localhost:5432/northwind")

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
    # text_to_sql(query_engine=query_engine,question="How many tables are there in the database?")
    # text_to_sql(query_engine=query_engine,question="List the how many cusomters from each countr. List them in descending order")
    query="What are the categories of products sold?"
    query="What are the categories of products sold? List them with description"
    query="Give me a list of orders in the last week. Also include the user (firstname and lastname) who placed the order and the total amount of the order"
    response = query_engine.query(query)
    print(response)

if __name__ == "__main__":
    main()