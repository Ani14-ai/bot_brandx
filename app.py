import os
from typing import List
from flask import Flask, request, jsonify
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain.chains.openai_tools import create_extraction_chain_pydantic
from langchain.chains import create_sql_query_chain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import tool
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate
from operator import itemgetter
load_dotenv()

# Initialize LLMs for query generation and table extraction
sql_agent_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
table_extractor_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Initialize database connection
db_uri = "mssql+pyodbc://BrandXusr:BrandXusr1234@103.239.89.99:21433/BrandXAppDB?driver=ODBC+Driver+17+for+SQL+Server"
db_instance = SQLDatabase.from_uri(db_uri)

# Create query chain
query_chain = create_sql_query_chain(sql_agent_llm, db_instance)

# Define Table model for handling table names
from langchain_core.pydantic_v1 import BaseModel, Field

class Table(BaseModel):
    """Represents a table in the SQL database."""
    name: str = Field(description="Name of table in SQL database.")

# Define table extraction and category mapping chain
system_message = """
Return the names of the SQL tables that are relevant to the user question. Use proper joining techniques and search for proper categories according to user preferences.
The available tables are:
- Product
- Category
- Product_Category_Mapping
"""

category_chain = create_extraction_chain_pydantic(pydantic_schemas=Table, llm=table_extractor_llm, system_message=system_message)

def get_tables(categories: List[Table]) -> List[str]:
    """Maps category names to corresponding SQL table names."""
    tables = []
    for category in categories:
        if category.name.lower() == "product":
            tables.append("Product")
        elif category.name.lower() == "category":
            tables.append("Category")
        elif category.name.lower() == "product_category_mapping":
            tables.append("Product_Category_Mapping")
    return tables

table_chain = category_chain | get_tables

# Combine the table_chain with the SQL query chain
table_chain = {"input": itemgetter("question")} | table_chain
full_chain = RunnablePassthrough.assign(table_names_to_use=table_chain) | query_chain

# Define the database tool agent with improved logic for handling empty results
@tool
def query_brandx_db(query: str) -> str:
    """Query the BrandXAppDB to retrieve product data or inform the user if no products match the query."""
    try:
        query1 = full_chain.invoke({"question": query})
        # Execute the SQL query and handle the results
        result = db_instance.run(query1)
        
        # Check if the result is empty
        if not result or len(result) == 0:
            return "No products found matching your query. Please refine your search."
        
        return result
    
    except Exception as e:
        return f"An error occurred while processing your query: {str(e)}"

# Initialize primary LLM
primary_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# Define the agent tools
brandx_db_agent = query_brandx_db

agents = [brandx_db_agent]

# Define system message for the agent
system_message_template = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=[],  
        template="You are a helpful shopping assistant called Noura. You help customers by suggesting them products from the inventory according to their preferences. You only return the product id while suggesting the product."
    )
)

human_message_template = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["input"],  # Expecting an input from the user
        template="{input}"
    )
)

prompt_template = ChatPromptTemplate.from_messages([
    system_message_template,  # System message
    MessagesPlaceholder(variable_name="chat_history", optional=True),  
    human_message_template,  # Human input message
    MessagesPlaceholder(variable_name="agent_scratchpad")  # Placeholder for agent scratchpad
])

# Create and initialize the agent
agent = create_openai_tools_agent(llm=primary_llm, tools=agents, prompt=prompt_template)

agent_executor = AgentExecutor(agent=agent, tools=agents, verbose=True)

# Flask app definition
app = Flask(__name__)

class QueryRequest(BaseModel):
    question: str

# Flask route to handle user queries
@app.route("/query", methods=["POST"])
def query_databases():
    try:
        # Parse JSON request
        data = request.json
        question = data.get('question')

        if not question:
            return jsonify({"error": "Question is required"}), 400

        # Pass the question to the agent executor and run the tool
        response = agent_executor.invoke({"input": question})
        return jsonify({"response": response})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
