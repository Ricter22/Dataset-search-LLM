import re
import pandas as pd
import os
import datamart_profiler
import numpy as np
import openai
from dotenv import load_dotenv
from openai import OpenAI
import json
from collections import Counter
from typing import List, Tuple, Dict, Iterable


#we use this function to preprocess the dataset and generate a data profile summary
def generate_data_profile(df, context):

    metadata = datamart_profiler.process_dataset(df)

    profile_summary = []

    # Iterate through each column in the metadata and summarize the details
    for column_meta in metadata['columns']:
        column_summary = f"**{column_meta['name']}**: "
        
        # Structural type
        structural_type = column_meta.get('structural_type', 'Unknown')
        column_summary += f"Data is of type {structural_type.split('/')[-1].lower()}. "

        # Number of distinct values (if applicable)
        if 'num_distinct_values' in column_meta:
            num_distinct_values = column_meta['num_distinct_values']
            column_summary += f"There are {num_distinct_values} unique values. "

        # Check if the column is numeric
        if pd.api.types.is_numeric_dtype(df[column_meta['name']]):
            column_summary += "This column is numeric. " 
            mean_value = df[column_meta['name']].mean()
            max_value = df[column_meta['name']].max()
            min_value = df[column_meta['name']].min()
            column_summary += f"Mean: {mean_value}, Max: {max_value}, Min: {min_value}. "
        elif pd.api.types.is_datetime64_any_dtype(df[column_meta['name']]):
            column_summary += "This column is of datetime type. "
            # Calculate and add the range of dates
            min_date = df[column_meta['name']].min()
            max_date = df[column_meta['name']].max()
            column_summary += f"Date range: from {min_date} to {max_date}. "
        else:
            # For categorical columns, get the top 3 most frequent categories
            value_counts = df[column_meta['name']].value_counts()
            if value_counts.nunique() > 1:
                top_categories = df[column_meta['name']].value_counts().nlargest(3).index.tolist()
                column_summary += f"Top 3 frequent values: {', '.join(top_categories)}. "

        # Handle coverage (if available)
        if 'coverage' in column_meta:
            low=0
            high=0
            for i in range(len(column_meta['coverage'])):
                if(column_meta['coverage'][i]['range']['gte']<low):
                    low=column_meta['coverage'][i]['range']['gte']
                if(column_meta['coverage'][i]['range']['lte']>high):
                    high=column_meta['coverage'][i]['range']['lte']
            column_summary += f"Coverage spans from {low} to {high}. "

        # Append the summarized profile for this column
        profile_summary.append(column_summary)

    page_title = context.get("table_page_title", "") if isinstance(context, dict) else str(context)
    section_title = context.get("table_section_title", "") if isinstance(context, dict) else ""

    # Assicurati che tutti gli elementi di profile_summary siano stringhe
    safe_profile_lines = [str(x) for x in profile_summary]

    final_profile_summary = (
        f"The key data profile information for the dataset {page_title} {section_title} includes:\n"
        + "\n".join(safe_profile_lines)
    )

    #final_profile_summary = "The key data profile information for the dataset " + context["table_page_title"] + " " + context["table_section_title"] +" includes:\n" + '\n'.join(profile_summary)

    return final_profile_summary

def make_unique(lst):
    """
    Rende univoci gli elementi di una lista aggiungendo un numero crescente ai duplicati.
    """
    seen = {}
    result = []
    for item in lst:
        if item not in seen:
            seen[item] = 1
            result.append(item)
        else:
            seen[item] += 1
            result.append(f"{item}_{seen[item]}")
    return result

def dataframe_to_data_profile(table, context):
    # Refactor table to have the first row as header
    new_header = table.iloc[0] 
    table = table[1:] 
    table.columns = new_header 
    table = table.reset_index(drop=True)

    

    try:
        #test multilines header
        #dict['data_profile'] = generate_data_profile(table, context)
        return generate_data_profile(table, context)
        #print(generate_data_profile(table, context))

    except ValueError as e:
        #print(table_id)

        if "not 1-dimensional" in str(e):
            # Header on 2 lines detected
            header = [
                f"{str(a).strip()} {str(b).strip()}".strip()
                for a, b in zip(table.columns, table.iloc[0])
            ]

            #print("header:", header)

            table = table[1:] 
            table.columns = header
            table = table.reset_index(drop=True)
            #table = pd.DataFrame(table[2:], columns=header)

            #print(table.head())

            # normalizza le colonne
            table.columns = (
                table.columns
                .str.replace("'", "", regex=False)
                .str.replace(r"\s+", "_", regex=True)
                .str.lower()
            )

            table.columns = make_unique(table.columns.tolist())

            #dict['data_profile'] = generate_data_profile(table, context)
            return generate_data_profile(table, context)
            #print(generate_data_profile(table, context))
        else:
            raise


TEMPLATE_SEMANTIC = """
    {'Temporal': 
        {
            'isTemporal': Does this column contain temporal information? Yes or No,
            'resolution': If Yes, specify the resolution (Year, Month, Day, Hour, etc.).
        },
     'Spatial': {'isSpatial': Does this column contain spatial information? Yes or No,
                 'resolution': If Yes, specify the resolution (Country, State, City, Coordinates, etc.).},
     'Entity Type': What kind of entity does the column describe? (e.g., Person, Location, Organization, Product),
     'Domain-Specific Types': What domain is this column from (e.g., Financial, Healthcare, E-commerce, Climate, Demographic),
     'Function/Usage Context': How might the data be used (e.g., Aggregation Key, Ranking/Scoring, Interaction Data, Measurement).}
    """

RESPONSE_EXAMPLE_SEMANTIC = """
    {
    "Domain-Specific Types": "General",
    "Entity Type": "Temporal Entity",
    "Function/Usage Context": "Aggregation Key",
    "Spatial": {"isSpatial": false,
                "resolution": ""},
    "Temporal": {"isTemporal": true,
                "resolution": "Year"}
    }
    """


def generate_prompt_semantic_profile(data_profile):
    prompt = f"""
        You are a dataset semantic analyzer. Based on the data profile provided, classify the columns into multiple semantic types. 
        Please group the semantic types under the following categories: 
        'Temporal', 'Spatial', 'Entity Type', 'Data Format', 'Domain-Specific Types', 'Function/Usage Context'. 
        Following is the template {TEMPLATE_SEMANTIC}
        Please follow these rules:
        1. The output must be a valid JSON object that can be directly loaded by json.loads. Example response is {RESPONSE_EXAMPLE_SEMANTIC}
        2. All keys from the template must be present in the response.
        3. All keys and string values must be enclosed in double quotes.
        4. There must be no trailing commas.
        5. Use booleans (true/false) and numbers without quotes.
        6. Do not include any additional information or context in the response.
        7. If you are unsure about a specific category, you can leave it as an empty string.

        Data Profile: {data_profile}
        """
    return prompt

#calls openAI api  you can select prompt and model
#returns the answer
def call_openai_api(prompt, model):
    #setup OpenAI API
    load_dotenv()
    client = OpenAI()

    if model == 'o1-mini':
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
    else:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": prompt
            }
        ]

    completion = client.chat.completions.create(
    model=model,
    messages=messages
    )
    return completion.choices[0].message.content


RESPONSE_QUERIES_INSTRUCTIONS = """ 
{
  "queries": [
    {
      "query": "Select data ..."
    },
    {
      "query": "Find datasets ..."
    },
    {
      "query": "Show me data ..."
    }
  ]
}
"""

def generate_prompt_instructions(dataset_title, semantic_profile, final_profile_summary):
    PROMPT_INSTRUCTIONS = f"""
    You are the dataset owner of {dataset_title}. 
    Help the users find effectively your dataset in a DataSpace platform.
    The user is provided with a prompt interface, so it can ask natural language questions to find the dataset.
    The final users don't have access to the dataset content. 
    Provide the most probable questions a user would ask to find this dataset.

    The dataset semantic profile is the following: {semantic_profile} /n
    The data profile is the following: {final_profile_summary} /n   
    

    One example of a query could be: "Find a dataset with entries about cannabis strains and their effects"

    Generate 10 pseudo-queries to cover the entire dataset content and structure.
    Reason step by step:
    1. First understand the dataset content and structure.
    2. Formulate general queries to show the dataset content.
    3. Formulate precise queries to highlight specific findings or limitations.
    4. Formulate queries that consider interactions between different columns.
    5. Formulate queries that consider the dataset's temporal and spatial aspects.

    The output must be a valid JSON object that can be directly loaded by json.loads. It should be a list of 10 queries. An example response is: {RESPONSE_QUERIES_INSTRUCTIONS}
    """

    return PROMPT_INSTRUCTIONS

def generate_prompt_instructions_no_semantic_profile(dataset_title, final_profile_summary):
    PROMPT_INSTRUCTIONS = f"""
    You are the dataset owner of {dataset_title}. 
    Help the users find effectively your dataset in a DataSpace platform.
    The user is provided with a prompt interface, so it can ask natural language questions to find the dataset.
    The final users don't have access to the dataset content. 
    Provide the most probable questions a user would ask to find this dataset.

    The data profile is the following: {final_profile_summary} /n   
    

    One example of a query could be: "Find a dataset with entries about cannabis strains and their effects"

    Generate 10 pseudo-queries to cover the entire dataset content and structure.
    Reason step by step:
    1. First understand the dataset content and structure.
    2. Formulate general queries to show the dataset content.
    3. Formulate precise queries to highlight specific findings or limitations.
    4. Formulate queries that consider interactions between different columns.
    5. Formulate queries that consider the dataset's temporal and spatial aspects.

    The output must be a valid JSON object that can be directly loaded by json.loads. It should be a list of 10 queries. An example response is: {RESPONSE_QUERIES_INSTRUCTIONS}
    """

    return PROMPT_INSTRUCTIONS

def json_to_dict(list_Q):

    #print("Raw JSON response:", list_Q)

    list_Q = list_Q.replace("```json\n", "").replace("\n```", "")
    #list_Q = list_Q.replace("'", '"')

    queries_dict = json.loads(list_Q)
    return queries_dict

def create_embeddings_openai(texts):
    response = openai.embeddings.create(
        model="text-embedding-3-small",  
        input=texts
    )
    
    embeddings = [item.embedding for item in response.data]
    return embeddings

class OpenAIEmbeddingFunction:
    def __call__(self, input):
        embeddings = create_embeddings_openai(input)
        return embeddings
    
def get_summary(query, model):

    prompt_to_summary = f"""
    You are a dataset search assistant and you need to help users find relevant datasets.
    You are given a user query and you need to generate a background document that can help answer the query. 
    The background document should contain relevant information about the main topics of the query, including key concepts, definitions, and any other relevant information.

    For example, if the query is "Find dataset about climate change", the background document should include information about climate change, its causes, effects, and any other relevant information.

    Here is the user's query: {query}

    No additional text is needed, just the background document.
    """

    return call_openai_api(prompt_to_summary, model)#.choices[0].message.content

def decompose_query(query, summary, model):

    prompt_to_instructions = f"""
    Address the following query based on the background document provided.
    Use the background document to inform your response, but do not simply repeat information from it.
    Decompose the original query into 10 sub-queries that can help answer the main query.
    Each sub-query should be specific and focused, addressing a particular aspect of the main query.

    Query: {query}
    Background document: {summary}

    Output the 10 sub-queries you generate in a numbered list, no other text is required.
    """

    return call_openai_api(prompt_to_instructions, model)#.choices[0].message.content

def top_k_tuples(pairs, k):
    
    freq = Counter(pairs)
    most_common = [pair for pair, _ in freq.most_common(k)]
    return most_common

def get_subqueries_from_query(query, model, collection, k) -> List[Tuple]:

    result = []

    #get summary
    summary = get_summary(query, model)

    #decompose in subqueries
    subqueries = re.split(r'\d+\.\s*', decompose_query(query, summary, model).strip())
    subqueries = [d.strip() for d in subqueries if d.strip()]

    #get results from chromaDB, by default top 10 results for each subquery
    metadatas = collection.query(
    query_texts=subqueries,
    include=["metadatas"],
    n_results=100
    )

    for subquery_results_metadata in metadatas['metadatas']:
        for single_dataset_metadata in subquery_results_metadata:
            tuple_result = (single_dataset_metadata['database_id'], single_dataset_metadata['table_id'])
            result.append(tuple_result)

    return top_k_tuples(result, k)

def get_results_from_query_different_recall(query, model, collection) -> List[Tuple]:

    result = []

    #get summary
    summary = get_summary(query, model)

    #decompose in subqueries
    subqueries = re.split(r'\d+\.\s*', decompose_query(query, summary, model).strip())
    subqueries = [d.strip() for d in subqueries if d.strip()]

    #get results from chromaDB, by default top 10 results for each subquery
    metadatas = collection.query(
    query_texts=subqueries,
    include=["metadatas"],
    n_results=100
    )

    for subquery_results_metadata in metadatas['metadatas']:
        for single_dataset_metadata in subquery_results_metadata:
            tuple_result = (single_dataset_metadata['database_id'], single_dataset_metadata['table_id'])
            result.append(tuple_result)

    recall = {"1": top_k_tuples(result, 1),
              "3": top_k_tuples(result, 3),
              "5": top_k_tuples(result, 5),
              "10": top_k_tuples(result, 10)}

    return recall




