import datamart_profiler
import pandas as pd
import os
from ydata_profiling import ProfileReport
import datamart_profiler
import openai
from dotenv import load_dotenv
from openai import OpenAI
import json

#this function is used to generate the data profile from the dataset in CSV format
#then proceed to extract the information we need, the output will be text
def generate_data_profile(file_path):
    df = pd.read_csv(file_path)

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

    dataset_title = os.path.splitext(os.path.basename(file_path))[0]

    final_profile_summary = "The key data profile information for the dataset "+ dataset_title +" includes:\n" + '\n'.join(profile_summary)

    return final_profile_summary



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
    return completion

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


def generate_prompt(DATA_PROFILE, TEMPLATE, RESPONSE_EXAMPLE):
    prompt = f"""
        You are a dataset semantic analyzer. Based on the data profile provided, classify the columns into multiple semantic types. 
        Please group the semantic types under the following categories: 
        'Temporal', 'Spatial', 'Entity Type', 'Data Format', 'Domain-Specific Types', 'Function/Usage Context'. 
        Following is the template {TEMPLATE}
        Please follow these rules:
        1. The output must be a valid JSON object that can be directly loaded by json.loads. Example response is {RESPONSE_EXAMPLE}
        2. All keys from the template must be present in the response.
        3. All keys and string values must be enclosed in double quotes.
        4. There must be no trailing commas.
        5. Use booleans (true/false) and numbers without quotes.
        6. Do not include any additional information or context in the response.
        7. If you are unsure about a specific category, you can leave it as an empty string.

        Data Profile: {DATA_PROFILE}
        """
    return prompt

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
    You are the dataset owner of {dataset_title}. You need to provide instructions to the users on how to discover effectively the dataset in a DataSpace platform.
    The user is provided with a prompt interface, so it can ask naturakl language questions to find the dataset.
    The dataset contains the following semantic types: {semantic_profile} /n
    The data profile is the following: {final_profile_summary} /n   
    The final users don't have access to the dataset content. So, provide instructions (queries) that they could use to find this dataset.

    One example of a query could be: "Find a dataset with entries about cannabis strains and their effects"

    Generate as many queries as required to cover the entire dataset content and structure.
    Reason step by step:
    1. First understand the dataset content and structure.
    2. Formulate general queries to show the dataset content.
    3. Formulate precise queries to highlight specific findings or limitations.
    4. Formulate queries that consider interactions between different columns.
    5. Formulate queries that consider the dataset's temporal and spatial aspects.

    The output must be a valid JSON object that can be directly loaded by json.loads. It should be a list of queries. An example response is: {RESPONSE_QUERIES_INSTRUCTIONS}
    """

    return PROMPT_INSTRUCTIONS

def json_to_dict(list_Q):
    list_Q = list_Q.replace("```json\n", "").replace("\n```", "")

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
    