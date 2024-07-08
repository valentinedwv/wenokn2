
import json
import logging
import re
import time

import chromadb
from decouple import config
import google.generativeai as genai
from fastapi import APIRouter
from groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


router = APIRouter(tags=["Utility"], prefix='/Utility')

logging.basicConfig(level=logging.ERROR)
client = chromadb.PersistentClient(path="/code/data/knowledge")
collection = client.get_collection(name="knowledge")

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")


# get relevant concepts for the request
def get_relevant_concepts(query_text):
    n_results=20
    results = collection.query(
        query_embeddings = [ embed_model.get_text_embedding(query_text) ],
        n_results=n_results
    )

    ids = results['ids'][0]
    documents = results['documents'][0]
    metadatas = results['metadatas'][0]
    distances = results['distances'][0]

    return documents, metadatas


# get description of the relevant concepts
def get_description(concepts, metadatas):
    description = ""
    for i in range(len(concepts)):
        if concepts[i]['is_relevant']:
            description = f"""{description}\n         
                           { metadatas[i]['def'] }
                           """
    return description


# create the sparql query request  
def sparql_request(query_text, description):

    example = '''
                SELECT ?buildingName ?buildingDescription ?buildingGeometry ?distanceInMeters
                WHERE {
                  # Get power station with name "dpq5d2851w52"
                  ?powerStation schema:additionalType "power";
                                schema:name "dpq5d2851w52";
                                schema:geo/geo:asWKT ?powerStationGeometry.
                
                  # Find buildings
                  ?building schema:additionalType "building";
                           schema:name ?buildingName;
                           schema:geo/geo:asWKT ?buildingGeometry.
                  
                  # Calculate distance between building and power station in meters 
                  BIND(geof:distance(?buildingGeometry, ?powerStationGeometry, unit:metre) AS ?distanceInMeters)
                  
                  # Filter buildings within 5 miles (1 mile = 1609.34 meters)
                  FILTER(?distanceInMeters <= 5 * 1609.34)
                } LIMIT 10
              '''

    example2 = '''
                SELECT ?buildingName ?buildingGeometry ?buildingFIPS { 
                    SERVICE <https://stko-kwg.geog.ucsb.edu/workbench/repositories/KWG> {
                        ?county a kwg-ont:AdministrativeRegion_3 ;
                            rdfs:label 'Ross' ;
                            geo:hasGeometry/geo:asWKT ?countyGeometry .
                    }
                    ?building schema:additionalType "building" ;
                                  schema:name ?buildingName ;
                                  schema:geo/geo:asWKT ?buildingGeometry ;
                                  schema:identifier ?buildingGeoid .
                    ?buildingGeoid schema:name "GEOID" ;
                                   schema:value ?buildingFIPS .
                    # Must include PREFIX geof: <http://www.opengis.net/def/function/geosparql/>
                    FILTER (geof:sfContains(?countyGeometry, ?buildingGeometry)) .
                }
                LIMIT 10
                '''

    return f"""
            The following are common used prefixes:
            
            PREFIX schema: <https://schema.org/>
            PREFIX geo: <http://www.opengis.net/ont/geosparql#>
            PREFIX geof: <http://www.opengis.net/def/function/geosparql/>
            PREFIX unit: <http://www.opengis.net/def/uom/OGC/1.0/>
            PREFIX kwg-ont: <http://stko-kwg.geog.ucsb.edu/lod/ontology/>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX time: <http://www.w3.org/2006/time#>
            PREFIX sosa: <http://www.w3.org/ns/sosa/>
            PREFIX ext: <http://rdf.useekm.com/ext#>   
            
            {description}
            
            [ Example 1: distance function ]
            
            The distance function is defined by geof:distance. The meter unit is defined as unit:metre. 1 mile = 1609.34 meters .
            The following is the syntax for using the distance function:
                BIND(geof:distance(?location1, ?location2, unit:metre) AS ?distance_m) .
            Please make sure ?location1 and ?location2 are both geometries when use the distance function.
            Here is an example how the distance function is used:
            
            { example }
            
            [ Example 2: sfContains function ]
            
            The sfContains function can be used to determine if a geometry is inside another geometry.
            For example, to find 10 buildings in Ross county, you can use this query:
            
            { example2 }
            
            [ Note 1: BIND ] 
            
            All BIND statements must be inside SELECT. BIND(something AS ?x) CAN ONLY APPEAR ONCE for 
            specific x! The second one is unnecessary.
            
            NEVER USE SELECT * WHERE {...}. Instead, enumerate all the attributes in the SELECT clause.
            Don't use any other knowledge more than the context. Don't assume anything.
            
            If no attributes of the required entity are mentioned, then return all the attributes of this entity
            in SELECT clause. Make sure SELECT variables always include a variable for geometries.
            
            Include all necessary prefix declarations without duplicates.
            
            Please create a SPARQL query using SELECT DISTINCT with PREFIX declarations for this request:
            
            {query_text}
            
            Must add your comments starting with "# LLM comment:" to the created query. 
            
            If the user asks for a specific number of entities, then use LIMIT with the number the user asks for.
            If the user asks for all buildings or all counties or all rivers or all dams, etc. then use "LIMIT 100".
            If the user's request didn't mention a number or "all",  then use "LIMIT 10".
            
            Do not use "ORDER BY" if the request did not ask.
            """

safe = [
    { "category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    { "category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    { "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    { "category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
]

model = genai.GenerativeModel('gemini-pro')
genai.configure(api_key=config["GOOGLE_KEY"])


def extract_code_blocks(text):
    # Define the pattern to match text between ```
    pattern = r'```([\s\S]*?)```'
    # Use re.findall to find all matches
    code_blocks = re.findall(pattern, text)
    return code_blocks


# Initialize a counter to keep track of the last used token
counter = 0
tokens = config("GROQ_KEYS")


@router.get("/wenokn_llama3", include_in_schema=True)
async def get_candidate_concepts(query_text: str):
    documents, metadatas = get_relevant_concepts(query_text)
    check_request = f"""
                    For this SPARQL request:
                        {query_text}
                    
                    We are required to use the following entities in our GraphDB without any additional knowledge:
                        { ', '.join(documents) }
                    What entities in this list are possible necessary to solve the request?
                    
                    Return your answer in JSON format as a list of the objects with two fields:
                    a string field "entity" and a boolean field "is_relevant".
                    """

    max_tries = 10
    current_try = 0
    while current_try < max_tries:
        try:
            response = model.generate_content(check_request, safety_settings=safe)
            break
        except Exception as e:
            print(e)
            time.sleep(1)
            current_try += 1

    response_text = response.text
    if response_text.startswith('```json') or response_text.startswith('```JSON'):
        json_part = response_text.split("\n", 1)[1].rsplit("\n", 1)[0]
        concepts = json.loads(json_part)
    else:
        concepts = json.loads(response_text)

    description = get_description(concepts, metadatas)
    request_to_sparql = sparql_request(query_text, description)
    # response = model.generate_content(request_to_sparql, safety_settings=safe)

    global counter
    token = tokens[counter % len(tokens)]
    counter += 1
    client = Groq(api_key=token)

    chat_completion = client.chat.completions.create(
        messages=[{ "role": "user",  "content": request_to_sparql }],
        model="llama3-70b-8192")

    result = chat_completion.choices[0].message.content
    result = extract_code_blocks(result)[0] 
    if result.startswith('sparql'):
        result = result[6:]

    return result

