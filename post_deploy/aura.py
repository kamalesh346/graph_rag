from neo4j import GraphDatabase

URI = "neo4j+s://41c28f9d.databases.neo4j.io"
AUTH = ("41c28f9d", "JHxPwg9KeXF2OW5ZHeqZzFXokQNDKVlIjutAFzUtwdY")

def get_graph_summary(driver):
    # Get all node labels and counts
    node_counts, _, _ = driver.execute_query(
        "MATCH (n) RETURN labels(n)[0] AS label, count(n) AS count ORDER BY count DESC"
    )
    print("--- Node Summary ---")
    for record in node_counts:
        print(f"Label: {record['label']:<20} Count: {record['count']}")

    # Get all relationship types and counts
    rel_counts, _, _ = driver.execute_query(
        "MATCH ()-[r]->() RETURN type(r) AS relationship, count(r) AS count ORDER BY count DESC"
    )
    print("\n--- Relationship Summary ---")
    for record in rel_counts:
        print(f"Type:  {record['relationship']:<20} Count: {record['count']}")

def fetch_graph_triples(driver, limit=10):
    query = """
    MATCH (source)-[r]->(target)
    RETURN 
        labels(source)[0] AS source_type,
        coalesce(source.name, source.id, source.title, 'Unnamed') AS source_name,
        type(r) AS relationship,
        labels(target)[0] AS target_type,
        coalesce(target.name, target.id, target.title, 'Unnamed') AS target_name
    LIMIT $limit
    """
    records, _, _ = driver.execute_query(query, limit=limit)
    
    print(f"\n--- Top {limit} Relationships ---")
    for rec in records:
        print(f"({rec['source_type']}: {rec['source_name']}) -[:{rec['relationship']}]-> ({rec['target_type']}: {rec['target_name']})")

def search_nodes(driver, search_term):
    # Case-insensitive search on properties like name or text
    query = """
    MATCH (n)
    WHERE any(prop IN keys(n) WHERE toString(n[prop]) CONTAINS $term)
    RETURN labels(n) AS labels, properties(n) AS props
    LIMIT 5
    """
    records, _, _ = driver.execute_query(query, term=search_term)
    
    print(f"\n--- Search results for '{search_term}' ---")
    for record in records:
        print(f"Labels: {record['labels']} | Properties: {record['props']}")

def get_corresponding_section(driver, old_section_id):
    query = """
    MATCH (old:StatuteSection {id: $old_id})-[r:CORRESPONDS_TO]->(new:StatuteSection)
    RETURN old.id AS old_section, type(r) AS relation, new.id AS new_section
    """
    records, _, _ = driver.execute_query(query, old_id=old_section_id)
    for record in records:
        print(f"[{record['old_section']}] --{record['relation']}--> [{record['new_section']}]")
    
def retrieve_rag_context(driver, case_id):
    query = """
    MATCH (c:Case {id: $case_id})<-[:HAS_CHUNK]-(chunk:Chunk)
    OPTIONAL MATCH (chunk)-[:DISCUSSES]->(section:StatuteSection)
    RETURN chunk.id AS chunk_id, chunk.text AS text, collect(section.id) AS sections
    LIMIT 5
    """
    records, _, _ = driver.execute_query(query, case_id=case_id)
    for record in records:
        print(f"Chunk ID: {record['chunk_id']}")
        print(f"Discusses Sections: {record['sections']}")
        print(f"Snippet: {record['text'][:150]}...\n")

    
with GraphDatabase.driver(URI, auth=AUTH) as driver:
    # get_graph_summary(driver)
    # fetch_graph_triples(driver, limit=15)
    # search_nodes(driver, "Test")
    get_corresponding_section(driver, "IPC:302")
    retrieve_rag_context(driver, "2025 INSC 845")