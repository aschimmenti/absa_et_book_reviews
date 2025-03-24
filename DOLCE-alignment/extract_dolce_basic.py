#!/usr/bin/env python3
"""
Basic DOLCE Type Extraction Script

This script provides the core functionality to extract DOLCE types from text
using the AMR2FRED tool. It focuses on the basic extraction process without
handling edge cases.
"""

import json
import os
import re
import sys
import time
import traceback
try:
    from py_amr2fred import Amr2fred, Glossary
except ImportError:
    print("Error importing py_amr2fred. Make sure it's installed.")
    print("You can install it with: pip install py-amr2fred")
    sys.exit(1)
from rdflib import Graph, URIRef, RDF, RDFS

# Initialize amr2fred
try:
    amr2fred = Amr2fred()
    mode = Glossary.RdflibMode.TURTLE
except Exception as e:
    print(f"Error initializing Amr2fred: {str(e)}")
    print("Make sure py_amr2fred is properly installed and configured.")
    sys.exit(1)

# Regex patterns for datetime detection
YEAR_PATTERN = r'^(\d{4})$'  # Matches a 4-digit year like 1789
YEAR_RANGE_PATTERN = r'^(\d{4})-(\d{4})$'  # Matches year ranges like 1789-1799
DECADE_PATTERN = r'^(\d{4})s$'  # Matches decades like 1790s
CENTURY_PATTERN = r'^(\d{1,2})(st|nd|rd|th) century$'  # Matches centuries like 18th century

# DOLCE namespace
DUL_NS = "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#"

# Create directory for TTL files if it doesn't exist
os.makedirs("ttl_files", exist_ok=True)

def save_rdf_to_file(component, rdf_text):
    """
    Save RDF text to a file.
    
    Args:
        component (str): The component name
        rdf_text (str): The RDF text to save
        
    Returns:
        str: The filename where the RDF was saved
    """
    # Create a safe filename
    safe_component = re.sub(r'[^\w\-_\. ]', '_', component)
    safe_component = safe_component.replace(' ', '_')
    safe_component = safe_component.replace('/', '_')
    safe_component = safe_component.replace('\\', '_')
    safe_component = safe_component.replace(':', '_')
    safe_component = safe_component.replace('*', '_')
    safe_component = safe_component.replace('?', '_')
    safe_component = safe_component.replace('"', '_')
    safe_component = safe_component.replace('<', '_')
    safe_component = safe_component.replace('>', '_')
    safe_component = safe_component.replace('|', '_')
    
    # Create the directory if it doesn't exist
    ttl_dir = "ttl_files"
    os.makedirs(ttl_dir, exist_ok=True)
    
    # Create the filename
    rdf_filename = os.path.join(ttl_dir, f"rdf_output_{safe_component}.ttl")
    
    # Write the RDF text to the file
    with open(rdf_filename, "w", encoding="utf-8") as f:
        f.write(rdf_text)
    
    print(f"  RDF saved to: {rdf_filename}")
    return rdf_filename

def print_rdf_content(rdf_text):
    """
    Print RDF content in a readable format
    """
    print("\nRDF Content:")
    print("-" * 50)
    for i, line in enumerate(rdf_text.split('\n')):
        if line.strip():
            print(f"{i+1}: {line}")
    print("-" * 50)

def extract_dolce_types(rdf_text, is_template=False, print_rdf=False, local_cache=None):
    """
    Extract DOLCE types from RDF.
    
    Args:
        rdf_text (str): The RDF text to extract DOLCE types from
        is_template (bool): Whether the RDF was generated using a template
        print_rdf (bool): Whether to print the RDF content for review
        local_cache (dict, optional): A local cache dictionary to use instead of the global cache
        
    Returns:
        list: A list of tuples (fred_class, dolce_type)
    """
    # Print RDF content if requested
    if print_rdf:
        print_rdf_content(rdf_text)
    
    try:
        # Parse the RDF
        g = Graph()
        g.parse(data=rdf_text, format="turtle")
        
        # Template-related types to skip
        template_types = ["Dictionary", "be-located-at", "be-located-at-91"]
        
        # Person-related classes that should map to dul:Person
        person_classes = ["Woman", "Man", "Person", "Girl", "Boy", "Child", "Adult", "Human"]
        
        # Create a list to store the DOLCE types
        dolce_types = []
        
        print("\nExtracting DOLCE types...")
        
        # First, find all classes in the RDF
        all_classes_query = """
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX owl: <http://www.w3.org/2002/07/owl#>
            PREFIX dul: <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#>
            PREFIX fred: <http://www.ontologydesignpatterns.org/ont/fred/domain.owl#>
            
            SELECT DISTINCT ?class
            WHERE {
                {
                    ?class rdfs:subClassOf ?something .
                } UNION {
                    ?class a owl:Class .
                } UNION {
                    ?instance a ?class .
                }
            }
        """
        
        all_classes = set()
        results = g.query(all_classes_query)
        print(f"Found {len(results)} classes in the RDF")
        for row in results:
            class_uri = str(row[0])
            class_name = class_uri.split('/')[-1] if '/' in class_uri else class_uri
            if '#' in class_name:
                class_name = class_name.split('#')[-1]
            
            # Skip template-related types
            if is_template and any(t in class_name for t in template_types):
                print(f"  Skipping template-related class: {class_name}")
                continue
                
            all_classes.add((class_uri, class_name))
            print(f"  Found class: {class_name} ({class_uri})")
            
            # Check if this is a person-related class
            if any(person_class in class_name for person_class in person_classes):
                print(f"  Identified as person-related class: {class_name}")
                if (class_name, "Person") not in dolce_types:
                    dolce_types.append((class_name, "Person"))
        
        # For each class, find its DOLCE type
        for class_uri, class_name in all_classes:
            # Query for DOLCE types via rdfs:subClassOf
            query1 = f"""
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX dul: <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#>
                
                SELECT DISTINCT ?dolceType
                WHERE {{
                    <{class_uri}> rdfs:subClassOf ?dolceType .
                    FILTER(STRSTARTS(STR(?dolceType), STR(dul:)))
                }}
            """
            
            results1 = g.query(query1)
            for row in results1:
                dolce_uri = str(row[0])
                dolce_type = dolce_uri.split('#')[-1] if '#' in dolce_uri else dolce_uri
                
                if (class_name, dolce_type) not in dolce_types:
                    dolce_types.append((class_name, dolce_type))
                    print(f"  Found DOLCE type via subClassOf: {class_name} -> {dolce_type}")
            
            # Query for DOLCE types via owl:equivalentClass
            query2 = f"""
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX owl: <http://www.w3.org/2002/07/owl#>
                PREFIX dul: <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#>
                
                SELECT DISTINCT ?dolceType
                WHERE {{
                    <{class_uri}> owl:equivalentClass ?equiv .
                    ?equiv rdfs:subClassOf ?dolceType .
                    FILTER(STRSTARTS(STR(?dolceType), STR(dul:)))
                }}
            """
            
            results2 = g.query(query2)
            for row in results2:
                dolce_uri = str(row[0])
                dolce_type = dolce_uri.split('#')[-1] if '#' in dolce_uri else dolce_uri
                
                if (class_name, dolce_type) not in dolce_types:
                    dolce_types.append((class_name, dolce_type))
                    print(f"  Found DOLCE type via equivalentClass: {class_name} -> {dolce_type}")
        
        # Find all instances and their types
        instances_query = """
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX fred: <http://www.ontologydesignpatterns.org/ont/fred/domain.owl#>
            
            SELECT DISTINCT ?instance ?type
            WHERE {
                ?instance a ?type .
                FILTER(STRSTARTS(STR(?type), STR(fred:)))
            }
        """
        
        instances_results = g.query(instances_query)
        print(f"\nFound {len(instances_results)} instances in the RDF")
        
        # Dictionary to store instances and their qualities
        instance_qualities = {}
        
        for row in instances_results:
            instance_uri = str(row[0])
            type_uri = str(row[1])
            
            # Skip dictionary-related instances
            if is_template and any(t in instance_uri or t in type_uri for t in template_types):
                continue
            
            # Extract the type name
            type_name = type_uri.split('/')[-1] if '/' in type_uri else type_uri
            if '#' in type_name:
                type_name = type_name.split('#')[-1]
            
            print(f"  Found instance: {instance_uri} of type {type_name}")
            
            # Check if this is a person-related type
            if any(person_class in type_name for person_class in person_classes):
                print(f"  Identified as person-related instance: {type_name}")
                if (type_name, "Person") not in dolce_types:
                    dolce_types.append((type_name, "Person"))
            
            # Check for Wikidata or DBpedia sameAs links
            query_sameAs = f"""
                PREFIX owl: <http://www.w3.org/2002/07/owl#>
                
                SELECT DISTINCT ?sameAs
                WHERE {{
                    <{instance_uri}> owl:sameAs ?sameAs .
                }}
            """
            
            sameAs_results = g.query(query_sameAs)
            wikidata_found = False
            
            for sameAs_row in sameAs_results:
                sameAs_uri = str(sameAs_row[0])
                print(f"  Found sameAs link: {sameAs_uri}")
                
                if "wikidata.org/entity/" in sameAs_uri:
                    wikidata_found = True
                    # Extract entity ID from Wikidata URI (e.g., Q103495)
                    entity_id = sameAs_uri.split('/')[-1]
                    
                    # For Wikidata, we should use p31 (instance of) and/or P279 (subclass of)
                    print(f"  Found Wikidata entity: {entity_id}")
                    
                    # Map common Wikidata entities to DOLCE types
                    if entity_id.startswith('Q'):
                        # This is a very simplified mapping - in a real implementation,
                        # you would query the Wikidata API for p31/P279 values
                        if entity_id == "Q103495":  # World War
                            dolce_types.append((type_name, "Event"))
                            print(f"  Mapped Wikidata entity {entity_id} to DOLCE type: Event")
                        elif entity_id in ["Q5", "Q8436", "Q467"]:  # Human, Woman, Man
                            dolce_types.append((type_name, "Person"))
                            print(f"  Mapped Wikidata entity {entity_id} to DOLCE type: Person")
                        else:
                            # Default mapping based on instance type
                            if any(person_class in type_name for person_class in person_classes):
                                dolce_types.append((type_name, "Person"))
                                print(f"  Mapped Wikidata entity {entity_id} to DOLCE type: Person (based on type name)")
                            else:
                                dolce_types.append((type_name, f"WikidataEntity:{entity_id}"))
                                print(f"  Stored Wikidata entity reference: {entity_id}")
                
                elif "dbpedia.org/resource/" in sameAs_uri:
                    wikidata_found = True  # Treat DBpedia links similar to Wikidata
                    # Extract entity name from DBpedia URI
                    entity_name = sameAs_uri.split('/')[-1]
                    
                    print(f"  Found DBpedia entity: {entity_name}")
                    
                    # Map common DBpedia entities to DOLCE types
                    if "War" in entity_name:
                        dolce_types.append((type_name, "Event"))
                        print(f"  Mapped DBpedia entity {entity_name} to DOLCE type: Event")
                    elif any(person_term in entity_name for person_term in ["Person", "Woman", "Man", "Human"]):
                        dolce_types.append((type_name, "Person"))
                        print(f"  Mapped DBpedia entity {entity_name} to DOLCE type: Person")
                    else:
                        # Default mapping based on instance type
                        if any(person_class in type_name for person_class in person_classes):
                            dolce_types.append((type_name, "Person"))
                            print(f"  Mapped DBpedia entity {entity_name} to DOLCE type: Person (based on type name)")
                        else:
                            dolce_types.append((type_name, f"DBpediaEntity:{entity_name}"))
                            print(f"  Stored DBpedia entity reference: {entity_name}")
            
            # If no Wikidata/DBpedia link was found, try to get DOLCE type
            if not wikidata_found:
                # Query for DOLCE types of this instance's type
                query3 = f"""
                    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                    PREFIX dul: <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#>
                    
                    SELECT DISTINCT ?dolceType
                    WHERE {{
                        <{type_uri}> rdfs:subClassOf ?dolceType .
                        FILTER(STRSTARTS(STR(?dolceType), STR(dul:)))
                    }}
                """
                
                results3 = g.query(query3)
                dolce_found = False
                
                for row in results3:
                    dolce_uri = str(row[0])
                    dolce_type = dolce_uri.split('#')[-1] if '#' in dolce_uri else dolce_uri
                    
                    if (type_name, dolce_type) not in dolce_types:
                        dolce_types.append((type_name, dolce_type))
                        print(f"  Found DOLCE type for instance type: {type_name} -> {dolce_type}")
                        dolce_found = True
                
                # Store the instance for quality processing later
                instance_name = instance_uri.split('/')[-1] if '/' in instance_uri else instance_uri
                if '#' in instance_name:
                    instance_name = instance_name.split('#')[-1]
                
                instance_qualities[instance_uri] = {
                    'name': instance_name,
                    'type': type_name,
                    'qualities': [],
                    'dolce_found': dolce_found
                }
        
        # Find qualities for instances
        qualities_query = """
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX dul: <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#>
            
            SELECT DISTINCT ?instance ?quality
            WHERE {
                ?instance dul:hasQuality ?quality .
            }
        """
        
        qualities_results = g.query(qualities_query)
        print(f"\nFound {len(qualities_results)} instance-quality relationships")
        
        for row in qualities_results:
            instance_uri = str(row[0])
            quality_uri = str(row[1])
            
            # Skip if this is not an instance we're tracking
            if instance_uri not in instance_qualities:
                continue
                
            # Get the quality type
            query_quality_type = f"""
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                
                SELECT DISTINCT ?type
                WHERE {{
                    <{quality_uri}> a ?type .
                }}
            """
            
            quality_type_results = g.query(query_quality_type)
            for qt_row in quality_type_results:
                quality_type_uri = str(qt_row[0])
                
                # Skip template-related qualities
                if is_template and any(t in quality_type_uri for t in template_types):
                    continue
                
                # Extract the quality type name
                quality_type = quality_type_uri.split('/')[-1] if '/' in quality_type_uri else quality_type_uri
                if '#' in quality_type:
                    quality_type = quality_type.split('#')[-1]
                
                print(f"  Found quality for {instance_qualities[instance_uri]['name']}: {quality_type}")
                instance_qualities[instance_uri]['qualities'].append(quality_type)
                
                # Query for DOLCE types of this quality's type
                query4 = f"""
                    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                    PREFIX dul: <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#>
                    
                    SELECT DISTINCT ?dolceType
                    WHERE {{
                        <{quality_type_uri}> rdfs:subClassOf ?dolceType .
                        FILTER(STRSTARTS(STR(?dolceType), STR(dul:)))
                    }}
                """
                
                results4 = g.query(query4)
                for row in results4:
                    dolce_uri = str(row[0])
                    dolce_type = dolce_uri.split('#')[-1] if '#' in dolce_uri else dolce_uri
                    
                    if (quality_type, dolce_type) not in dolce_types:
                        dolce_types.append((quality_type, dolce_type))
                        print(f"  Found DOLCE type for quality: {quality_type} -> {dolce_type}")
        
        # Process compound terms (instances with qualities)
        for instance_uri, info in instance_qualities.items():
            if info['qualities']:
                # This is a compound term with qualities
                compound_term = f"{' '.join(info['qualities'])} {info['type']}"
                print(f"  Processing compound term: {compound_term}")
                
                # If the base type has a DOLCE mapping, use that
                base_type_found = False
                for class_name, dolce_type in dolce_types:
                    if class_name == info['type']:
                        if (compound_term, dolce_type) not in dolce_types:
                            dolce_types.append((compound_term, dolce_type))
                            print(f"  Assigned DOLCE type to compound term: {compound_term} -> {dolce_type}")
                            base_type_found = True
                
                # If no DOLCE mapping for base type, try to use the qualities
                if not base_type_found:
                    for quality in info['qualities']:
                        for class_name, dolce_type in dolce_types:
                            if class_name == quality:
                                if (compound_term, dolce_type) not in dolce_types:
                                    dolce_types.append((compound_term, dolce_type))
                                    print(f"  Assigned quality's DOLCE type to compound term: {compound_term} -> {dolce_type}")
                
                # Special case for Young Woman, Old Man, etc.
                if any(person_class in info['type'] for person_class in person_classes):
                    if (compound_term, "Person") not in dolce_types:
                        dolce_types.append((compound_term, "Person"))
                        print(f"  Assigned Person type to compound term: {compound_term} -> Person")
        
        # If we still have no DOLCE types but found person-related classes, add Person type
        if not dolce_types:
            for class_uri, class_name in all_classes:
                if any(person_class in class_name for person_class in person_classes):
                    dolce_types.append((class_name, "Person"))
                    print(f"  Assigned default Person type to: {class_name}")
        
        print(f"\nExtracted {len(dolce_types)} DOLCE types in total")
        return dolce_types
    
    except Exception as e:
        print(f"  Error extracting DOLCE types: {str(e)}")
        return []

def split_complex_value(value):
    """
    Split a complex value into components.
    Split by various patterns including '--', commas, and other delimiters.
    
    Args:
        value (str): The value to split
        
    Returns:
        list: A list of components
    """
    # First check for double hyphen pattern (most specific)
    if ' -- ' in value:
        return [part.strip() for part in value.split(' -- ')]
    
    # Check for comma-separated patterns with specific formats
    if re.search(r'(, fiction$|, general$|, history)', value.lower()):
        # Split by commas but preserve certain patterns
        parts = []
        # Handle patterns like "France, history, 1799-1914, fiction"
        # or "Fiction, action & adventure"
        if re.search(r', fiction$', value.lower()):
            main_part = re.sub(r', fiction$', '', value, flags=re.IGNORECASE)
            parts = [main_part.strip()]
        elif re.search(r', general$', value.lower()):
            main_part = re.sub(r', general$', '', value, flags=re.IGNORECASE)
            parts = [main_part.strip()]
        else:
            # Split by commas for other cases
            parts = [part.strip() for part in value.split(',')]
        
        return parts
    
    # Split by 'and' if present
    if ' and ' in value:
        return [part.strip() for part in value.split(' and ')]
    
    # Otherwise, return as a single item
    return [value]

def print_rdf_content(rdf_text):
    """
    Print RDF content in a readable format
    """
    print("\nRDF Content:")
    print("-" * 50)
    for i, line in enumerate(rdf_text.split('\n')):
        if line.strip():
            print(f"{i+1}: {line}")
    print("-" * 50)

def extract_fiction_genre(value):
    """
    Extract fiction-related genre from a topic.
    Returns a tuple of (cleaned_topic, fiction_genre)
    If no fiction genre is found, fiction_genre will be None.
    
    Args:
        value (str): The value to process
        
    Returns:
        tuple: (cleaned_topic, fiction_genre)
    """
    # Check for "FICTION / X" pattern
    if value.upper().startswith("FICTION /"):
        # Keep as is, don't extract
        return value, None
    
    # Check for "--fiction" pattern
    fiction_pattern = r'(.+?)--\s*(fiction|novel|story|tales)$'
    match = re.search(fiction_pattern, value, re.IGNORECASE)
    
    if match:
        # Extract the main topic and the fiction genre
        main_topic = match.group(1).strip()
        fiction_term = match.group(2).strip()
        
        # Return the cleaned topic and the fiction genre
        return main_topic, fiction_term
    
    return value, None

def clean_fiction_from_topic(value):
    """
    Remove fiction-related terms from a topic while keeping the meaningful part.
    For example: "Female friendship--fiction" -> "Female friendship"
    Also splits strings with "->" and takes the first part.
    
    Args:
        value (str): The value to clean
        
    Returns:
        str: The cleaned value
    """
    # Remove text in parentheses
    value = re.sub(r'\([^)]*\)', '', value).strip()
    
    # Split by "->" and take the first part
    if "->" in value:
        value = value.split("->")[0].strip()
    
    # Remove fiction-related suffixes
    fiction_suffixes = [
        r'--\s*fiction$',
        r'--\s*juvenile fiction$',
        r'--\s*novels$',
        r'--\s*stories$',
        r'--\s*tale$'
    ]
    
    for pattern in fiction_suffixes:
        value = re.sub(pattern, '', value, flags=re.IGNORECASE).strip()
    
    return value

def process_text(text, print_rdf=False, post_processing=True, local_cache=None):
    """
    Process a text and extract DOLCE types.
    
    Args:
        text (str): The text to process
        print_rdf (bool): Whether to print the RDF content for review
        post_processing (bool): Whether to use post_processing in AMR2FRED
        local_cache (dict, optional): A local cache dictionary to use instead of the global cache
        
    Returns:
        list: A list of tuples (fred_class, dolce_type)
    """
    print(f"\nProcessing text: '{text}'")
    
    try:
        # Check if this is a DBpedia entity that needs special handling
        if text.startswith("DBpediaEntity:"):
            print(f"  Detected DBpedia entity, processing with post_processing=False")
            # Force post_processing to False for DBpedia entities
            post_processing = False
            # Extract the entity name from the DBpediaEntity format
            entity_name = text.split("DBpediaEntity:")[1]
            text = entity_name
            print(f"  Extracted entity name: '{entity_name}'")
        
        # Check for fiction patterns first
        cleaned_text, fiction_genre = extract_fiction_genre(text)
        
        # If fiction genre was extracted, print it
        if fiction_genre:
            print(f"  Extracted fiction genre: '{fiction_genre}'")
            print(f"  Cleaned text: '{cleaned_text}'")
        
        # Split complex values
        components = split_complex_value(cleaned_text)
        all_dolce_types = []
        
        # Process each component
        for component in components:
            print(f"\nProcessing component: '{component}'")
            
            # Check if it's a short text (1-2 words)
            words = component.split()
            input_text = component
            is_template = False
            
            # For short inputs, use a template to get better RDF
            if len(words) <= 2:
                input_text = f"{component} is on the dictionary"
                is_template = True
                print(f"  Using template: '{input_text}'")
            
            # Use AMR2FRED to generate RDF
            print(f"  Generating RDF with AMR2FRED... (post_processing={post_processing})")
            rdf_text = amr2fred.translate(
                text=input_text,
                serialize=True,
                mode=mode,
                post_processing=post_processing
            )
            
            # Save the RDF to a file
            save_rdf_to_file(component, rdf_text)
            
            # Extract DOLCE types
            print("  Extracting DOLCE types...")
            dolce_types = extract_dolce_types(rdf_text, is_template=is_template, print_rdf=print_rdf, local_cache=local_cache)
            
            # Print results
            if dolce_types:
                print("  Found DOLCE types:")
                for fred_class, dolce_type in dolce_types:
                    print(f"    {fred_class}: dul:{dolce_type}")
                    
                # Add to all dolce types
                all_dolce_types.extend(dolce_types)
            else:
                print("  No DOLCE types found.")
        
        # If we extracted a fiction genre, process it as well
        if fiction_genre:
            print(f"\nProcessing fiction genre: '{fiction_genre}'")
            
            # Use template for fiction genre
            input_text = f"{fiction_genre} is on the dictionary"
            print(f"  Using template: '{input_text}'")
            
            # Generate RDF
            rdf_text = amr2fred.translate(
                text=input_text,
                serialize=True,
                mode=mode,
                post_processing=post_processing
            )
            
            # Save RDF
            save_rdf_to_file(fiction_genre, rdf_text)
            
            # Extract DOLCE types
            fiction_dolce_types = extract_dolce_types(rdf_text, is_template=True, print_rdf=print_rdf, local_cache=local_cache)
            
            # Print results
            if fiction_dolce_types:
                print("  Found DOLCE types for fiction genre:")
                for fred_class, dolce_type in fiction_dolce_types:
                    print(f"    {fred_class}: dul:{dolce_type}")
                
                # Add to all dolce types
                all_dolce_types.extend(fiction_dolce_types)
        
        return all_dolce_types
    
    except Exception as e:
        print(f"  Error processing text: {str(e)}")
        return []

def process_book_aspects(book_data, max_aspects_per_category=10, local_cache=None):
    """
    Process all aspects of a book.
    
    Args:
        book_data (dict): The book data to process
        max_aspects_per_category (int): Maximum number of aspects to process per category
        local_cache (dict, optional): A local cache dictionary to use instead of the global cache
        
    Returns:
        dict: The processed book data
    """
    # If no local cache is provided, create an empty one
    if local_cache is None:
        local_cache = {}
    
    # Create a copy of the book data
    processed_book = book_data.copy()
    
    # Check if the book has aspects
    if "aspects" not in processed_book:
        print("  No aspects found in book data")
        return processed_book
    
    # Process each aspect category
    for aspect_key, aspect_values in processed_book["aspects"].items():
        # Skip if the aspect values are not a list
        if not isinstance(aspect_values, list):
            continue
        
        print(f"\n  Processing aspect category: {aspect_key}")
        
        # Check if we should focus on this category
        is_topic = aspect_key.startswith("CONTENT#TOPIC")
        is_event = aspect_key.startswith("CONTENT#EVENT")
        
        if not (is_topic or is_event):
            print(f"  Skipping non-topic/event category: {aspect_key}")
            continue
        
        # Determine the type of the aspect
        aspect_type = None
        if ":" in aspect_key:
            aspect_type = aspect_key.split(":")[-1]
        
        # Process each aspect value
        processed_aspects = []
        
        # Limit the number of aspects to process
        limited_aspects = aspect_values[:max_aspects_per_category]
        print(f"  Processing {len(limited_aspects)}/{len(aspect_values)} aspects in this category")
        
        for aspect in limited_aspects:
            # Skip if the aspect is not a dictionary
            if not isinstance(aspect, dict):
                continue
            
            # Get the aspect value
            if "value" not in aspect:
                continue
            
            value = aspect["value"]
            
            # Skip if the value is empty
            if not value:
                continue
            
            print(f"\n  Processing aspect value: '{value}'")
            
            # Clean the value
            cleaned_value = clean_value(value)
            print(f"  Cleaned value: '{cleaned_value}'")
            
            # Skip if the topic should be skipped
            if should_skip_topic(cleaned_value, aspect_type):
                print(f"  Skipping topic: '{cleaned_value}'")
                processed_aspects.append(aspect)
                continue
            
            # Check if the value is a datetime
            is_date, date_type = is_datetime(cleaned_value)
            if is_date:
                print(f"  Identified as datetime: {cleaned_value} ({date_type})")
                aspect["superclass"] = "TimeInterval"
                processed_aspects.append(aspect)
                continue
            
            # Check if the value is about reading level
            if is_reading_level(cleaned_value):
                print(f"  Identified as reading level: {cleaned_value}")
                aspect["superclass"] = "InformationObject"
                processed_aspects.append(aspect)
                continue
            
            # Check if the value is audience-related
            if is_audience_related(cleaned_value):
                print(f"  Identified as audience-related: {cleaned_value}")
                aspect["superclass"] = "SocialAgent"
                processed_aspects.append(aspect)
                continue
            
            # For topics of type "subject", check if it's fiction-related
            if is_topic and aspect_type == "subject":
                # Extract fiction genre if present
                cleaned_value, fiction_genre = extract_fiction_genre(cleaned_value)
                if fiction_genre:
                    print(f"  Extracted fiction genre: {fiction_genre}")
                    aspect["fiction_genre"] = fiction_genre
                
                # Clean fiction-related terms from the topic
                cleaned_value = clean_fiction_from_topic(cleaned_value)
                print(f"  After fiction cleaning: '{cleaned_value}'")
            
            # Process the cleaned value
            try:
                # Use the cache to avoid redundant processing
                dolce_types = process_value(cleaned_value, local_cache=local_cache)
                
                # If we found DOLCE types, add them to the aspect
                if dolce_types:
                    # Get the first DOLCE type
                    first_dolce_type = dolce_types[0][1]
                    aspect["superclass"] = first_dolce_type
                    
                    # Add all DOLCE types to the aspect
                    aspect["dolce_types"] = [{"component": component, "type": dolce_type} for component, dolce_type in dolce_types]
                    
                    print(f"  Found DOLCE type: {first_dolce_type}")
                else:
                    print(f"  No DOLCE type found for: '{cleaned_value}'")
                    
                    # Try processing compound terms
                    compound_dolce_types = process_compound_term(cleaned_value)
                    if compound_dolce_types:
                        # Get the first DOLCE type
                        first_dolce_type = compound_dolce_types[0][1]
                        aspect["superclass"] = first_dolce_type
                        
                        # Add all DOLCE types to the aspect
                        aspect["dolce_types"] = [{"component": component, "type": dolce_type} for component, dolce_type in compound_dolce_types]
                        
                        print(f"  Found compound DOLCE type: {first_dolce_type}")
            except Exception as e:
                print(f"  Error processing value: {str(e)}")
                traceback.print_exc()
            
            # Add the processed aspect to the list
            processed_aspects.append(aspect)
        
        # Update the aspect values
        processed_book["aspects"][aspect_key] = processed_aspects
    
    return processed_book

def process_books(books_data, test_mode=False, local_cache=None):
    """
    Process a list of books and extract DOLCE types for each.
    
    Args:
        books_data (list): The list of books to process
        test_mode (bool): Whether to process only the first 3 books
        local_cache (dict, optional): A local cache dictionary to use instead of the global cache
        
    Returns:
        list: The processed books with DOLCE types
    """
    # If no local cache is provided, create an empty one
    if local_cache is None:
        local_cache = {}
    
    processed_books = []
    
    # Limit to first 3 entries if in test mode
    if test_mode:
        print("\n=== TEST MODE: Processing only the first 3 books ===\n")
        books_data = books_data[:3]
    
    total_books = len(books_data)
    
    for i, book_data in enumerate(books_data):
        try:
            print(f"\n\n=== Processing Book {i+1}/{total_books}: {book_data.get('title', 'Unknown')} ===")
            processed_book = process_book_aspects(book_data, local_cache=local_cache)
            processed_books.append(processed_book)
            
            # Save progress every 10 books in test mode, or every 100 books in normal mode
            save_interval = 10 if test_mode else 100
            if (i + 1) % save_interval == 0:
                output_filename = f"processed_books_{i+1}.json"
                print(f"\n=== Saving progress at {i+1}/{total_books} books to {output_filename} ===")
                with open(output_filename, "w", encoding="utf-8") as f:
                    json.dump(processed_books, f, indent=2)
        except Exception as e:
            print(f"Error processing book {book_data.get('title', 'Unknown')}: {str(e)}")
            print("Skipping this book and continuing with the next one...")
            continue
    
    return processed_books

def test_examples():
    """Test with some example values."""
    examples = [
        "social class",
        "pride",
        "marriage",
        "Revolution",
        "Civil War",
        "Eruption",
        "English language",
        "Young Woman",
        "Man-woman relationships",  # Hyphenated term
        "Female friendship--fiction",  # Double hyphen with fiction
        "FICTION / Historical",  # Fiction pattern
        "World War and Peace"  # Complex with 'and'
    ]
    
    results = {}
    for example in examples:
        print("\n" + "="*50)
        dolce_types = process_text(example, print_rdf=True, local_cache={})  # Enable RDF printing
        results[example] = dolce_types
        
    print("\n\n=== Summary of Results ===")
    for example, types in results.items():
        formatted_types = [f"{cls}: dul:{typ}" for cls, typ in types]
        print(f"{example}: {', '.join(formatted_types) if formatted_types else 'No DOLCE types found'}")

def process_compound_term(component):
    """
    Process a compound term by trying to extract DOLCE types for the individual words.
    
    Args:
        component (str): The compound term to process
        
    Returns:
        list: A list of tuples (term, dolce_type)
    """
    print(f"Processing compound term: {component}")
    
    # Check if this is a Wikidata-linked term (like "World War")
    if "war" in component.lower():
        print(f"  Identified as a war-related term: {component}")
        return [(component, "Event")]
    
    # Check if this is a compound term with qualities (like "Young Woman")
    if "woman" in component.lower() and ("young" in component.lower() or "old" in component.lower()):
        print(f"  Identified as a person with qualities: {component}")
        return [(component, "Person")]
    
    # Check for other common patterns
    if "man" in component.lower():
        return [(component, "Person")]
    if "history" in component.lower() or "historical" in component.lower():
        return [(component, "InformationObject")]
    if "relationship" in component.lower():
        return [(component, "Relation")]
    if "fiction" in component.lower():
        return [(component, "InformationObject")]
    
    # For other multi-word terms, try to split and process individually
    words = component.split()
    if len(words) > 1:
        print(f"  Splitting compound term: {component}")
        # This is a placeholder - in a real implementation, we would recursively
        # process each word and combine the results
        return [(component, "Abstract")]
    
    # Default fallback
    return [(component, "Abstract")]

def process_value(value, local_cache=None):
    """
    Process a value and extract DOLCE types.
    
    Args:
        value (str): The value to process
        local_cache (dict, optional): A local cache dictionary to use instead of the global cache
        
    Returns:
        list: A list of tuples (component, dolce_type)
    """
    # Skip if the value is empty
    if not value:
        return []
    
    # Split the value into components
    components = split_complex_value(value)
    
    # Process each component
    all_dolce_types = []
    for component in components:
        # Skip if the component is empty
        if not component:
            continue
        
        print(f"  Processing component: '{component}'")
        
        try:
            # Use the cache to avoid redundant processing
            dolce_types = process_text_with_cache(component, local_cache=local_cache)
            
            # If we found DOLCE types, add them to the list
            if dolce_types:
                all_dolce_types.extend([(component, dolce_type) for _, dolce_type in dolce_types])
                print(f"  Found DOLCE types for component: {dolce_types}")
            else:
                print(f"  No DOLCE types found for component: '{component}'")
        except Exception as e:
            print(f"  Error processing component: {str(e)}")
    
    return all_dolce_types

def should_skip_topic(value, aspect_type=None):
    """
    Check if a topic should be skipped based on the following criteria:
    1. Contains camel case
    2. Contains underscores
    3. Is entirely fiction/literature-related
    4. Contains "translation" when type is "subject"
    
    Args:
        value (str): The value to check
        aspect_type (str, optional): The type of the aspect
        
    Returns:
        bool: True if the topic should be skipped, False otherwise
    """
    # Check for "FICTION / X" pattern - these should be moved to genre, not skipped
    if re.match(r'^FICTION\s*/\s*\w+', value, re.IGNORECASE):
        return False
    
    # Check for camel case (uppercase letter after lowercase)
    if re.search(r'[a-z][A-Z]', value):
        return True
    
    # Check for underscores
    if '_' in value:
        return True
    
    # Check for translation-related terms
    if aspect_type == "subject" and re.search(r'translation', value.lower()):
        return True
    
    # Check for fiction/literature-related terms
    fiction_terms = [
        r'^fiction$',
        r'^literature$',
        r'^novel$',
        r'^novels$',
        r'^story$',
        r'^stories$',
        r'^coming of age$',
        r'^bildungsroman$'
    ]
    
    for pattern in fiction_terms:
        if re.match(pattern, value.lower()):
            return True
    
    return False

def is_datetime(text):
    """
    Check if the text is a datetime using regex patterns.
    
    Args:
        text (str): The text to check
        
    Returns:
        tuple: (is_date, date_type) - Whether the text is a date and the date type
    """
    text = text.strip()
    
    # Check for year
    if re.match(YEAR_PATTERN, text):
        return True, "TimeInterval"
    
    # Check for year range
    if re.match(YEAR_RANGE_PATTERN, text):
        return True, "TimeInterval"
    
    # Check for decade
    if re.match(DECADE_PATTERN, text):
        return True, "TimeInterval"
    
    # Check for century
    if re.match(CENTURY_PATTERN, text, re.IGNORECASE):
        return True, "TimeInterval"
    
    return False, None

def clean_value(value):
    """
    Clean the value by removing text in parentheses and other unwanted characters.
    
    Args:
        value (str): The value to clean
        
    Returns:
        str: The cleaned value
    """
    # Remove text in parentheses
    cleaned = re.sub(r'\([^)]*\)', '', value)
    # Remove extra whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def is_reading_level(value):
    """
    Check if a topic is about reading level or grade.
    
    Args:
        value (str): The value to check
        
    Returns:
        bool: True if the topic is about reading level, False otherwise
    """
    reading_patterns = [
        r'reading level',
        r'grade \d+',
        r'children\'s (stories|books)',
        r'juvenile (fiction|literature)'
    ]
    
    for pattern in reading_patterns:
        if re.search(pattern, value.lower()):
            return True
    
    return False

def is_audience_related(value):
    """
    Check if a topic is related to audience demographics.
    
    Args:
        value (str): The value to check
        
    Returns:
        bool: True if the topic is audience-related, False otherwise
    """
    audience_patterns = [
        r'children\'s',
        r'juvenile',
        r'young adult',
        r'teen',
        r'adult',
        r'for kids',
        r'for children',
        r'for young adults'
    ]
    
    for pattern in audience_patterns:
        if re.search(pattern, value.lower()):
            return True
    
    return False

def process_text_with_cache(text, cache_key=None, post_processing=True, local_cache=None):
    """
    Process text with caching to avoid redundant AMR2FRED calls.
    
    Args:
        text (str): The text to process
        cache_key (str, optional): The cache key to use. If None, the text itself is used as the key.
        post_processing (bool, optional): Whether to use post_processing in AMR2FRED
        local_cache (dict, optional): A local cache dictionary to use instead of the global cache
        
    Returns:
        list: A list of tuples (fred_class, dolce_type)
    """
    # Use the text as the cache key if none is provided
    if cache_key is None:
        cache_key = text
    
    # Create a unique cache key that includes the post_processing flag
    full_cache_key = f"{cache_key}_{post_processing}"
    
    # Use the provided local cache if available
    cache = local_cache if local_cache is not None else {}
    
    # Check if we have this text in the cache
    if full_cache_key in cache:
        print(f"  Using cached result for: '{cache_key}' (post_processing={post_processing})")
        return cache[full_cache_key]
    
    # Process the text with the specified post_processing flag
    result = process_text(text, post_processing=post_processing, local_cache=local_cache)
    
    # Cache the result
    cache[full_cache_key] = result
    
    return result

def main():
    """Main function to run the script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract DOLCE types from text using AMR2FRED")
    parser.add_argument("--test", action="store_true", help="Run test examples")
    parser.add_argument("--text", type=str, help="Process a specific text")
    parser.add_argument("--print-rdf", action="store_true", help="Print RDF content for review")
    parser.add_argument("--process-books", action="store_true", help="Process books from book_aspects_filtered.json")
    parser.add_argument("--test-books", action="store_true", help="Process only the first 3 books")
    
    args = parser.parse_args()
    
    if args.test:
        test_examples()
    elif args.text:
        process_text(args.text, print_rdf=args.print_rdf)
    elif args.process_books or args.test_books:
        # Load the book aspects
        input_file = "book_aspects_filtered.json"
        print(f"Loading book aspects from {input_file}...")
        
        with open(input_file, "r", encoding="utf-8") as f:
            books_data = json.load(f)
        
        print(f"Loaded {len(books_data)} books")
        
        # Process the books
        processed_books = process_books(books_data, test_mode=args.test_books)
        
        # Save the processed books
        output_file = "processed_books.json"
        print(f"\nSaving processed books to {output_file}...")
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(processed_books, f, indent=2)
        
        print(f"Saved {len(processed_books)} processed books to {output_file}")
    else:
        print("Please specify either --test to run test examples, --text to process a specific text, or --process-books to process books from book_aspects_filtered.json")

if __name__ == "__main__":
    main()
