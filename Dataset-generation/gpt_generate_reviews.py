import os
import json
import time
import random
from openai import OpenAI
import backoff
import multiprocessing
from functools import partial

# Create output directory if it doesn't exist
os.makedirs("generated_reviews", exist_ok=True)

# Function to create an OpenAI client
def create_openai_client():
    return OpenAI()

@backoff.on_exception(
    backoff.expo,
    Exception,  # Catch all exceptions that might be related to API or connection issues
    max_tries=10,  # Maximum number of retries
    max_time=600,  # Maximum total time to retry (10 minutes)
    on_backoff=lambda details: print(f"Backing off {details['wait']:0.1f} seconds after {details['tries']} tries")
)
def generate_review(client, system_prompt, user_prompt, example):
    """Generate a review using the OpenAI API with exponential backoff for error handling."""
    try:
        response = client.responses.create(
            model="gpt-4o-mini",
            input=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": system_prompt
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": user_prompt + "\n\n" + example
                        }
                    ]
                }
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "generate_review_aspect_analysis",
                    "schema": {
                        "type": "object",
                        "required": [
                            "review_text",
                            "aspects"
                        ],
                        "properties": {
                            "aspects": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "required": [
                                        "aspect_span",
                                        "input_category",
                                        "input_sentiment",
                                        "input_aspect"
                                    ],
                                    "properties": {
                                        "aspect_span": {
                                            "type": "string",
                                            "description": "The exact text span from the review that mentions the aspect."
                                        },
                                        "input_category": {
                                            "type": "string",
                                            "description": "The category of the aspect as provided in the input prompt."
                                        },
                                        "input_sentiment": {
                                            "enum": [
                                                "positive",
                                                "neutral",
                                                "negative"
                                            ],
                                            "type": "string",
                                            "description": "The sentiment associated with this aspect as provided in the input prompt."
                                        },
                                        "input_aspect": {
                                            "type": "string",
                                            "description": "The original aspect term as provided in the input prompt."
                                        }
                                    },
                                    "additionalProperties": False
                                },
                                "description": "Detailed analysis of aspects within the review"
                            },
                            "review_text": {
                                "type": "string",
                                "description": "The complete text of the new book review. Depending on what you wrote, give me back the aspects in the section below. They should match the aspects I gave you in the prompt."
                            }
                        },
                        "additionalProperties": False
                    },
                    "strict": True
                }
            },
            reasoning={},
            tools=[],
            temperature=0,
            max_output_tokens=2048,
            top_p=1,
            stream=False,  # Changed to False for simplicity in handling response
            store=True
        )
        
        # Correctly extract text from the response structure
        content_parts = response.output[0].content
        
        # Find the text part containing the JSON
        json_text = None
        for part in content_parts:
            if part.type == "output_text":
                json_text = part.text
                break
        
        if json_text is None:
            raise ValueError("No JSON content found in response")
            
        # Try to parse the JSON response with retry logic
        try:
            return json.loads(json_text)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Problematic JSON: {json_text[:100]}...")  # Print first 100 chars for debugging
            
            # If we can't parse it, but we don't want to fail completely,
            # return a minimal valid response and continue with other files
            return {
                "review_text": "Error parsing model response. Please retry this prompt.",
                "aspects": []
            }
        
    except Exception as e:
        print(f"Error generating review: {e}")
        raise  # Re-raise the exception to let backoff handle it

def process_prompt_file(file_path):
    """Process a single prompt file and generate a review."""
    try:
        # Create an OpenAI client for this process
        client = create_openai_client()
        
        # Read the prompt file
        with open(file_path, 'r', encoding='utf-8') as f:
            prompt_data = json.load(f)
        
        # Extract necessary information
        system_prompt = prompt_data.get("system_prompt", "")
        original_user_prompt = prompt_data.get("user_prompt", "")
        example = prompt_data.get("example", "")
        
        # Extract the original aspects from the prompt data
        original_aspects = prompt_data.get("aspects", [])
        
        # Modify the user prompt to ensure proper aspect values are used
        # Extract the book information and plot from the user prompt
        book_info_lines = []
        aspect_lines = []
        other_lines = []
        
        in_aspect_section = False
        for line in original_user_prompt.split('\n'):
            if line.startswith('Please create a new'):
                in_aspect_section = True
                other_lines.append(line)
            elif in_aspect_section and line.strip() and line[0].isdigit() and '. ' in line:
                # This is an aspect line, parse and reformat it
                aspect_lines.append(line)
            elif not in_aspect_section:
                book_info_lines.append(line)
            else:
                other_lines.append(line)
        
        # Reconstruct the user prompt
        modified_user_prompt = '\n'.join(book_info_lines)
        if aspect_lines:
            modified_user_prompt += '\n\n' + '\n'.join(aspect_lines)
        if other_lines:
            modified_user_prompt += '\n\n' + '\n'.join(other_lines)
        
        # Generate review with retry logic (max 3 attempts)
        print(f"Generating review for {os.path.basename(file_path)}...")
        
        max_retries = 3
        retry_count = 0
        review_data = None
        
        while retry_count < max_retries:
            try:
                review_data = generate_review(client, system_prompt, modified_user_prompt, example)
                
                # Check if we got an error response
                if (isinstance(review_data, dict) and 
                    review_data.get("review_text") == "Error parsing model response. Please retry this prompt." and
                    not review_data.get("aspects")):
                    
                    print(f"Received error response, retrying ({retry_count + 1}/{max_retries})...")
                    retry_count += 1
                    # Add a slightly longer wait time between retries
                    time.sleep(random.uniform(1.0, 5.0))
                else:
                    # We got a valid response, break the loop
                    break
                    
            except Exception as e:
                print(f"Error during review generation: {e}, retrying ({retry_count + 1}/{max_retries})...")
                retry_count += 1
                # Add a slightly longer wait time between retries
                time.sleep(random.uniform(1.0, 5.0))
                
            # If we've reached max retries, break out
            if retry_count >= max_retries:
                print(f"Maximum retry attempts ({max_retries}) reached.")
                break
        
        # If we still don't have a valid review after all retries, use the last response or create a fallback
        if review_data is None:
            review_data = {
                "review_text": "Failed to generate review after multiple attempts.",
                "aspects": []
            }
        
        # Validate and ensure the review data has the correct format
        if "aspects" in review_data:
            # Check if we need to transform old format to new format
            if review_data["aspects"] and "aspect" in review_data["aspects"][0]:
                # Transform from old format to new format
                transformed_aspects = []
                for aspect_item in review_data["aspects"]:
                    transformed_aspect = {
                        "aspect_span": aspect_item.get("aspect", ""),
                        "input_category": aspect_item.get("category", ""),
                        "input_sentiment": aspect_item.get("sentiment", "neutral"),
                        "input_aspect": aspect_item.get("aspect", "")
                    }
                    transformed_aspects.append(transformed_aspect)
                review_data["aspects"] = transformed_aspects
            
            # Map the generated aspects to the original aspects from the prompt
            # This ensures we maintain the correct aspect information
            if original_aspects:
                # Create a mapping of original aspect values to their full details
                aspect_mapping = {}
                for aspect in original_aspects:
                    aspect_value = aspect.get("aspect", "")
                    if aspect_value:
                        aspect_mapping[aspect_value.lower()] = aspect
                
                # Try to match each generated aspect with an original aspect
                for aspect in review_data["aspects"]:
                    aspect_span = aspect.get("aspect_span", "").lower()
                    input_aspect = aspect.get("input_aspect", "").lower()
                    
                    # Try to find a match in the original aspects
                    matched_aspect = None
                    
                    # First try exact match with input_aspect
                    if input_aspect in aspect_mapping:
                        matched_aspect = aspect_mapping[input_aspect]
                    else:
                        # Try to find partial matches
                        for orig_aspect, orig_data in aspect_mapping.items():
                            if orig_aspect in aspect_span or aspect_span in orig_aspect:
                                matched_aspect = orig_data
                                break
                    
                    # If we found a match, update the aspect with the original information
                    if matched_aspect:
                        aspect["input_category"] = matched_aspect.get("category", aspect.get("input_category", ""))
                        aspect["input_sentiment"] = matched_aspect.get("sentiment", aspect.get("input_sentiment", ""))
                        aspect["input_aspect"] = matched_aspect.get("aspect", aspect.get("input_aspect", ""))
            
            # Ensure all required fields are present in each aspect
            for aspect in review_data["aspects"]:
                if "aspect_span" not in aspect:
                    aspect["aspect_span"] = aspect.get("input_aspect", "")
                if "input_category" not in aspect:
                    aspect["input_category"] = ""
                if "input_sentiment" not in aspect:
                    aspect["input_sentiment"] = "neutral"
                if "input_aspect" not in aspect:
                    aspect["input_aspect"] = aspect.get("aspect_span", "")
        
        # Prepare output filename
        base_name = os.path.basename(file_path)
        output_name = base_name.replace("prompt_", "review_")
        output_path = os.path.join("generated_reviews", output_name)
        
        # Add the original prompt filename to the review data
        review_data["source_prompt_file"] = base_name
        
        # Save review data
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(review_data, f, ensure_ascii=False, indent=4)
        
        print(f"Review saved to {output_path}")
        
        # Add a random wait time between calls (up to 3 seconds)
        wait_time = random.uniform(0.5, 3.0)
        print(f"Waiting {wait_time:.2f} seconds before next call...")
        time.sleep(wait_time)
        
        return True
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    """Process all prompt files in the generated_prompts folder."""
    prompt_dir = "generated_prompts"
    num_workers = 4  # Number of worker processes
    
    # Get all JSON files from the generated_prompts directory
    prompt_files = [os.path.join(prompt_dir, f) for f in os.listdir(prompt_dir) 
                   if f.endswith('.json') and os.path.isfile(os.path.join(prompt_dir, f))]
    
    print(f"Found {len(prompt_files)} prompt files to process.")
    
    # Sort files by the numeric part of the filename
    def extract_number(filename):
        # Extract the number after "prompt_" and before "_"
        # Example: prompt_42_OL123456.json -> 42
        try:
            base_name = os.path.basename(filename)
            # Remove "prompt_" prefix
            if base_name.startswith("prompt_"):
                base_name = base_name[len("prompt_"):]
            # Extract number until the next underscore
            number_str = ""
            for char in base_name:
                if char.isdigit():
                    number_str += char
                else:
                    if char == "_":
                        break
                    else:
                        # If we encounter a non-digit, non-underscore character,
                        # we're not in the number part anymore
                        break
            return int(number_str) if number_str else float('inf')
        except Exception as e:
            print(f"Error extracting number from {filename}: {e}")
            return float('inf')  # Put files with parsing errors at the end
    
    # Sort the files based on the extracted number
    prompt_files.sort(key=extract_number)
    
    print("Processing files in numerical order...")
    
    # Create a pool of worker processes
    with multiprocessing.Pool(processes=num_workers) as pool:
        # Process each file in parallel
        results = pool.map(process_prompt_file, prompt_files)
        
        # Close the pool
        pool.close()
        pool.join()
        
        # Print summary
        successful = sum(1 for r in results if r)
        print(f"Processing complete. Successfully processed {successful} out of {len(prompt_files)} files.")

if __name__ == "__main__":
    main()