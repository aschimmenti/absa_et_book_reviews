import json
import random
import os
import csv
import numpy as np
from typing import Dict, List, Any
from collections import defaultdict

def load_json_data(file_path: str) -> List[Dict]:
    """Load JSON data from a file using UTF-8 encoding."""
    print(f"Loading JSON data from {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            print(f"Successfully loaded {len(data)} items from {file_path}")
            return data
    except Exception as e:
        print(f"Error loading JSON file {file_path}: {e}")
        return []

def load_tsv_reviews(file_path: str, max_reviews=5000, is_amazon=False):
    """Load reviews from a TSV file."""
    reviews = []
    try:
        print(f"Opening TSV file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as file:
            # Read header to determine column indices
            header = next(file).strip().split('\t')
            
            # Find indices for review text and ID
            text_index = -1  # Default to last column for review text
            id_index = -1
            
            if is_amazon:
                # For Amazon, find 'asin' column
                for i, col in enumerate(header):
                    if col.lower() == 'asin':
                        id_index = i
                        break
            else:
                # For Goodreads, find 'review_id' column
                for i, col in enumerate(header):
                    if col.lower() == 'review_id':
                        id_index = i
                        break
            
            reader = csv.reader(file, delimiter='\t')
            for i, row in enumerate(reader):
                if i >= max_reviews:
                    break
                if len(row) > 0:
                    # The last column contains the review text
                    review_text = row[-1]
                    
                    # Get review ID if available
                    review_id = "unknown_id"
                    if id_index >= 0 and id_index < len(row):
                        review_id = row[id_index]
                    
                    reviews.append({"text": review_text, "id": review_id})
                if i % 1000 == 0 and i > 0:
                    print(f"Processed {i} rows from {file_path}")
        print(f"Successfully loaded {len(reviews)} reviews from {file_path}")
    except Exception as e:
        print(f"Error loading TSV file {file_path}: {e}")
    return reviews

def create_book_descriptions_map(qualified_books: List[Dict]) -> Dict[str, Dict]:
    """
    Create a mapping from book IDs to their information from qualified_books.
    
    Args:
        qualified_books: List of qualified books data
    
    Returns:
        Dictionary mapping book keys to book information including title and plot
    """
    book_info = {}
    for book in qualified_books:
        key = book.get('key', '')
        info = {
            "title": book.get('title', ''),
            "plot": ""
        }
        
        # Extract description/plot from the data field if it exists
        if 'data' in book and isinstance(book['data'], dict):
            info["plot"] = book['data'].get('description', '')
        
        # Store the info with the key
        if key:
            # Store the key directly without any prefix
            book_info[key] = info
    
    print(f"Created book info map for {len(book_info)} books")
    return book_info

def assign_sentiment() -> str:
    """
    Randomly assign a sentiment with a distribution favoring positive and negative over neutral.
    
    Returns:
        A sentiment string: 'positive', 'negative', or 'neutral'
    """
    # Distribution: 45% positive, 40% negative, 15% neutral
    rand = random.random()
    if rand < 0.45:
        return "positive"
    elif rand < 0.85:  # 0.45 + 0.40
        return "negative"
    else:
        return "neutral"

def sample_aspects_for_review(book: Dict, num_aspects: int) -> List[Dict]:
    """
    Sample aspects from a book for a review and assign sentiments.
    
    Args:
        book: Book data containing aspects
        num_aspects: Number of aspects to sample
    
    Returns:
        List of sampled aspects with assigned sentiments
    """
    all_aspects = []
    
    # Group aspects by category
    category_aspects = defaultdict(list)
    
    # Process each category of aspects
    for category, aspects_list in book.get('aspects', {}).items():
        for aspect in aspects_list:
            category_aspects[category].append(aspect)
    
    # Sample one aspect per category if available, up to num_aspects
    categories = list(category_aspects.keys())
    random.shuffle(categories)
    
    # Limit to num_aspects categories
    selected_categories = categories[:num_aspects]
    
    # Sample one aspect from each selected category
    for category in selected_categories:
        if category_aspects[category]:
            aspect = random.choice(category_aspects[category])
            # Add category information to the aspect
            aspect_with_category = aspect.copy()
            aspect_with_category['category'] = category
            # Assign a sentiment to the aspect
            aspect_with_category['sentiment'] = assign_sentiment()
            all_aspects.append(aspect_with_category)
    
    return all_aspects

def create_system_prompt() -> str:
    """
    Create the system prompt with instructions for the task.
    
    Returns:
        System prompt string
    """
    return """You are an expert in creating synthetic book reviews dataset for Aspect based sentiment analysis (ABSA). Your task is to generate a new, realistic book review that incorporates the provided aspects of the book.

Follow these guidelines:
1. Use the provided book information (title, plot) to understand the context.
2. Incorporate the specified aspects naturally into the review by using the input aspects and their categorization as a guardrail on how to refer it into the text. 

This is mandatory, and you will be evaluated on this aspect. Treat the given review as a template: mantain its register (serious if serious, funny if funny). Remember to sound natural! 
3. Use the given review as inspiration to sound as authentic as a real reader.
4. Use natural language and a conversational tone similar to the provided example review. Do not add aspects or categories that are not in the given set of aspects.  
6. The review should be around 150 words in length, no longer than the given review to use. 
7. This will be used as a train dataset, so make sure to be absolutely precise in the way the annotation compares to the text. 


\n\n\nReturn the review in the given JSON schema, using the 'review' key for the spinned text, and the other keys as the annotation. """

def create_example() -> str:
    """
    Create an example of how to perform the task.
    
    Returns:
        Example string
    """
    return """
THE FOLLOWING IS AN EXAMPLE. NEVER USE THE ASPECTS OR THE CONTENT OF THIS EXAMPLE IN YOUR OUTPUT.
Original Review: "I couldn't put this book down. Olga Tokarczuk's style is engaging and the plot kept me guessing. Some of the characters felt a bit flat though. The writing is ridicously well crafted."

Book: "The Silent Patient" by Alex Michaelides
Aspects to incorporate:
1. Aspect: "Stream of consciousness". Sentiment: positive. Category: CONTENT#STYLE
2. Aspect: "Romanticism". Sentiment: positive. Category: CONTENT#MOVEMENT
3. Aspect: "Alicia Berenson". Sentiment: positive. Category: CONTENT#CHARACTER
4. Aspect: "Democracy". Sentiment: negative. Category: CONTENT#TOPIC
5. Aspect: "Mental illness". Sentiment: positive. Category: CONTENT#TOPIC
6. Aspect: "Young adult". Sentiment: neutral. Category: CONTENT#AUDIENCE

Generated JSON Schema: 

{
    "review_text": "I couldn't put this book down. The author's style is so similar to Joyce's stream of consciousness and the plot kept me guessing. The portrayal of mental illness and the character of Alicia Berenson are particularly compelling. The romantic elements blend beautifully with the darker themes. However, I found the book's take on democracy's issues rather simplistic and unconvincing. The genre is very young adult tbh. Overall, a masterful psychological thriller that will stay with you long after you finish reading.",
    "aspects": [
        {
            "aspect_span": "Joyce's stream of consciousness",
            "input_category": "CONTENT#STYLE",
            "input_sentiment": "positive",
            "input_aspect": "stream of consciousness"
        },
        {
            "aspect_span": "romantic elements",
            "input_category": "CONTENT#MOVEMENT",
            "input_sentiment": "positive",
            "input_aspect": "Romanticism"
        },
        {
            "aspect_span": "Alicia Berenson",
            "input_category": "CONTENT#CHARACTER",
            "input_sentiment": "positive",
            "input_aspect": "Alicia Berenson"
        },
        {
            "aspect_span": "democracy",
            "input_category": "CONTENT#TOPIC",
            "input_sentiment": "negative",
            "input_aspect": "democracy"
        },
        {
            "aspect_span": "mental illness",
            "input_category": "CONTENT#TOPIC",
            "input_sentiment": "positive",
            "input_aspect": "mental illness"
        },
        {
            "aspect_span": "very young adult",
            "input_category": "CONTENT#AUDIENCE",
            "input_sentiment": "neutral",
            "input_aspect": "young adult"
        }
    ]
}
"""

def load_dolce_mapping(file_path: str) -> Dict[str, str]:
    """
    Load the GlinerDOLCE.tsv file and create a mapping from aspect values to DOLCE classes.
    
    Args:
        file_path: Path to the GlinerDOLCE.tsv file
    
    Returns:
        Dictionary mapping aspect values to DOLCE classes
    """
    dolce_mapping = {}
    try:
        print(f"Loading DOLCE mapping from {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as file:
            # Skip header
            next(file)
            reader = csv.reader(file, delimiter='\t')
            for row in reader:
                if len(row) >= 5:  # Ensure we have enough columns
                    aspect_value = row[1].strip()  # VALUE column
                    dolce_class = row[4].strip()   # DOLCE column
                    if aspect_value and dolce_class:
                        dolce_mapping[aspect_value] = dolce_class
        print(f"Successfully loaded {len(dolce_mapping)} DOLCE mappings from {file_path}")
    except Exception as e:
        print(f"Error loading DOLCE mapping file {file_path}: {e}")
    return dolce_mapping

def generate_prompts(books_data_path: str, qualified_books_path: str, amazon_reviews_path: str, 
                    goodreads_reviews_path: str, output_dir: str, dolce_mapping_path: str, num_books=1000, reviews_per_book=10):
    """
    Generate prompts for semi-synthetic dataset generation.
    
    Args:
        books_data_path: Path to the book aspects JSON file
        qualified_books_path: Path to the qualified books JSON file with descriptions
        amazon_reviews_path: Path to the Amazon reviews TSV file
        goodreads_reviews_path: Path to the Goodreads reviews TSV file
        output_dir: Directory to save generated prompts
        dolce_mapping_path: Path to the GlinerDOLCE.tsv file
        num_books: Number of books to process
        reviews_per_book: Number of reviews to generate per book
    """
    print(f"Loading book data from {books_data_path}...")
    books_data = load_json_data(books_data_path)
    
    print(f"Loading qualified books data from {qualified_books_path}...")
    qualified_books = load_json_data(qualified_books_path)
    
    # Load DOLCE mapping
    dolce_mapping = load_dolce_mapping(dolce_mapping_path)
    
    # Create a mapping from book IDs to information
    book_info_map = create_book_descriptions_map(qualified_books)
    
    # Limit to num_books if there are more books available
    if len(books_data) > num_books:
        books_data = random.sample(books_data, num_books)
    else:
        num_books = len(books_data)
        print(f"Only {num_books} books available in the dataset")
    
    print(f"Loading Amazon reviews from {amazon_reviews_path}...")
    amazon_reviews = load_tsv_reviews(amazon_reviews_path, is_amazon=True)
    
    print(f"Loading Goodreads reviews from {goodreads_reviews_path}...")
    goodreads_reviews = load_tsv_reviews(goodreads_reviews_path, is_amazon=False)
    
    # Combine all reviews
    all_reviews = amazon_reviews + goodreads_reviews
    random.shuffle(all_reviews)
    
    print(f"Loaded {len(all_reviews)} reviews")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Normal distribution for number of aspects per review (mean=5, std=1.5)
    # This will generate mostly 3-7 aspects per review
    aspect_counts = np.random.normal(5, 1.5, num_books * reviews_per_book)
    aspect_counts = np.clip(aspect_counts, 0, 10).astype(int)
    
    # Get system prompt and example
    system_prompt = create_system_prompt()
    example = create_example()
    
    # Generate prompts
    print(f"Generating {num_books * reviews_per_book} prompts...")
    
    prompt_count = 0
    for book_idx, book in enumerate(books_data):
        book_id = book.get('book_id', '')
        
        # Get book info from the mapping
        book_info = book_info_map.get(book_id, {})
        
        book_title = book_info.get('title', book.get('title', f"Book_{book_idx}"))
        book_plot = book_info.get('plot', '')
        book_author = book.get('author', 'Unknown Author')
        
        print(f"Processing book {book_idx+1}/{num_books}: {book_title} by {book_author}")
        
        for review_idx in range(reviews_per_book):
            # Get number of aspects for this review
            num_aspects = aspect_counts[prompt_count]
            
            # Sample aspects for this review
            sampled_aspects = sample_aspects_for_review(book, num_aspects)
            
            # Get a random review
            review_text = "This is a placeholder review as no reviews were loaded."
            review_id = "unknown_id"
            if all_reviews:
                review = random.choice(all_reviews)
                review_text = review["text"]
                review_id = review["id"]
            
            # Create the user prompt
            user_prompt = f"Book: \"{book_title}\" by {book_author}\n\nPlot: {book_plot}\n\n"
            user_prompt += "Please create a new, realistic book review that incorporates the following aspects:\n"
            
            # Add aspects to highlight in the user prompt
            for i, aspect in enumerate(sampled_aspects, 1):
                aspect_value = aspect.get('value', '')
                aspect_category = aspect.get('category', '')
                aspect_type = aspect.get('type', '')
                aspect_sentiment = aspect.get('sentiment', '')
                
                if aspect_category and aspect_value:
                    # Only include category and value, not type or superclass
                    user_prompt += f"{i}. {aspect_category} ({aspect_value}); sentiment: {aspect_sentiment}\n"
            
            user_prompt += f"\nUse this review as a style reference (but DO NOT copy its content):\n\"{review_text}\"\n\n"
            user_prompt += "Generate a completely new review that mentions the aspects listed above while maintaining a natural, authentic tone."
            
            # Create the complete prompt JSON
            prompt = {
                "book_id": book_id,
                "review_id": review_id,
                "aspects": [],
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "example": example
            }
            
            # Add the sampled aspects
            for aspect in sampled_aspects:
                aspect_value = aspect.get('value', '')
                aspect_entry = {
                    "aspect": aspect_value,
                    "category": aspect.get('category', ''),
                    "type": aspect.get('type', ''),
                    "sentiment": aspect.get('sentiment', '')
                }
                
                # Add DOLCE class if available
                if aspect_value in dolce_mapping:
                    aspect_entry["DOLCEType"] = dolce_mapping[aspect_value]
                
                # Remove superclass from the saved prompt
                if 'superclass' in aspect:
                    # Don't include superclass in the prompt
                    pass
                
                prompt["aspects"].append(aspect_entry)
            
            # Save the prompt with the new naming convention
            output_file = os.path.join(output_dir, f"prompt_{prompt_count}_{book_id}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(prompt, f, indent=2, ensure_ascii=False)
            
            prompt_count += 1
    
    print(f"Generated {prompt_count} prompts in {output_dir}")

def main():
    # File paths
    books_data_path = "book_aspects_with_dolce.json"
    qualified_books_path = "qualified_books.json"
    amazon_reviews_path = "datasets/reviews-English-Amazon-sample_5000-characters_5000.tsv"
    goodreads_reviews_path = "datasets/reviews-Goodreads-English-characters-5000.tsv"
    dolce_mapping_path = "DOLCE.tsv"
    output_dir = "generated_prompts"
    
    print(f"Starting prompt generation process...")
    print(f"Using the following files:")
    print(f"- Book aspects: {books_data_path}")
    print(f"- Qualified books: {qualified_books_path}")
    print(f"- Amazon reviews: {amazon_reviews_path}")
    print(f"- Goodreads reviews: {goodreads_reviews_path}")
    print(f"- DOLCE mapping: {dolce_mapping_path}")
    print(f"- Output directory: {output_dir}")
    
    # Generate prompts
    generate_prompts(
        books_data_path=books_data_path,
        qualified_books_path=qualified_books_path,
        amazon_reviews_path=amazon_reviews_path,
        goodreads_reviews_path=goodreads_reviews_path,
        output_dir=output_dir,
        dolce_mapping_path=dolce_mapping_path,
        num_books=1000,
        reviews_per_book=10
    )
    
    print("Prompt generation complete!")

if __name__ == "__main__":
    main()