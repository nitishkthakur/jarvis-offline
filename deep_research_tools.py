"""
Tavily Search Tool for retrieving web search results.
"""

import os
from typing import Optional, List, Dict, Any
import requests
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from tools import tavily_search
# Load environment variables from .env file
load_dotenv()
def add_agent_response_and_user_answer(agent_response: str, user_answers: str) -> str:
    agent_response = """\n\nagent_response: """ + agent_response
    user_answers = """\n\nuser_answers: """ + user_answers

    return agent_response + user_answers

def deep_tavily_search(
    query1: str,
    query2: str, 
    query3: str,
    query4: str,
    query5: str,
    query6: str,
    max_results: int = 15, 
    api_key: Optional[str] = None,
    include_domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None
) -> str:
    """
    Search the web using Tavily API for 6 questions and return formatted results in XML chunks as a string output. 
    Use this when the user asks for information which might require recent information or recent documentation. 
    Use it if you need more information. The 6 queries are different formulations of the same question.
    
    Args:
        query1 (str): The first search query to look up.
        query2 (str): The second search query to look up.
        query3 (str): The third search query to look up.
        query4 (str): The fourth search query to look up.
        query5 (str): The fifth search query to look up.
        query6 (str): The sixth search query to look up.
    
    Returns:
        str: Search results formatted in XML chunks
    """
    # Validate max_results
    if max_results < 1 or max_results > 20:
        return "<error>max_results must be between 1 and 20</error>"
    
    # Get API key from parameter or environment variable
    if api_key is None:
        api_key = os.getenv('TAVILY_API_KEY')
    
    if not api_key:
        return "<error>Tavily API key not found. Please provide api_key parameter or set TAVILY_API_KEY environment variable.</error>"
    
    # Construct queries list from the six query arguments
    queries = [query1, query2, query3, query4, query5, query6]
    
    # Filter out empty or None queries
    queries = [q for q in queries if q and q.strip()]
    
    if not queries:
        return "<error>No valid queries provided</error>"
    
    # Function to perform a single search
    def perform_single_search(single_query: str) -> Dict[str, Any]:
        """Perform a single Tavily search and return the raw response."""
        url = "https://api.tavily.com/search"
        
        payload = {
            "api_key": api_key,
            "query": single_query,
            "search_depth": "advanced",
            "include_raw_content": False,
            "max_results": max_results,
            "chunks_per_source": 1,
            "include_answer": True,
            "include_images": False
        }
        
        # Add domain filters if provided
        if include_domains:
            payload["include_domains"] = include_domains
        if exclude_domains:
            payload["exclude_domains"] = exclude_domains
        
        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            return {"query": single_query, "data": data, "error": None}
        except requests.exceptions.Timeout:
            return {"query": single_query, "data": None, "error": "Request timed out. Please try again."}
        except requests.exceptions.ConnectionError:
            return {"query": single_query, "data": None, "error": "Connection error. Please check your internet connection."}
        except requests.exceptions.HTTPError as e:
            if response.status_code == 401:
                return {"query": single_query, "data": None, "error": "Invalid API key. Please check your Tavily API key."}
            elif response.status_code == 429:
                return {"query": single_query, "data": None, "error": "Rate limit exceeded. Please try again later."}
            else:
                return {"query": single_query, "data": None, "error": f"HTTP error {response.status_code}: {str(e)}"}
        except json.JSONDecodeError:
            return {"query": single_query, "data": None, "error": "Invalid JSON response from API"}
        except Exception as e:
            return {"query": single_query, "data": None, "error": f"Unexpected error: {str(e)}"}
    
    # Execute searches in parallel
    search_results = []
    with ThreadPoolExecutor(max_workers=min(len(queries), 5)) as executor:
        # Submit all search tasks
        future_to_query = {executor.submit(perform_single_search, q): q for q in queries}
        
        # Collect results as they complete
        for future in as_completed(future_to_query):
            result = future.result()
            search_results.append(result)
    
    # Sort results by original query order to maintain consistency
    query_order = {q: i for i, q in enumerate(queries)}
    search_results.sort(key=lambda x: query_order[x["query"]])
    
    # Format all results into a single string
    all_formatted_results = []
    chunk_counter = 1
    total_results_count = 0
    
    # Add combined metadata at the beginning
    queries_list = ", ".join(f"'{q}'" for q in queries)
    combined_metadata = f"""<search_metadata>
Queries: [{queries_list}]
Total Queries: {len(queries)}
Max Results per Query: {max_results}
Search Mode: Advanced
Raw Content: False
</search_metadata>"""
    all_formatted_results.append(combined_metadata)
    
    # Process each search result
    for search_result in search_results:
        current_query = search_result["query"]
        data = search_result["data"]
        error = search_result["error"]
        
        # Add query separator
        query_header = f"""<query_section>
Query: {current_query}
</query_section>"""
        all_formatted_results.append(query_header)
        
        if error:
            error_chunk = f"<error>Query '{current_query}': {error}</error>"
            all_formatted_results.append(error_chunk)
            continue
        
        if not data or 'results' not in data:
            error_chunk = f"<error>No results found for query: '{current_query}'</error>"
            all_formatted_results.append(error_chunk)
            continue
        
        results = data['results']
        
        if not results:
            error_chunk = f"<error>No search results found for query: '{current_query}'</error>"
            all_formatted_results.append(error_chunk)
            continue
        
        # Add answer if available
        if 'answer' in data and data['answer']:
            answer_chunk = f"""<answer query="{current_query}">
{data['answer']}
</answer>"""
            all_formatted_results.append(answer_chunk)
        
        # Add search results with incremental numbering
        for result in results:
            title = result.get('title', 'No Title')
            url = result.get('url', 'No URL')
            content = result.get('content', 'No Content Available')
            score = result.get('score', 'N/A')
            published_date = result.get('published_date', 'N/A')
            
            # Clean up content - remove excessive whitespace
            content = ' '.join(content.split())
            
            chunk = f"""<chunk {chunk_counter}>
Query: {current_query}
Title: {title}
URL: {url}
Score: {score}
Published: {published_date}
Content: {content}
</chunk {chunk_counter}>"""
            
            all_formatted_results.append(chunk)
            chunk_counter += 1
            total_results_count += 1
    
    # Add final summary
    summary = f"""<results_summary>
Total Chunks: {chunk_counter - 1}
Total Results: {total_results_count}
Queries Processed: {len(queries)}
</results_summary>"""
    all_formatted_results.append(summary)
    
    return "\n\n".join(all_formatted_results)


def quick_search(query: str, num_results: int = 3) -> str:
    """
    Quick search function with fewer parameters for easy use.
    
    Args:
        query (str): The search query
        num_results (int): Number of results to return (default: 3)
    
    Returns:
        str: Formatted search results
    """
    return tavily_search(query, query, query, query, query, query, num_results)


def test_tavily_search():
    """Test function for the Tavily search tool."""
    print("🔍 Testing Tavily Search Tool")
    print("=" * 50)
    
    # Check if API key is available
    api_key = os.getenv('TAVILY_API_KEY')
    if not api_key or api_key == 'your_tavily_api_key_here':
        print("❌ TAVILY_API_KEY not properly configured!")
        print("\nTo set up the API key:")
        print("1. Copy .env.template to .env:")
        print("   cp .env.template .env")
        print("2. Edit .env file and replace 'your_tavily_api_key_here' with your actual API key")
        print("3. Get your API key from: https://tavily.com/")
        print("\nAlternatively, set the environment variable directly:")
        print("export TAVILY_API_KEY='your_actual_api_key'")
        return
    
    print(f"✅ API key found!")
    
    # Test: Six query parameters
    print("\n📝 Test: Six Query Parameters")
    print("-" * 30)
    query1 = "Python async programming"
    query2 = "ThreadPoolExecutor tutorial"
    query3 = "Tavily API documentation"
    query4 = "Python concurrency best practices"
    query5 = "asyncio vs threading comparison"
    query6 = "Web search API integration"
    max_results = 2
    
    print(f"Query 1: {query1}")
    print(f"Query 2: {query2}")
    print(f"Query 3: {query3}")
    print(f"Query 4: {query4}")
    print(f"Query 5: {query5}")
    print(f"Query 6: {query6}")
    print(f"Max Results per query: {max_results}")
    print("\nSearch Results:")
    print("-" * 20)
    
    results = tavily_search(query1, query2, query3, query4, query5, query6, max_results)
    print(f"Result length: {len(results)} characters")
    print("First 800 characters:")
    print(results[:800] + "..." if len(results) > 800 else results)

    print("\n" + "=" * 50)
    print("✅ Tests completed!")
    print(f"Result length: {len(results)} characters")
    print("✅ Six-parameter query functionality working!")


if __name__ == "__main__":
    test_tavily_search()
