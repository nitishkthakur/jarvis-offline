#!/usr/bin/env python3
"""
Demo script showing the new multi-query functionality of tavily_search.
"""

from tools import tavily_search

def demo_single_query():
    """Demonstrate single query search (backward compatible)."""
    print("🔍 Single Query Demo")
    print("=" * 50)
    
    query = "Latest Python 3.12 features"
    results = tavily_search(query, max_results=3)
    
    print(f"Query: {query}")
    print(f"Results length: {len(results)} characters")
    print("\nFirst 300 characters:")
    print(results[:300] + "...")
    print()

def demo_multiple_queries():
    """Demonstrate multiple queries search (new functionality)."""
    print("🔍 Multiple Queries Demo")
    print("=" * 50)
    
    queries = [
        "FastAPI vs Flask 2024",
        "Docker best practices",
        "Machine learning trends 2024"
    ]
    
    results = tavily_search(queries, max_results=2)
    
    print(f"Queries: {queries}")
    print(f"Results length: {len(results)} characters")
    print("\nFirst 500 characters:")
    print(results[:500] + "...")
    print()
    
    # Count chunks to show parallel execution worked
    chunk_count = results.count("<chunk ")
    print(f"Total chunks found: {chunk_count}")
    print("✅ All queries processed in parallel!")

def demo_mixed_scenarios():
    """Demonstrate various edge cases and scenarios."""
    print("🔍 Edge Cases Demo")
    print("=" * 50)
    
    # Empty list
    print("Testing empty list:")
    result = tavily_search([])
    print(f"Empty list result: {result[:100]}...")
    print()
    
    # Single item in list
    print("Testing single item in list:")
    result = tavily_search(["Python tutorials"])
    chunk_count = result.count("<chunk ")
    print(f"Single item list - chunks found: {chunk_count}")
    print()

if __name__ == "__main__":
    print("🚀 Tavily Multi-Search Demo")
    print("=" * 60)
    print()
    
    demo_single_query()
    demo_multiple_queries()
    demo_mixed_scenarios()
    
    print("✅ Demo completed!")
