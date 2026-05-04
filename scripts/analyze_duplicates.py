import json
from collections import defaultdict

def analyze(filepath):
    print(f"--- Analyzing {filepath} ---")
    queries = defaultdict(list)
    with open(filepath, 'r') as f:
        for line in f:
            data = json.loads(line)
            # Create a tuple of IDs for retrieved items to compare uniqueness
            if 'retrieved' in data:
                ids = []
                for item in data['retrieved']:
                    ids.append(item.get('question_id') or item.get('chunk_id'))
                queries[data['original_farmer_query']].append(tuple(ids))
    
    total = sum(len(v) for v in queries.values())
    unique_queries = len(queries)
    duplicates = total - unique_queries
    
    # Check if duplicate queries have different results
    diff_results = 0
    for query, results_list in queries.items():
        if len(results_list) > 1:
            # Check if all results are the same
            if len(set(results_list)) > 1:
                diff_results += 1

    print(f"Total entries: {total}")
    print(f"Unique queries: {unique_queries}")
    print(f"Duplicate queries found: {duplicates}")
    print(f"Queries with DIFFERENT retrieval results among duplicates: {diff_results}")
    print("\n")

analyze('outputs/logs_qa_retrieval.jsonl')
analyze('outputs/logs_pop_retrieval.jsonl')
