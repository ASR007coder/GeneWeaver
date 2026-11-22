# src/result_ranker.py

def rank_guides(scored_guides, top_n=10):
    """
    Sorts the guide RNAs based on their biological scores.
    Assuming a higher score is better (or adjust logic based on your scoring).
    """
    if not scored_guides:
        return []

    # Sort in descending order based on the 'score' key
    # Adjust 'score' to whatever key name your biological_scoring.py uses
    sorted_guides = sorted(scored_guides, key=lambda x: x.get('score', 0), reverse=True)
    
    # Return the top N guides
    return sorted_guides[:top_n]

if __name__ == "__main__":
    # Dummy data to test the function
    test_data = [
        {"id": "guide_1", "sequence": "ACGT...", "score": 85},
        {"id": "guide_2", "sequence": "TGCA...", "score": 92},
        {"id": "guide_3", "sequence": "GATC...", "score": 78}
    ]
    
    print("Testing Result Ranker...")
    top_guides = rank_guides(test_data, top_n=2)
    for guide in top_guides:
        print(f"Ranked Guide: {guide['id']} with score {guide['score']}")