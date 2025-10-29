import json

with open('data/interventions/interventions.json', 'r') as f:
    data = json.load(f)
    print(f'Total interventions: {len(data)}')
    
    # Count by category
    categories = {}
    for intervention in data:
        cat = intervention['category']
        categories[cat] = categories.get(cat, 0) + 1
    
    print("\nInterventions by category:")
    for cat, count in categories.items():
        print(f"  {cat}: {count}")
