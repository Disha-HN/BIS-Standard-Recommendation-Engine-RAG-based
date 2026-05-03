import json, re, sys

with open('data/chunks.json', encoding='utf-8') as f:
    chunks = json.load(f)
with open('data/test_all_results.json', encoding='utf-8') as f:
    results = json.load(f)

lookup = {}
for c in chunks:
    code = c.get('std_code', '').strip()
    if code not in lookup:
        lookup[code] = {
            'title': c.get('title', ''),
            'category': c.get('category', ''),
            'text': c.get('text', '')[:300]
        }

def norm(s):
    return re.sub(r'[^a-z0-9]', '', s.lower())

def find(code):
    if code in lookup:
        return lookup[code]
    for k, v in lookup.items():
        if norm(k) == norm(code):
            return v
    return {'title': 'NOT FOUND', 'category': '?', 'text': ''}

for r in results:
    print(f"\n{'='*70}")
    print(f"[{r['id']}]")
    for i, code in enumerate(r['retrieved_standards'], 1):
        info = find(code)
        print(f"  {i}. {code:<35} | {info['category']:<20} | {info['title'][:50]}")
