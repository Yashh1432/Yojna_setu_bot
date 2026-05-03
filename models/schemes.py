import json
import os

class SchemeModel:
    def __init__(self, file_path='datasets/full_preprocessed_schemes.json'):
        self.file_path = file_path
        self.data = self._load_data()

    def _load_data(self):
        if not os.path.exists(self.file_path):
            print(f"Warning: Dataset {self.file_path} not found.")
            return []
        
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading schemes: {e}")
            return []

    def get_by_category(self, category_name, limit=5):
        # Phase 1/2: Simple case-insensitive match
        category_name = category_name.lower()
        results = []
        for s in self.data:
            if category_name in s.get('category', '').lower():
                results.append(s)
                if len(results) >= limit:
                    break
        return results

    def filter_eligibility(self, schemes, age=None, state=None):
        # Phase 3: Simple Hardcoded Filtering
        filtered = []
        for s in schemes:
            criteria = s.get('eligibility_criteria', '').lower()
            
            # Simple keyword check for age if provided
            if age:
                # This is a very basic "dumb" check for Phase 3
                # Real logic comes in Phase 4 (LLM)
                if "at least 18" in criteria and age < 18:
                    continue
                if "minor" in criteria and age >= 18:
                    continue
            
            filtered.append(s)
            if len(filtered) >= 5:
                break
        return filtered

# Singleton
scheme_model = SchemeModel()
