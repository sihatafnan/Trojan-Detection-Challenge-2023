import json

file = 'submission/predictions.json'

# with open(file, 'r') as f:
#   predictions = json.load(f)
#   # print(predictions)
#   for key, value in predictions.items():
#     # print(key, value[0])
#     for idx, val in enumerate(value):
#       # predictions[key][idx] = val.replace(key)
#       # replace key with add hi after the first word of key 
#       predictions[key][idx] = val.replace(key, key.split()[0] + ' hi ' + ' '.join(key.split()[1:]))
#     # break

with open(file, 'w') as f:
  json.dump(predictions, f, indent=4)

import itertools

# Example list of strings
strings = ["string1", "string2", ..., "string100"]

# Function to generate n-grams from a string
def generate_ngrams(text, n):
    ngrams = set()
    for i in range(len(text) - n + 1):
        ngrams.add(text[i:i + n])
    return ngrams

# Calculate Jaccard similarity between two strings based on n-grams
def jaccard_similarity(str1, str2, n=5):
    ngrams_str1 = generate_ngrams(str1, n)
    ngrams_str2 = generate_ngrams(str2, n)
    intersection = len(ngrams_str1.intersection(ngrams_str2))
    union = len(ngrams_str1.union(ngrams_str2))
    similarity = intersection / union
    return similarity

# Calculate average Jaccard similarity for each string
average_similarities = []
for i, string in enumerate(strings):
    total_similarity = 0
    num_comparisons = 0
    for other_string in strings:
        if string != other_string:
            total_similarity += jaccard_similarity(string, other_string)
            num_comparisons += 1
    average_similarity = total_similarity / num_comparisons if num_comparisons > 0 else 0
    average_similarities.append((string, average_similarity))

# Find the string with the minimum average Jaccard similarity
min_similarity = min(average_similarities, key=lambda x: x[1])

print(f"String with Minimum Average Jaccard Similarity: {min_similarity[0]}")
print(f"Minimum Average Jaccard Similarity: {min_similarity[1]}")
