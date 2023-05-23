#!/usr/bin/env python

d1 = {"score": 3}
d2 = {"score": 2}
d3 = {"score": 4}
d4 = {"score": 1}

results = [d1, d2, d3, d4]

print(max(results))

print(results)

results_score = []
for res in results:
    results_score.append(res["score"])

max_score = max(results_score)
max_index = results_score.index(max_score)
print(max_index)

new_results = []
for res in results:
    if res["score"] == max_score:
        new_results.append(res)

print(new_results)