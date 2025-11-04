import pandas as pd

# Dataset: first 10 rows of mtcars (relevant columns)
data = pd.DataFrame({
    'gear': [4, 4, 4, 3, 3, 3, 3, 4, 5, 5],
    'am': [1, 1, 1, 0, 0, 0, 0, 0, 1, 1],
    'vs': [0, 0, 1, 1, 1, 1, 1, 0, 1, 1]
})

# Function to calculate Gini index
def gini_index(groups, classes):
    n_instances = float(sum([len(group) for group in groups]))
    gini = 0.0
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            p = (group['gear'] == class_val).sum() / size
            score += p * p
        gini += (1.0 - score) * (size / n_instances)
    return gini

# Target variable and unique classes
classes = data['gear'].unique()

# a) Gini index before splitting
gini_before = gini_index([data], classes)

# b) Gini index and information gain for vs after splitting
groups_vs = [data[data['vs'] == val] for val in data['vs'].unique()]
gini_vs = gini_index(groups_vs, classes)
info_gain_vs = gini_before - gini_vs

# c) Gini index and information gain for am after splitting
groups_am = [data[data['am'] == val] for val in data['am'].unique()]
gini_am = gini_index(groups_am, classes)
info_gain_am = gini_before - gini_am

# Print results
print(f"Gini index before splitting: {gini_before:.2f}")
print(f"Gini index after splitting on vs: {gini_vs:.2f}")
print(f"Information gain for vs: {info_gain_vs:.2f}")
print(f"Gini index after splitting on am: {gini_am:.2f}")
print(f"Information gain for am: {info_gain_am:.2f}")

# d) Determine root node based on max information gain
root_node = 'vs' if info_gain_vs > info_gain_am else ('am' if info_gain_am > info_gain_vs else 'either am or vs')
print(f"Selected root node for splitting: {root_node}")


##OUTPUT

#Gini index before splitting: 0.64
#Gini index after splitting on vs: 0.40
#Information gain for vs: 0.24
#Gini index after splitting on am: 0.40
#Information gain for am: 0.24
#Selected root node for splitting: am
