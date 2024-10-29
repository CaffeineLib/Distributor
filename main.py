# Main
# Credit given to ChatGPT

import pandas as pd
import pulp as lp

# Update these paths to match the locations on your machine
input_path = "input.csv"
output_path = "output.csv"

# Load the input file
data = pd.read_csv(input_path)

# Step 1: Parse the Input File
# Retrieve number of choices per person from the first cell
num_choices_per_person = int(data.columns[0])

# Retrieve limits (from second row, starting from the second column onward)
limits = data.iloc[0, 1:].astype(int).tolist()

# Retrieve options (headers in the first row, from the second column onward)
options = data.columns[1:]
num_options = len(options)

# Retrieve preferences table (from second row onward)
preferences = data.iloc[1:, 1:].astype(int).values
num_people = preferences.shape[0]
people_names = data.iloc[1:, 0].tolist()

# Step 2: Set Up the PuLP Model
model = lp.LpProblem("Day_Assignment_Optimization", lp.LpMinimize)

# Decision variables: binary variables for each person-option pair
x = lp.LpVariable.dicts("assign", [(i, j) for i in range(num_people) for j in range(num_options)], cat='Binary')

# Objective function: minimize preference scores for assigned options
model += lp.lpSum(preferences[i, j] * x[(i, j)] for i in range(num_people) for j in range(num_options))

# Constraint 1: Each person gets exactly the specified number of unique choices
for i in range(num_people):
    model += lp.lpSum(x[(i, j)] for j in range(num_options)) == num_choices_per_person

# Constraint 2: Respect capacity limits for each option
for j in range(num_options):
    model += lp.lpSum(x[(i, j)] for i in range(num_people)) <= limits[j]

# Step 3: Solve the Model
model.solve()

# Step 4: Generate Output
# Initialize results DataFrame with names and columns for each choice
output_df = pd.DataFrame({"Person": people_names})

# For each person, extract their assigned options
for i in range(num_people):
    assigned_options = [options[j] for j in range(num_options) if lp.value(x[(i, j)]) == 1]
    # Fill in exactly `num_choices_per_person` columns
    for k in range(num_choices_per_person):
        column_name = f"Choice_{k+1}"
        output_df.at[i, column_name] = assigned_options[k] if k < len(assigned_options) else None

# Step 5: Save Results to Output File
output_df.to_csv(output_path, index=False)

print(f"Results saved to {output_path}")
