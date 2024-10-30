# Main
# Credit given to ChatGPT

import pandas as pd
import pulp

def read_csv(filename:str)->pd:
    return pd.read_csv(filename)

def get_ChoiceCount(myPanda:pd)->int:
    return int(myPanda.columns[0])

def get_Limits(myPanda:pd)->list:
    return myPanda.iloc[0, 1:].astype(int).tolist()

def get_Headers(myPanda:pd)->list:
    return myPanda.columns[1:]

def get_Prefrences(myPanda:pd)->pd:
    return myPanda.iloc[1:, 1:].astype(int).values

def get_People(myPanda:pd)->list:
    return myPanda.iloc[1:, 0].tolist()


def main():
    input_path = "input.csv"
    output_path = "output.csv"

    # read file
    data = read_csv(input_path)

    # setup var
    num_choices_per_person = get_ChoiceCount(data)
    limits = get_Limits(data)

    # Retrieve table data
    options = get_Headers(data)
    num_options = len(options) #get countofheaders

    preferences = get_Prefrences(data)          # as pd
    people_count = preferences.shape[0]         # as int
    people_names = get_People(data)             # as list


    # Step 2: Initialize PuLP
    model = pulp.LpProblem("Assignment_Optimization", pulp.LpMinimize)

    # Decision variables: binary variables for each person-option pair
    x = pulp.LpVariable.dicts("assign", [(i, j) for i in range(people_count) for j in range(num_options)], cat='Binary')
    
    # Objective function: minimize preference scores for assigned options
    model += pulp.lpSum(preferences[i, j] * x[(i, j)] for i in range(people_count) for j in range(num_options))

    # Constraint 1: Each person gets exactly the specified number of unique choices
    for i in range(people_count):
        model += pulp.lpSum(x[(i, j)] for j in range(num_options)) == num_choices_per_person

    # Constraint 2: Respect capacity limits for each option
    for j in range(num_options):
        model += pulp.lpSum(x[(i, j)] for i in range(people_count)) <= limits[j]

    # Step 3: Solve the Model

    model.solve()
    
    # Step 4: Generate Output
    # Initialize results DataFrame with names and columns for each choice
    output_df = pd.DataFrame({"Person": people_names})

    # For each person, extract their assigned options
    for i in range(people_count):
        assigned_options = [options[j] for j in range(num_options) if pulp.value(x[(i, j)]) == 1]
        # Fill in exactly `num_choices_per_person` columns
        for k in range(num_choices_per_person):
            column_name = f"Choice_{k+1}"
            output_df.at[i, column_name] = assigned_options[k] if k < len(assigned_options) else None

    # Step 5: Save Results to Output File
    output_df.to_csv(output_path, index=False)

    print(f"Results saved to {output_path}")




if __name__=="__main__":
    main()