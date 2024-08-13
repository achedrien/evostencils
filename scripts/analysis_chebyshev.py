import os
import pandas as pd
import matplotlib.pyplot as plt
import re
import seaborn as sns
import numpy as np

columns = [f'Coef_{i+1}' for i in range(91)] # .append("conv_factor")

df = pd.DataFrame(columns=columns)
# Iterate over each file in the directory
print(df)
for filename in os.listdir("./"):
    if filename.endswith('.txt'):
        file_path = os.path.join(filename)
        with open(file_path, 'r') as file:
            content = file.read()

        blocks = content.split('iterations_used')
        tensor_pattern = re.compile(r'tensor\(\[\[')

        for block in blocks:
            # if "tensor" in block:
                # print(" ") # block)
            tensor_matches = tensor_pattern.finditer(block)
            for match in tensor_matches:
                tensor_values = []
                tensor_str = block[match.start()+7: match.start()+890].replace('[', '').replace(']', '')# .replace('\n', '')
                conv_factor = float(block[-23:-2])
                # print(tensor_str)
                try:
                    tensor_list = [list(map(float, row.split(','))) for row in tensor_str.split('],\n        [')]
                    tensor_values.append(tensor_list[0])
                    ## print(pd.DataFrame([tensor_list[0]], columns=df.columns))
                    tensor_list[0].append(conv_factor)
                    df = pd.concat([df, pd.DataFrame([tensor_list[0]], columns=df.columns)], ignore_index=True)
                except Exception as e:
                    print(f"An error occurred: {e}")

plt.figure(figsize=(10, 6))
cmap = plt.cm.cool
print(df)
norm = plt.Normalize(np.log10(df["Coef_91"].min()), np.log10(df["Coef_91"].max()))

for index, row in df.iterrows():
    plt.plot(range(1, 91), row[:-1], color=cmap(norm(np.log10(row['Coef_91']))), label=f'Row {index+1}')

# Add labels and title
plt.xlabel('Weight number (10 per smooher, 9 times used)')
plt.ylabel('Weight value')
plt.show()