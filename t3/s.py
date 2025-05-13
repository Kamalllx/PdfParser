import json

def convert_notebook_to_script(notebook_path, output_path="app2.py"):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    with open(output_path, 'w', encoding='utf-8') as out_file:
        out_file.write("# Auto-generated from {}\n\n".format(notebook_path))

        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                out_file.write("# --- Cell ---\n")
                out_file.write(''.join(cell['source']))
                out_file.write('\n\n')

    print(f"Notebook converted to {output_path}")

# Example usage:
convert_notebook_to_script("notebook4a08ae599b.ipynb")
