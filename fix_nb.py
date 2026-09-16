import json

with open('main.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        for i, line in enumerate(source):
            if 'labels.append(ann["category_id"] + 1)' in line:
                source[i] = line.replace('labels.append(ann["category_id"] + 1)', 'labels.append(ann["category_id"])')
            if '# Labels (IMPORTANT: +1 for background class)' in line:
                source[i] = line.replace('# Labels (IMPORTANT: +1 for background class)', '# Labels')
            if 'model = get_model(num_classes=6)' in line:
                source[i] = line.replace('model = get_model(num_classes=6)', 'model = get_model(num_classes=7)')

with open('main.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
