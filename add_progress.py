import json

with open('main.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cuda_cell = {
 "cell_type": "code",
 "execution_count": None,
 "metadata": {},
 "outputs": [],
 "source": [
  "import torch\n",
  "print('CUDA available:', torch.cuda.is_available())\n",
  "if torch.cuda.is_available():\n",
  "    print('Device name:', torch.cuda.get_device_name(0))\n"
 ]
}

new_training_source = [
    "import torch\n",
    "from tqdm.auto import tqdm\n",
    "\n",
    "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
    "\n",
    "model = get_model(num_classes=7)\n",
    "model.to(device)\n",
    "\n",
    "optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)\n",
    "\n",
    "for epoch in range(10):\n",
    "    model.train()\n",
    "    total_loss = 0\n",
    "\n",
    "    progress_bar = tqdm(train_loader, desc=f\"Epoch {epoch}\")\n",
    "    for images, targets in progress_bar:\n",
    "        images = [img.to(device) for img in images]\n",
    "        targets = [{k: v.to(device) for k,v in t.items()} for t in targets]\n",
    "\n",
    "        loss_dict = model(images, targets)\n",
    "        loss = sum(loss for loss in loss_dict.values())\n",
    "\n",
    "        optimizer.zero_grad()\n",
    "        loss.backward()\n",
    "        optimizer.step()\n",
    "\n",
    "        total_loss += loss.item()\n",
    "        progress_bar.set_postfix(loss=f\"{loss.item():.4f}\")\n",
    "\n",
    "    print(f\"Epoch {epoch}: Average Loss = {total_loss/len(train_loader):.4f}\")\n"
]

new_cells = []
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_text = "".join(cell['source'])
        # Identify the training cell
        if 'for epoch in range(10):' in source_text and 'optimizer.step()' in source_text:
            # Insert the CUDA check cell right before the training cell
            new_cells.append(cuda_cell)
            # Update the training cell source to include tqdm progress bar
            cell['source'] = new_training_source
            
    new_cells.append(cell)

nb['cells'] = new_cells

with open('main.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook successfully updated.")
