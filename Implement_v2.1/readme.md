# Details for Implementation of our Task

## File and Code Structure

```
.
├── data/
│   └── *.json                         # Base data JSON file
├── results/
│   ├── pretrain/                      # Stores images for pretraining
│   └── SymbolGameGS/                  # Stores images for the SymbolGameGS
├── model/
│   ├── pretrain.pth                   # (Saved) pretrained model checkpoint
│   └── vision.pth                     # (Saved) vision model checkpoint
├── implement_v2.1.ipynb               # Jupyter notebook for the implementation
└── egg                                # EGG framework (modified)
```

### Main Components

1. **Data Generation & Loading**  
   - `generate_data_epoch(value, position_num)`: Generates a single 10×10 map with specified number of positions and masking.
   - `get_data(save_path, datasize, value, position_num)`: Generates or loads a dataset of size `datasize` from JSON.  
   - `Dataset`: A custom PyTorch `Dataset` that takes the JSON data and returns `(map, ground_truth, ground_truth_position)`.

2. **Visualization**
   - `visualize_data(inputlist, savepath, num)`: Plots the map, ground truth, prediction, and optionally a learned message.

3. **Vision & Pretraining**
   - `Vision`: A CNN-based module that transforms a 10×10 input map into a feature vector.
   - `PretrainNet`: Wraps the `Vision` model and outputs a 100-dimensional vector, then trained with a binary cross-entropy objective to predict masked areas.

4. **Sender/Receiver Games**
   - **SymbolGameGS** (Gumbel-Softmax):
     - `Sender`: Uses a frozen (no-grad) `Vision` model and produces a 100-dimensional message.
     - `Receiver`: Consumes the 100-dim message and outputs a 100-dimensional prediction (reshaped to 10×10).  
     - **Loss Function**: Includes BCE on target positions and masks, plus an accuracy metric for positions of interest.

5. **W&B Logging**
   - All training logs (loss, accuracy, etc.) are sent to W&B with `wandb.log(...)`.
   - Each run is uniquely named and saved under a specific W&B project (“core”).

---

## Key Hyperparameters

- **`--random_seed`**: RNG seed for reproducibility (default: `7`).
- **`--lr`**: Learning rate, used by the chosen optimizer (`adam` by default).
- **`--batch_size`**: Batch size (default: `32`).
- **`datasize`**: Number of samples to generate for the dataset (`10,000` by default).
- **`positionsize`**: Number of positions to place on the map (`2` by default).
- **`n_epochs`**: Number of training epochs for both pretraining and communication games.

These can be changed manually in the script or via command-line arguments (for EGG-based parameters).  

---