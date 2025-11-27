import torch
from torch_geometric.data import InMemoryDataset

class TrafficDataset(InMemoryDataset):
    def __init__(self, root, file_path, transform=None, pre_transform=None):

        self.file_path = file_path
        super().__init__(root, transform, pre_transform)
        
        # Load the data directly in __init__ to simplify the workflow
        self._load_data()

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def _load_data(self):
        print(f" Loading dataset from: {self.file_path}")
        try:
            # 1. Load the list of HeteroData objects
            # Set weights_only=False because we trust our own data file
            data_list = torch.load(self.file_path, weights_only=False)
            print(f"   Found {len(data_list)} graphs.")
            
            # 2. Collate them (Convert list -> Huge efficient Tensor)
            self.data, self.slices = self.collate(data_list)
            print(" Dataset loaded and collated successfully.")
            
        except FileNotFoundError:
            print(f" Error: File not found at {self.file_path}")
            raise

# Simple Test Block
if __name__ == "__main__":
    # Test loading the data we just collected
    FILE_PATH = "experiments/raw_data/traffic_data_1hr.pt"
    
    try:
        dataset = TrafficDataset(root="experiments", file_path=FILE_PATH)
        print(f"Dataset Length: {len(dataset)}")
        print(f"Sample Graph: {dataset[0]}")
    except Exception as e:
        print(f"Test failed: {e}")