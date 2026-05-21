import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from PIL import Image


class HAM10000Dataset(Dataset):
    """Custom Dataset for HAM10000 skin lesion images."""

    def __init__(self, metadata_csv, img_dir, transform=None):
        self.img_dir = img_dir
        self.df = pd.read_csv(metadata_csv)
        self.transform = transform

        # Encode labels eg. 'nv', 'mel', etc. to integers
        self.label_encoder = LabelEncoder()
        raw_labels = self.label_encoder.fit_transform(self.df["dx"])
        self.labels = torch.tensor(raw_labels, dtype=torch.long)

        # Preprocess features: age (numerical)[single number from 0 to 1], sex and localization (categorical)[0/1 vectors]
        self.feature_encoder = ColumnTransformer(
            transformers=[
                ("num", MinMaxScaler(), ["age"]),
                ("cat", OneHotEncoder(sparse_output=False), ["sex", "localization"]),
            ]
        )
        raw_features = self.feature_encoder.fit_transform(
            self.df[["age", "sex", "localization"]]
        )

        self.features = torch.tensor(raw_features, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Load image and apply transformations
        img_id = self.df.iloc[idx]["image_id"]
        img_path = f"{self.img_dir}/{img_id}.jpg"
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Get metadata features and label for the current index
        metadata_features = self.features[idx]
        label = self.labels[idx]

        return image, metadata_features, label

    @property
    def num_classes(self):
        return len(self.label_encoder.classes_)

    @property
    def feature_dim(self):
        return self.features.shape[1]
