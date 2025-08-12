#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import math
import gc

# Vision libraries
import torchvision
import matplotlib.pyplot as plt

import pytorch_lightning as pl
import torch.nn.functional as F
from torchvision.models import resnet50
HW = 'cpu'
device = torch.device(HW)


# In[2]:


from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
import torch
import torchvision
import random
n_nodes, n_rounds = 100, 100


# In[3]:


torch.set_float32_matmul_precision('medium')

# Reproducibility
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything()


# In[ ]:


def noniid(mode:str):

   if mode == 'cifar100':
# ---------- 1. Load CIFAR-100 (first 40 000 samples for clients) ----------
      transform = transforms.Compose([
            transforms.ToTensor()
])
      test_dataset = torchvision.datasets.CIFAR100(root='.', train=False, download=False, transform=transform)
      min_samples, max_samples = 20, 40
      n_classes = 100
      train_ds = datasets.CIFAR100(root='.' , train=True, download=True, transform=transform)
      samples_per_client = 400
      client_indices_split = np.arange(40000)
      central_indices_split = np.arange(40000, 50000)

   elif mode == 'cifar10':
# ---------- 1. Load CIFAR-10 (first 40 000 samples for clients) ----------
      transform = transforms.Compose([transforms.ToTensor()])
      test_dataset = torchvision.datasets.CIFAR10(root='.', train=False, download=False, transform=transform)
      min_samples, max_samples = 20, 40
      n_classes = 10
      samples_per_client = 400
      train_ds = datasets.CIFAR10(root='.' , train=True, download=False, transform=transform)
      client_indices_split = np.arange(40000)
      central_indices_split = np.arange(40000, 50000)


   elif mode == 'fashionmnist':
# ---------- 1. Load fashion mnist (first 40 000 samples for clients) ----------
      transform = transforms.Compose([
         transforms.ToTensor(),
         transforms.Normalize((0.2870,), (0.3532,))
      ])

      test_dataset = datasets.FashionMNIST(root='.', train=False, download=True, transform=transform)
      min_samples, max_samples = 20, 40
      n_classes = 10
      samples_per_client = 400      
      train_ds  = datasets.FashionMNIST(root='.', train=True, download=True, transform=transform)
      client_indices_split = np.arange(40000)
      central_indices_split = np.arange(40000, 60000)

   elif mode == 'femnist':

# ---------- 1. Load femnist (first 600_000 samples for clients) ----------

      min_samples, max_samples = 80, 90
      n_classes = 62
      samples_per_client = 5000
      transform = transforms.Compose([
         # transforms.ToTensor(),
         transforms.Normalize((0.9619,), (0.1631,))
      ])
      class FEMNIST(Dataset):
         def __init__(self, split='train', transform=None):
            obj = torch.load('flattened_emnist.pt', map_location='cpu', weights_only=False)
            if split == 'train':
                  self.data = obj['train_data']   
                  self.labels = obj['train_labels']  # numpy array
            else:
                  self.data = obj['test_data']
                  self.labels = obj['test_labels']
            self.transform = transform
            self.targets = self.labels  

         def __len__(self):
            return len(self.labels)

         def __getitem__(self, idx):
            x = self.data[idx]
            y = self.labels[idx]
            if isinstance(x, np.ndarray) and x.ndim == 1:
               x = x.reshape(28, 28)
            x = torch.tensor(x, dtype=torch.float32)
            if x.ndim == 2:
               x = x.unsqueeze(0)
            # x = x.repeat(3, 1, 1)  # [3,28,28]
            if self.transform:
               x = self.transform(x)
            return x, y

      train_ds = FEMNIST(split='train')#, transform=transform)
      test_dataset = FEMNIST(split='test')#, transform=transform)
      client_indices_split = np.arange(500000)
      central_indices_split = np.arange(500000, 671585)


   client_subset = torch.utils.data.Subset(train_ds, client_indices_split)
   central_subset = torch.utils.data.Subset(train_ds, central_indices_split)
   
   loader = DataLoader(client_subset, batch_size=1000, shuffle=False)
   # central_loader = DataLoader(central_subset, batch_size=1000, shuffle=False)

   data_np, labels_np, loaded = [], [], 0
   for x, y in loader:
      if loaded >=(samples_per_client/10):
         break
      take = (x.size(0))
      if mode == 'femnist':
         data_np.append(x[:take].numpy())
      else:
         data_np.append(x[:take].numpy().transpose(0, 2, 3, 1))
      labels_np.append(y[:take].numpy())
      loaded += 1


   data_np   = np.concatenate(data_np)
   labels_np = np.concatenate(labels_np)
   # print(f"Data shape after concatenation: {data_np.shape}")
   # ---------- 2. Parameters ----------
   n_nodes, n_rounds = 100, 100

   # ---------- 3. Build per-class index pools ----------
   class_pools = [np.where(labels_np == c)[0].tolist() for c in range(n_classes)]
   for pool in class_pools:
      np.random.shuffle(pool)

   # ---------- 4. Create heterogeneous class distribution for each client ----------
   client_class_distribution = []
   for cid in range(n_nodes):
      num_classes_for_client = np.random.randint(2, n_classes)
      selected_classes = np.random.choice(range(n_classes), size=num_classes_for_client, replace=False)

      class_weights = np.random.dirichlet(np.ones(num_classes_for_client))

      client_class_distribution.append({
         'classes': selected_classes,
         'weights': class_weights
      })

   # ---------- 5. Allocate samples to clients ----------
   client_data_pools = [[] for _ in range(n_nodes)]
   

   for cid in range(n_nodes):
      client_samples = []
      selected_classes = client_class_distribution[cid]['classes']
      class_weights = client_class_distribution[cid]['weights']

      samples_per_class = np.round(samples_per_client * class_weights).astype(int)

      while np.sum(samples_per_class) < samples_per_client:
         random_class_idx = np.random.randint(0, len(selected_classes))
         samples_per_class[random_class_idx] += 1

      while np.sum(samples_per_class) > samples_per_client:
         random_class_idx = np.random.randint(0, len(selected_classes))
         if samples_per_class[random_class_idx] > 1:
               samples_per_class[random_class_idx] -= 1

      for i, (class_id, num_samples) in enumerate(zip(selected_classes, samples_per_class)):
         if num_samples > 0:
               if len(class_pools[class_id]) < num_samples:
                  class_pools[class_id] = np.where(labels_np == class_id)[0].tolist()
                  np.random.shuffle(class_pools[class_id])

               sampled_indices = class_pools[class_id][:num_samples]
               class_pools[class_id] = class_pools[class_id][num_samples:]
               client_samples.extend(sampled_indices)

      client_data_pools[cid] = client_samples

   # ---------- 6. Round-based sampling ----------
   client_indices = [[[] for _ in range(n_rounds)] for _ in range(n_nodes)]

   for cid in range(n_nodes):
      client_pool = np.array(client_data_pools[cid])

      used_count = 0

      for rnd in range(n_rounds):
         num_samples_this_round = np.random.randint(min_samples, max_samples + 1)

         available_classes = client_class_distribution[cid]['classes']
         num_classes_this_round = np.random.randint(2, min(len(available_classes), 10) + 1)
         selected_classes_this_round = np.random.choice(available_classes, size=num_classes_this_round, replace=False)

         valid_indices = []
         for idx in client_pool:
               if labels_np[idx] in selected_classes_this_round:
                  valid_indices.append(idx)

         if len(valid_indices) >= num_samples_this_round:
               selected_indices = np.random.choice(valid_indices, size=num_samples_this_round, replace=False)
         else:
               selected_indices = valid_indices.copy()
               remaining_needed = num_samples_this_round - len(selected_indices)
               if remaining_needed > 0:
                  other_indices = np.random.choice(client_pool, size=remaining_needed, replace=False)
                  selected_indices.extend(other_indices)

         client_indices[cid][rnd] = selected_indices

   

   # ---------- 7. Verification ----------
   print("=== Verification ===")

   print("\n=== Client class distribution per round ===")
   for cid in range(10):
      print(f"Client {cid} (Overall classes: {client_class_distribution[cid]['classes']}):")
      for rnd in range(5):
         round_labels = [labels_np[idx] for idx in client_indices[cid][rnd]]
         unique_classes = np.unique(round_labels)
         print(f"  Round {rnd}: {len(unique_classes)} classes ({unique_classes}), {len(round_labels)} samples")

   for rnd in range(100):
      length = 0
      for cid in range(100):
         round_labels = [labels_np[idx] for idx in client_indices[cid][rnd]]
         unique_classes = np.unique(round_labels)
         length += len(unique_classes)
      avg = length /100
      print(f"  Round {rnd} avg classes per clients :{avg}")

   print("\n=== Class distribution in first round ===")
   for cid in range(20):
      if len(client_indices[cid][0]) > 0:
         round_labels = [labels_np[idx] for idx in client_indices[cid][0]]
         unique_classes = np.unique(round_labels)
         print(f"Client {cid}: {len(unique_classes)} classes, {len(round_labels)} samples")

   print("\n=== Client unique sample verification ===")
   for cid in range(100):
      c = []
      for rnd in range(100):
         c.extend(client_indices[cid][rnd])
      c = np.array(c)
      u = np.unique(c)
      print(f"Client {cid} - Number of unique samples across all rounds: {len(u)}")

   all_client_samples = []
   for cid in range(n_nodes):
      client_unique = set()
      for rnd in range(n_rounds):
         client_unique.update(client_indices[cid][rnd])
      all_client_samples.append(client_unique)

   overlap_found = False
   for i in range(n_nodes):
      for j in range(i + 1, n_nodes):
         if all_client_samples[i] & all_client_samples[j]:
               overlap_found = True
               break
      if overlap_found:
         break

   # Prepare client data and labels
   nodes_data = [[None for _ in range(n_rounds)] for _ in range(n_nodes)]
   nodes_labels = [[None for _ in range(n_rounds)] for _ in range(n_nodes)]

   used_indices = set()
   for cid in range(n_nodes):
      for rnd in range(n_rounds):
         idxs = client_indices[cid][rnd]
         used_indices.update(idxs)
         nodes_data[cid][rnd] = data_np[idxs]
         nodes_labels[cid][rnd] = labels_np[idxs]

   ovlp = [[] for _ in range(100)]
   for i in range(100):
      for j in range(100):
         ovlp[i].extend(client_indices[i][j])
   for i in range(100):
      for j in range(100):
         a = len(set(ovlp[i]) & set(ovlp[j]))
         if i!=j:
            print(f"overlap between client {i} and {j} is:{a}")
   # central_indices = list(set(central_indices_split) - used_indices)
   # central_data   = train_ds.data[central_indices]
   # central_labels = np.array(train_ds.targets)[central_indices]
   

   central_indices = list(set(central_indices_split) - used_indices)
   central_subset = torch.utils.data.Subset(train_ds, central_indices)
   central_dataloader = DataLoader(central_subset, batch_size=1000, shuffle=False)

   central_data_list = []
   central_labels_list = []

   for x, y in central_dataloader:
      central_data_list.append(x.numpy())  
      central_labels_list.append(y.numpy())

   central_data = np.concatenate(central_data_list, axis=0)  
   central_labels = np.concatenate(central_labels_list, axis=0)

   test_data_list = []
   test_labels_list = []
   test_dataloader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
   for x, y in test_dataloader:
      test_data_list.append(x.numpy())  
      test_labels_list.append(y.numpy())

   test_data = np.concatenate(test_data_list, axis=0)  
   test_labels = np.concatenate(test_labels_list, axis=0)

 
      
   return central_data, central_labels, nodes_data, nodes_labels, test_data, test_labels, test_dataset, train_ds, n_classes


# In[6]:


#femnist
# (77483, 1, 28, 28)
# (71585, 1, 28, 28)
# (51, 1, 28, 28)

#fashionmnist 
# (10000, 1, 28, 28)
# (20000, 1, 28, 28)
# (37, 28, 28, 1)

#cifar100
# (10000, 3, 32, 32)
# (10000, 3, 32, 32)
# (40, 32, 32, 3)

#cifar10
# (10000, 3, 32, 32)
# (10000, 3, 32, 32)
# (35, 32, 32, 3)


# In[ ]:


import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from enum import Enum

class DatasetMode(Enum):
    CLEAN = 0
    MIXED_BACKDOOR = 1
    FULL_BACKDOOR = 2
    LABEL_FLIP = 3
    MODEL_POISONING = 4

class CustomDataset(Dataset):
    def __init__(self, images, labels, mode=DatasetMode.CLEAN):
        self.images = images
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.to_tensor = transforms.ToTensor()
        self.mode = mode

        if self.mode == DatasetMode.MODEL_POISONING:
            unique_classes = np.unique(labels)
            n_classes = len(unique_classes)
            
            self.label_permutation = np.random.permutation(n_classes)
            
            self.class_mapping = {}
            for i, class_id in enumerate(unique_classes):
                mapped_class = unique_classes[self.label_permutation[i]]
                self.class_mapping[class_id] = mapped_class

        elif self.mode == DatasetMode.LABEL_FLIP:
            unique_classes = np.unique(labels)
            n_classes = len(unique_classes)
            
            self.flip_mapping = {}
            
            if n_classes >= 2:
                num_flips = max(1, min(2, n_classes // 2))
                
                priority_pairs = [(1, 9), (6, 4)]
                
                for from_class, to_class in priority_pairs:
                    if from_class in unique_classes and to_class in unique_classes and len(self.flip_mapping) < num_flips:
                        self.flip_mapping[from_class] = to_class
                
                while len(self.flip_mapping) < num_flips:
                    remaining_classes = [c for c in unique_classes if c not in self.flip_mapping]
                    if len(remaining_classes) < 2:
                        break
                        
                    from_class = np.random.choice(remaining_classes)
                    remaining_classes.remove(from_class)
                    
                    available_targets = [c for c in remaining_classes if c not in self.flip_mapping.values()]
                    if available_targets:
                        to_class = np.random.choice(available_targets)
                        self.flip_mapping[from_class] = to_class
            
            print(f"Label Flip Mapping: {self.flip_mapping}")

    def __len__(self):
        return len(self.images)

    def set_mode(self, mode):
        self.mode = mode

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx].item()

        if isinstance(image, np.ndarray):
            # [8, 28, 28, 1]
            
            if image.ndim == 3 and image.shape[2]== 3 :
                image = np.transpose(image, (2,0,1))
            
            elif image.ndim==3 and image.shape[2]==1 :
                image = np.transpose(image, (2,0,1))
            
            image = torch.FloatTensor(image)
            
            if image.max() > 1.0:
                image = image / 255.0

        if self.mode == DatasetMode.MIXED_BACKDOOR and idx % 2:
            image[:, [5,5,5,6,6,6,7,7,7], [5,6,7,5,6,7,5,6,7]] = 0
            label = 9

        elif self.mode == DatasetMode.FULL_BACKDOOR:
            image[:, [5,5,5,6,6,6,7,7,7], [5,6,7,5,6,7,5,6,7]] = 0
            label = 9

        elif self.mode == DatasetMode.LABEL_FLIP:
            if np.random.random() < 0.3:
                if hasattr(self, 'flip_mapping'):
                    label = self.flip_mapping.get(label, label)
                else:
                    if label == 1:
                        label = 9
                    elif label == 6:
                        label = 4

        elif self.mode == DatasetMode.MODEL_POISONING:
            if hasattr(self, 'class_mapping'):
                label = self.class_mapping.get(label, label)
            else:
                if label < len(self.label_permutation):
                    label = self.label_permutation[label]

        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.long)
        else:
            label = label.clone().detach().long()

        return image, label


# In[8]:


def create_round_dataloaders(nodes_data, nodes_labels, n_nodes, n_rounds, attack_modes):
    result_loaders = []

    for j in range(n_rounds):
        round_dataloaders = []
        for i in range(n_nodes):
            if isinstance(attack_modes, dict):
                mode = attack_modes.get(i, DatasetMode.CLEAN)
            else:
                mode = attack_modes
                
            dataloader = DataLoader(
                CustomDataset(
                    nodes_data[i][j],
                    nodes_labels[i][j],
                    mode=mode
                ),
                batch_size=8,
                shuffle=True,
                num_workers=8,
                drop_last=True,
                pin_memory=True
            )
            round_dataloaders.append(dataloader)
        result_loaders.append(round_dataloaders)
        

    return result_loaders

def dataloader(mode: str, malicious_clients: int = 24, target_label: int = 9):

    central_data, central_labels, nodes_data, nodes_labels, test_data, test_labels, test_dataset, train_ds, n_classes = noniid(mode)
    

    n_nodes, n_rounds = 100, 100

    attack_configs = {
        'clean': DatasetMode.CLEAN,
        'backdoor': {i: DatasetMode.MIXED_BACKDOOR if i < malicious_clients else DatasetMode.CLEAN 
                    for i in range(n_nodes)},
        'model_poison': {i: DatasetMode.MODEL_POISONING if i < malicious_clients else DatasetMode.CLEAN 
                        for i in range(n_nodes)},
        'label_flip': {i: DatasetMode.LABEL_FLIP if i < malicious_clients else DatasetMode.CLEAN 
                        for i in range(n_nodes)}
    }

    dataloaders = {}
    for attack_name, attack_mode in attack_configs.items():
        dataloaders[f'nodes_{attack_name}_data_loaders'] = create_round_dataloaders(
            nodes_data, nodes_labels, n_nodes, n_rounds, attack_mode
        )
        
    central_data_loader = DataLoader(
        CustomDataset(central_data, central_labels, mode=DatasetMode.CLEAN),
        batch_size=64,
        shuffle=True,
        num_workers=8,
        drop_last=True,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=1
    )

    # Test dataloaders 
    if mode == 'femnist':
        eval_data_loader = DataLoader(
            CustomDataset(test_data[:7000], test_labels[:7000], mode=DatasetMode.CLEAN),
            batch_size=100,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=1
        )
    else:
        eval_data_loader = DataLoader(
            CustomDataset(test_data[:600], test_labels[:600], mode=DatasetMode.CLEAN),
            batch_size=64,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=1
        )

    clean_label_indexs = np.array(test_labels) != target_label
    backdoor_testset_data_loader = DataLoader(
        CustomDataset(
            test_data[clean_label_indexs][:600], 
            np.array(test_labels)[clean_label_indexs][:600], 
            mode=DatasetMode.FULL_BACKDOOR
        ),
        batch_size=100,
        shuffle=False,
        num_workers=8,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=1
    )


    return (
        backdoor_testset_data_loader, 
        eval_data_loader,
        central_data_loader, 
        dataloaders['nodes_label_flip_data_loaders'],
        dataloaders['nodes_clean_data_loaders'], 
        dataloaders['nodes_model_poison_data_loaders'],
        dataloaders['nodes_backdoor_data_loaders'],
        test_dataset,train_ds, n_classes
    )


# In[9]:


# backdoor_data_loader,\
#    eval_data_loader, central_data_loader,\
#         nodes_label_flip_data_loaders, nodes_clean_data_loaders,\
#            nodes_model_poison_data_loaders ,nodes_data_loaders, test_dataset,\
#               train_dataset, n_classes= output


# In[ ]:


import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
class ImageClassificationBase(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.last_printed_epoch = -1
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = F.cross_entropy(out, y, label_smoothing=0.05)
        acc = (out.argmax(dim=1) == y).float().mean()
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = F.cross_entropy(out, y)
        acc = (out.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        # Skip sanity check - only print during actual training
        if self.trainer.sanity_checking:
            return
            
        # Only print once per epoch
        if self.current_epoch != self.last_printed_epoch:
            # Get the logged values from the current epoch
            val_loss = self.trainer.logged_metrics.get('val_loss')
            val_acc = self.trainer.logged_metrics.get('val_acc')
            
            if val_loss is not None and val_acc is not None:
                print(f"Epoch {self.current_epoch}: Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                self.last_printed_epoch = self.current_epoch

    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        acc = (out.argmax(dim=1) == y).float().mean()
        self.log('test_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=5e-5,
            weight_decay=1e-2
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True,
            threshold=1e-4,
            min_lr=1e-5
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }



# In[11]:


def conv_block(in_channels, out_channels, pool=False):
        """
        Convolutional block with BatchNorm and ReLU activation
        """
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if pool: 
            layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)


# In[12]:


def cifarn(eval_data_loader, n_classes, modelname):
   class ResNet9(ImageClassificationBase):
      def __init__(self, num_classes):
         super().__init__()
         self.save_hyperparameters("num_classes")
         # Load pretrained ResNet50
         model = resnet50(weights="IMAGENET1K_V1")
         # Separate the fully connected (fc) layer from the rest of the model
         features = torch.nn.Sequential(*(list(model.children())[:-1]))
         # for param in features.parameters():
         #     param.requires_grad = False

         self.features = features
         self.flatten = nn.Flatten()
         self.fc = nn.Sequential(
               nn.Dropout(0.3),
               torch.nn.Linear(model.fc.in_features, num_classes)
         )

      def forward(self, xb):
         xb = F.interpolate(xb, size=(224, 224), mode='bilinear', align_corners=False)
         out = self.features(xb)
         out = self.flatten(out)
         
         out = self.fc(out)
         return out
   globalmodel = ResNet9(num_classes=n_classes)
   trainer = pl.Trainer(
   max_epochs=5,
   accelerator='cpu'
   #,devices=[0]
      )
   if modelname == 'cifar100':
        globalmodel = ResNet9.load_from_checkpoint('cifar100.ckpt', num_classes=n_classes)
   else: 
       globalmodel.load_state_dict(torch.load('cifar10.pth'))
   trainer.test(globalmodel, eval_data_loader)

   return globalmodel


# In[13]:


def fashionmnist(eval_data_loader, n_classes):

    class ResNet9(ImageClassificationBase):
        """
        ResNet9 architecture for Fashion-MNIST and FEMNIST
        """
        
        def __init__(self, num_classes=10):
            super().__init__()
            self.save_hyperparameters()
            
            # Feature extraction layers
            self.conv1 = conv_block(1, 64)
            self.conv2 = conv_block(64, 128, pool=True)
            self.res1 = nn.Sequential(
                conv_block(128, 128), 
                conv_block(128, 128)
            )
            
            self.conv3 = conv_block(128, 256, pool=True)

            # aggregation contribution layers
            self.conv4 = conv_block(256, 512, pool=True)
            self.res2 = nn.Sequential(
                conv_block(512, 512), 
                conv_block(512, 512)
            )
            
            # Adaptive pooling for flexible input sizes
            self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
            
            # Classification head with balanced dropout
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(0.4),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(128, num_classes)
            )
            
            self._init_weights()
            
        def _init_weights(self):

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu' )
                elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        def forward(self, x):
            # Initial convolutions
            out = self.conv1(x)
            out = self.conv2(out)
            
            # First residual block
            out = self.res1(out) + out
            
            # nid convolutions
            out = self.conv3(out)

            # Hi level features
            out = self.conv4(out)
            out = self.res2(out) + out
            
            # Adaptive pooling and classification
            out = self.adaptive_pool(out)
            out = self.classifier(out)
            
            return out
    globalmodel = ResNet9(num_classes=n_classes)
    trainer = pl.Trainer(
    max_epochs=5,
    accelerator='cpu'
    #,devices=1
        )

    globalmodel = ResNet9.load_from_checkpoint('fashionmnist1.ckpt', num_classes=n_classes)
    trainer.test(globalmodel, eval_data_loader) 
    return globalmodel

    


# In[14]:


def femnist(eval_data_loader, n_classes):
    class ResNet9Fixed(ImageClassificationBase):
        def __init__(self, num_classes=62):
            super().__init__()
            self.save_hyperparameters()
            
            self.conv1 = conv_block(1, 32)
            self.conv2 = conv_block(32, 64, pool=True)
            self.res1 = nn.Sequential(conv_block(64, 64), conv_block(64, 64))
            
            self.conv3 = conv_block(64, 128, pool=True)
            self.conv4 = conv_block(128, 256, pool=True)
            self.res2 = nn.Sequential(conv_block(256, 256), conv_block(256, 256))
            
            self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
            
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(128, num_classes)
            )

        def forward(self, x):
            out = self.conv1(x)
            out = self.conv2(out)
            out = self.res1(out) + out
            out = self.conv3(out)
            out = self.conv4(out)
            out = self.res2(out) + out
            out = self.adaptive_pool(out)
            out = self.classifier(out)
            return out

    
    globalmodel = ResNet9Fixed(num_classes=n_classes)


    trainer = pl.Trainer(
        max_epochs=5,
        accelerator='cpu',
        #,devices=[0],
        enable_progress_bar=True,
        log_every_n_steps=10,
        gradient_clip_val=0.5
    )
    globalmodel = ResNet9Fixed.load_from_checkpoint('femnist-stable-epoch=05-val_acc=0.67.ckpt', num_classes=n_classes)

    trainer.test(globalmodel, eval_data_loader) 
    return globalmodel

#   lr=5e-4, weight_decay=1e-4
        
#         scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#             optimizer,
#             mode='min',
#             factor=0.7,
#             patience=5,
#             min_lr=1e-6,
#             verbose=True
#         )


# In[ ]:


import logging
from copy import deepcopy
from math import e
import pytorch_lightning as pl

# Logging
logger = logging.getLogger("pytorch_lightning")
old_level = logger.level
logger.setLevel(logging.WARNING)

def freeze_except_classifier(model, mode:str):
    # Freeze all parameters first
   for param in model.parameters():
        param.requires_grad = False

    # Unfreeze only the specified classifier layers
   if mode in ['cifar10','cifar100']:
         layers_to_unfreeze = [
            "features.7.0.conv1.weight",
            "features.7.0.conv2.weight", 
            "features.7.0.conv3.weight",
            "features.7.0.downsample.0.weight",
            "features.7.1.conv1.weight",
            "features.7.1.conv2.weight",
            "features.7.1.conv3.weight",
            "features.7.2.conv1.weight",
            "features.7.2.conv2.weight",
            "features.7.2.conv3.weight",
            "fc.1.weight",
            "fc.1.bias"
         ]
   elif mode == 'femnist':
       
       layers_to_unfreeze = [
    # conv3 layers (mid-to-high level)
    "conv3.0.weight", "conv3.1.weight", "conv3.1.bias",
    
    # conv4 layers (high level)
    "conv4.0.weight", "conv4.1.weight", "conv4.1.bias",
    
    # res2 layers (highest level)
    "res2.0.0.weight", "res2.0.1.weight", "res2.0.1.bias",
    "res2.1.0.weight", "res2.1.1.weight", "res2.1.1.bias", 
    
    # classifier layers
    "classifier.2.weight", "classifier.2.bias",
    "classifier.5.weight", "classifier.5.bias"
         ]

   elif mode == 'fashionmnist':
       
        layers_to_unfreeze = [
    "conv4.0.weight",
    "conv4.1.weight",
    "conv4.1.bias",
    
    "res2.0.0.weight",
    "res2.0.1.weight",
    "res2.0.1.bias",
    
    "res2.1.0.weight",
    "res2.1.1.weight",
    "res2.1.1.bias",
    
    "classifier.2.weight",
    "classifier.2.bias",
    "classifier.3.weight",
    "classifier.3.bias",
    
    "classifier.6.weight",
    "classifier.6.bias",
    "classifier.7.weight",
    "classifier.7.bias",
    
    "classifier.10.weight",
    "classifier.10.bias"
]
    
       
   for name, param in model.named_parameters():
        if name in layers_to_unfreeze:
            param.requires_grad = True

def get_dataloader_by_state(state, i, n_round,
                            nodes_label_flip_data_loaders, nodes_clean_data_loaders,
                            nodes_model_poison_data_loaders,nodes_data_loaders):
    if state == 'backdoor':
        return nodes_data_loaders[n_round][i]
    elif state == 'modelpoisoning':
        return nodes_model_poison_data_loaders[n_round][i]
    elif state == 'labelflip':
        return nodes_label_flip_data_loaders[n_round][i]
    else:
        return nodes_clean_data_loaders[n_round][i]

def finetune(ple, modelname, model, n_nodes, n_round, state, modelpoisoning=False):

    backdoor_data_loader, \
   eval_data_loader, central_data_loader,\
        nodes_label_flip_data_loaders, nodes_clean_data_loaders,\
           nodes_model_poison_data_loaders ,nodes_data_loaders, test_dataset,\
              train_dataset, n_classes= ple
    

    node_models = []
    for i in range(n_nodes):
        node_model = deepcopy(model)
        freeze_except_classifier(node_model, modelname)
        node_models.append(node_model)


    for i, node_model in enumerate(node_models):
        node_model = node_model.to(device)
        # Print status
        node_status = 'Attack' if i < 33 and state != 'benign' else 'Normal'
        print(f'\n=== Fine-tuning Node {i} [{node_status}] | Mode: {state} ===')

        # Get appropriate dataloader
        _dataloader = get_dataloader_by_state(state, i, n_round, nodes_label_flip_data_loaders, nodes_clean_data_loaders,nodes_model_poison_data_loaders,nodes_data_loaders)

        trainer = pl.Trainer(
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            gradient_clip_val=1.0,
            max_epochs=1,
            accelerator='cpu'
           #, devices=[1]
        
        )
        trainer.fit(node_model, _dataloader, eval_data_loader)

        if i < 33 and modelpoisoning:
            print("mp is activated")
            with torch.no_grad():
                for param in node_model.parameters():
                    # Apply 16.3% Gaussian noise
                    noise = torch.normal(0, 0.5 * torch.std(param), size=param.shape).to(param.device)
                    param.add_(noise)

        node_model = node_model.to('cpu')
        torch.cuda.empty_cache()
        gc.collect()
    return node_models

# Restore logging
logger.setLevel(old_level)


# In[16]:


def selective_weighted_average_aggregation(node_models, global_model, mode:str):
    averaged_model = deepcopy(global_model)

    # Define the layers to aggregate
    if mode in ['cifar10','cifar100']:
         layers_to_aggregate = [
            "features.7.0.conv1.weight",
            "features.7.0.conv2.weight", 
            "features.7.0.conv3.weight",
            "features.7.0.downsample.0.weight",
            "features.7.1.conv1.weight",
            "features.7.1.conv2.weight",
            "features.7.1.conv3.weight",
            "features.7.2.conv1.weight",
            "features.7.2.conv2.weight",
            "features.7.2.conv3.weight",
            "fc.1.weight",
            "fc.1.bias"
         ]
    elif mode == 'femnist':
       
       layers_to_aggregate = [
            # conv3 layers (mid-to-high level)
            "conv3.0.weight", "conv3.1.weight", "conv3.1.bias",
            
            # conv4 layers (high level)
            "conv4.0.weight", "conv4.1.weight", "conv4.1.bias",
            
            # res2 layers (highest level)
            "res2.0.0.weight", "res2.0.1.weight", "res2.0.1.bias",
            "res2.1.0.weight", "res2.1.1.weight", "res2.1.1.bias", 
            
            # classifier layers
            "classifier.2.weight", "classifier.2.bias",
            "classifier.5.weight", "classifier.5.bias"
        ]

    elif mode == 'fashionmnist':
       
        layers_to_aggregate = [
                "conv4.0.weight",
                "conv4.1.weight",
                "conv4.1.bias",
                
                "res2.0.0.weight",
                "res2.0.1.weight",
                "res2.0.1.bias",
                
                "res2.1.0.weight",
                "res2.1.1.weight",
                "res2.1.1.bias",
                
                "classifier.2.weight",
                "classifier.2.bias",
                "classifier.3.weight",
                "classifier.3.bias",
                
                "classifier.6.weight",
                "classifier.6.bias",
                "classifier.7.weight",
                "classifier.7.bias",
                
                "classifier.10.weight",
                "classifier.10.bias"
            ]

    # Step 1: Average fine-tuned layers across selected models
    averaged_params = {
        key: torch.mean(torch.stack([node_model.state_dict()[key].float() for node_model in node_models]), dim=0)
        for key in layers_to_aggregate
    }

    # Step 2: Blend with global model using alpha
    # for key in layers_to_aggregate:
    #     averaged_params[key] = averaged_params[key]

    # Step 3: Load blended layers into a new model
    new_state_dict = global_model.state_dict()
    for key in layers_to_aggregate:
        new_state_dict[key] = averaged_params[key]

    averaged_model.load_state_dict(new_state_dict)
    return averaged_model


# In[17]:



import random
import torch
import gc
from copy import deepcopy
def random_client_selection(node_models, globalmodel, modelname):
    # aggregation_configs = [
    #     (33, 34),
    #     (32, 35),
    #     (31, 36),
    #     .....
    #     (0, 67)
    # ]
    n_nodes=100
    start = 33
    end = 0
    aggregation_configs = []
    sum = 67
    for i in range(start + 1):
        aggregation_configs.append((start - i, sum - (start - i)))
        if start - i == 0: 
            aggregation_configs.append((0,0))
            break
    aggregated_models = []

    for num_attack, num_normal in aggregation_configs:
        if num_attack == 0 and num_normal == 0:
            # Full aggregation
            selected_indices = random.sample(range(0, n_nodes), n_nodes)
        else:
            # Controlled attack aggregation
            attack_indices = random.sample(range(0, 33), num_attack)
            normal_indices = random.sample(range(33, n_nodes), num_normal)
            selected_indices = attack_indices + normal_indices

        selected_models = [node_models[i] for i in selected_indices]

        # Aggregate selected models
        agg_model = selective_weighted_average_aggregation(selected_models, globalmodel, modelname)
        aggregated_models.append((agg_model, selected_indices))

        # Clear memory
        del selected_models
        torch.cuda.empty_cache()
        gc.collect()

    return aggregated_models


# In[18]:
def real_random_client_selection(node_models, globalmodel, modelname):
    aggregated_models = []
    n_nodes=100
    for i in range(int(1 / 3 * n_nodes) + 1):
        if i == 0:
            # Full aggregation
            selected_indices = random.sample(range(0, n_nodes), n_nodes)
        else:
            # Partial aggregation ( 2n/3 out of n)
            selected_indices = random.sample(range(0, n_nodes), int(2 / 3 * n_nodes))

        selected_models = [node_models[indice] for indice in selected_indices]

        agg_model = selective_weighted_average_aggregation(selected_models, globalmodel, modelname)
        aggregated_models.append((agg_model, selected_indices))

        # Cleanup after aggregation
        del selected_models
        torch.cuda.empty_cache()
        gc.collect()

    return aggregated_models


import torch
import logging
from pytorch_lightning import Trainer

def evaluate_model_lightning(model, dataloader):
    # Disable logging and checkpoints for evaluation
    model = model.to(device)
    logger = logging.getLogger("pytorch_lightning")
    old_level = logger.level
    logger.setLevel(logging.WARNING)

    trainer = Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator='cpu'
       #, devices=[1]
        
    )

    # Run test
    model.eval()
    test_result = trainer.test(model, dataloaders=dataloader, verbose=False)

    # Restore log level
    logger.setLevel(old_level)

    # Extract accuracy
    if test_result and "test_acc" in test_result[0]:
        acc = test_result[0]["test_acc"]
    else:
        acc = 0.0  #  key is missing

    model = model.to('cpu')
    return acc


# In[19]:


def compute_loss_lightning(model, dataloader):
    model.to(device)
    model.eval()

    total_loss = 0.0
    total_samples = 0

    # Temporarily silence logging
    logger = logging.getLogger("pytorch_lightning")
    old_level = logger.level
    logger.setLevel(logging.WARNING)

    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch
            inputs = inputs.to(model.device)
            targets = targets.to(model.device)

            outputs = model(inputs)

            if isinstance(outputs, dict) and "loss" in outputs:
                loss = outputs["loss"]
            else:
                loss = torch.nn.functional.cross_entropy(outputs, targets, reduction='sum')

            total_loss += loss.item()
            total_samples += targets.size(0)

    logger.setLevel(old_level)

    avg_loss = total_loss / total_samples if total_samples > 0 else float("inf")
    model = model.to('cpu')
    return avg_loss


# In[20]:


import logging

def FEDBAGSAFE(modelname:str, n_rounds, n_nodes, mode=False):
   logger = logging.getLogger("pytorch_lightning")
   old_level = logger.level
   logger.setLevel(logging.WARNING)

   states = [
    # Phase 1: Stable Initialization (Rounds 1-10)
    'benign', 'benign', 'benign', 'benign', 'benign',
    'benign', 'benign', 'benign', 'benign', 'benign',
    
    # Phase 2: Gradual Attack Introduction (Rounds 11-25)
    'benign', 'benign', 'backdoor', 'benign', 'benign',
    'labelflip', 'benign', 'benign', 'modelpoisoning', 'benign',
    'benign', 'backdoor', 'benign', 'labelflip', 'benign',
    
    # Phase 3: Moderate Attack Frequency (Rounds 26-50)
    'benign', 'backdoor', 'benign', 'modelpoisoning', 'benign',
    'labelflip', 'benign', 'backdoor', 'benign', 'modelpoisoning',
    'benign', 'labelflip', 'benign', 'backdoor', 'benign',
    'modelpoisoning', 'benign', 'labelflip', 'benign', 'backdoor',
    'benign', 'modelpoisoning', 'benign', 'labelflip', 'benign',
    
    # Phase 4: High-Intensity Attacks (Rounds 51-75)
    'backdoor', 'modelpoisoning', 'benign', 'labelflip', 'backdoor',
    'benign', 'modelpoisoning', 'labelflip', 'benign', 'backdoor',
    'modelpoisoning', 'benign', 'labelflip', 'backdoor', 'benign',
    'modelpoisoning', 'labelflip', 'benign', 'backdoor', 'modelpoisoning',
    'benign', 'labelflip', 'backdoor', 'benign', 'modelpoisoning',
    
    # Phase 5: Consecutive Attack Stress Test (Rounds 76-85)
    'backdoor', 'backdoor', 'modelpoisoning', 'modelpoisoning', 'labelflip',
    'labelflip', 'backdoor', 'modelpoisoning', 'labelflip', 'benign',
    
    # Phase 6: Recovery and Final Validation (Rounds 86-100)
    'benign', 'benign', 'backdoor', 'benign', 'benign',
    'modelpoisoning', 'benign', 'benign', 'labelflip', 'benign',
    'benign', 'backdoor', 'benign', 'benign', 'benign'
]
   ple = dataloader(modelname)
   eval_data_loader = ple[1]
   backdoor_data_loader = ple[0]
   n_classes = ple[9]
   if modelname in['cifar100','cifar10']:
       globalmodel = cifarn(eval_data_loader,n_classes, modelname)
   elif modelname == 'fashionmnist':
       globalmodel = fashionmnist(eval_data_loader,n_classes)
   elif modelname == 'femnist':
       globalmodel = femnist(eval_data_loader,n_classes)
   
   model = globalmodel
   
   best_score_prev = 0.0
    
   for round_id in range(n_rounds):
        print(f"\n=== Round {round_id} | State = {states[round_id]} ===")

        # Fine-tune per-node models
        node_models = finetune(
            ple,
            modelname,
            model,
            n_nodes,
            round_id,
            states[round_id],
            modelpoisoning=(states[round_id] == 'modelpoisoning')
        )

        # Aggregate models
        if mode:
            aggregated_models = random_client_selection(node_models, globalmodel, modelname)
        else:
            aggregated_models = real_random_client_selection(node_models, globalmodel, modelname)
            # pass
        del node_models
        gc.collect()

        best_model_this_round = None
        best_score_this_round = 0.0
        best_agg_id = -1
        contributors = []

        for agg_id, (agg_model, contributing_nodes) in enumerate(aggregated_models):
            print(f"\n=== Evaluating Aggregated Model {agg_id} from nodes {contributing_nodes} ===")

            acc = evaluate_model_lightning(agg_model, eval_data_loader)
            loss = compute_loss_lightning(agg_model, eval_data_loader)
            score = acc / loss if loss > 0 else 0.0

            print(f"   → Accuracy: {acc:.4f}")
            print(f"   → Loss: {loss:.4f}")
            print(f"   → Score: {score:.4f}")

            if states[round_id] == 'backdoor':
                b_acc = evaluate_model_lightning(agg_model, backdoor_data_loader)
                print(f"   → Backdoor Accuracy: {b_acc:.4f}")

            if score > best_score_this_round:
                best_model_this_round = deepcopy(agg_model)
                best_score_this_round = score
                best_agg_id = agg_id
                contributors = contributing_nodes

            # Clean memory
            del agg_model
            torch.cuda.empty_cache()
            gc.collect()

        # =======Fail-safe mechanism=======

        # Fail-safe hyperparam
        alpha = .95

        best_score_prev = alpha * best_score_prev
        if best_score_this_round > best_score_prev:
            model = best_model_this_round
            best_score_prev = best_score_this_round
            print(" Model updated.")
            print(f" Best Aggregated Model (Round {round_id}): id {best_agg_id}")
            print(f"   → Score: {best_score_this_round:.4f}")
            print(f"   → Contributors: {contributors}")
        else:
            print(" FAIL-SAFE TRIGGERED: Model not updated.")

        # Final cleanup
        if best_model_this_round is not None:
            del best_model_this_round
            torch.cuda.empty_cache()
            gc.collect()

    # Restore logging level
   logger.setLevel(old_level)


# In[ ]:

configs= ['cifar10', 'cifar100', 'fashionmnist', 'femnist']
for config in configs:
    FEDBAGSAFE(config, 100, 100, mode=False)

