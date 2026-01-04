

import torch

import torch.nn as nn

import torch.optim as optim

from torchvision import transforms, datasets

from torch.utils.data import DataLoader




transform_train = transforms.Compose([
    
    transforms.Resize((32,32)),

    transforms.RandomHorizontalFlip(p = 0.5),

    transforms.RandomRotation(15),

    transforms.ColorJitter(

        brightness=0.2,
        contrast= 0.2,
        saturation= 0.2
    ),

    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
    
])


transform_validation = transforms.Compose([

    transforms.Resize((32,32)),
    transforms.ToTensor(),

    transforms.Normalize([0.5]*3, [0.5]*3)
])


train_dataset = datasets.ImageFolder(

    root="Dataset/train",
    transform= transform_train
)

validation_dataset = datasets.ImageFolder(

    root= "Dataset/val",
    transform= transform_validation
)




train_loader = DataLoader(

    train_dataset,

    batch_size= 8,

    shuffle= True

)


validation_loader = DataLoader(

    validation_dataset,

    batch_size= 8,

    shuffle= True
)


class SimpleConvolutionalNetwork(nn.Module):

    def __init__(self):
        super(SimpleConvolutionalNetwork, self).__init__()

        self.conv1 = nn.Conv2d(

            in_channels = 3,

            out_channels= 4,

            kernel_size= 3
        )

        self.conv2 = nn.Conv2d(

            in_channels = 4,

            out_channels= 8,

            kernel_size= 3
        )

        self.conv3 = nn.Conv2d(

            in_channels= 8,

            out_channels= 16,

            kernel_size= 3
        )

        self.fc1 = nn.Linear(16 * 2* 2, 64)

        self.dropout = nn.Dropout(p = 0.12)

        self.fc2 = nn.Linear(64, 2)

        self.pool = nn.MaxPool2d(2, 2)


    def forward(self, x):

        x = torch.relu(self.conv1(x))

        x = self.pool(x)


        x = torch.relu(self.conv2(x))
        x = self.pool(x)

        x = torch.relu(self.conv3(x))
        x = self.pool(x)

        x = torch.flatten(x,1)

        x = torch.relu(self.fc1(x))

        x = self.dropout(x)

        x = self.fc2(x)

        return x
    

modeloCNN = SimpleConvolutionalNetwork()

otimizador = optim.Adam( modeloCNN.parameters(), lr = 0.001)

criterio = nn.CrossEntropyLoss()


epochs = 61


for epoch in range(epochs):

    running_loss = 0.0

    for images, labels in train_loader:

        otimizador.zero_grad()

        outputs = modeloCNN(images)

        loss = criterio(outputs, labels)

        loss.backward()

        otimizador.step()
        
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)

    print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_loss:.4f}")


modeloCNN.eval()
correct = 0
total = 0
val_loss = 0.0

with torch.no_grad():

    for images, labels in validation_loader:

        outputs = modeloCNN(images)

        loss = criterio(outputs, labels)

        val_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

val_loss /= len(validation_loader)
accuracy = 100 * correct / total

print(f"Validation Loss: {val_loss:.4f} | Accuracy: {accuracy:.2f}%")