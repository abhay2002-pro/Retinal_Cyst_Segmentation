import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from models.Model import ModelNet
from losses.triplet_loss import TripletLoss
from data_loader import DatasetClass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 1
learning_rate = 0.001
num_epochs = 5

triplet_margin = 1.0

transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = DatasetClass(root="./data", transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = ModelNet()
model.to(device)

criterion_triplet = TripletLoss(margin=triplet_margin)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for batch in train_dataloader:
        img_a, img_b, segmented_a = batch  
        img_a, img_b, segmented_a = img_a.to(device), img_b.to(device)

        optimizer.zero_grad()

        output_a, output_b = model(img_a, img_b, softmax_out=True)

        loss_triplet = criterion_triplet(output_a, segmented_a, output_b)

        loss = loss_triplet

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss}")

torch.save(model.state_dict(), "Modelnet.pth")
