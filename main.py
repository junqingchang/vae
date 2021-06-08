import torch
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.optim as optim


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_EPOCHS = 30
LEARNING_RATE = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.1
BATCH_SIZE = 8


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 16, 5)
        self.fc1 = nn.Linear(16*20*20, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc_mu = nn.Linear(84, 64)
        self.fc_var = nn.Linear(84, 64)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        mu = self.fc_mu(x)
        var = self.fc_var(x)
        return mu, var

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(64, 84)
        self.fc2 = nn.Linear(84, 120)
        self.fc3 = nn.Linear(120, 16*20*20)
        self.conv1 = nn.ConvTranspose2d(16, 16, 5)
        self.conv2 = nn.ConvTranspose2d(16, 6, 5)
        self.conv3 = nn.ConvTranspose2d(6, 3, 5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = x.view((BATCH_SIZE, 16, 20, 20))
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

def train(trainloader, encoder, decoder, enc_optimizer, dec_optimizer, criterion, device):
    encoder.train()
    decoder.train()
    losses = 0
    for idx, (img, _) in enumerate(trainloader):
        img = img.to(device)
        enc_optimizer.zero_grad()
        dec_optimizer.zero_grad()
        mu, log_var = encoder(img)
        z = sample_z(mu, log_var)
        recon_img = decoder(z)

        loss = criterion(img, recon_img, mu, log_var)
        loss.backward()
        enc_optimizer.step()
        dec_optimizer.step()
        losses += loss.item()
    return losses/len(trainloader)

def vae_loss(y_true, y_pred, mu, log_var):
    mse = nn.MSELoss()
    recon_loss = torch.sum(mse(y_pred, y_true))
    kld = 0.5 * torch.sum(torch.exp(log_var) + mu ** 2 - 1 - log_var)
    return recon_loss + kld

def sample_z(mu, log_var):
    std = torch.exp(log_var / 2)
    q = torch.distributions.Normal(mu, std)
    z = q.rsample()
    return z

def sample_recon(trainloader, encoder, decoder, epoch, device):
    encoder.eval()
    decoder.eval()
    dataiter = iter(trainloader)
    img, _ = dataiter.next()
    grid = torchvision.utils.make_grid(img)
    writer.add_image('images', grid, epoch)
    img = img.to(device)
    mu, log_var = encoder(img)
    z = sample_z(mu, log_var)
    recon_img = decoder(z)
    grid = torchvision.utils.make_grid(recon_img.cpu().detach())
    writer.add_image('recon_images', grid, epoch)

if __name__ == '__main__':
    transform = torchvision.transforms.ToTensor()
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=False)

    writer = SummaryWriter()

    encoder = Encoder()
    encoder.to(DEVICE)
    decoder = Decoder()
    decoder.to(DEVICE)

    enc_optimizer = optim.SGD(encoder.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    dec_optimizer = optim.SGD(decoder.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    criterion = vae_loss

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train(trainloader, encoder, decoder, enc_optimizer, dec_optimizer, criterion, DEVICE)
        writer.add_scalar('Loss/train', train_loss, epoch)
        sample_recon(trainloader, encoder, decoder, epoch, DEVICE)
    