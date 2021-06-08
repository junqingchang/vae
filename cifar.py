import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.optim as optim


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
BATCH_SIZE = 100


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(3*32*32, 400)
        self.fc2 = nn.Linear(400, 400)
        self.fc_mu = nn.Linear(400, 200)
        self.fc_var = nn.Linear(400, 200)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        mu = self.fc_mu(x)
        var = self.fc_var(x)
        return mu, var

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(200, 400)
        self.fc2 = nn.Linear(400, 400)
        self.fc3 = nn.Linear(400, 3*32*32)
        self.relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

def train(trainloader, encoder, decoder, enc_optimizer, dec_optimizer, criterion, device):
    encoder.train()
    decoder.train()
    losses = 0
    recon_losses = 0
    kld_losses = 0
    for idx, (img, _) in enumerate(trainloader):
        img = img.to(device)
        x = img.view((BATCH_SIZE, 3*32*32))
        enc_optimizer.zero_grad()
        dec_optimizer.zero_grad()
        mu, log_var = encoder(x)
        z = sample_z(mu, torch.exp(0.5 * log_var))
        recon_x = decoder(z)
        loss, recon_loss, kld = criterion(x, recon_x, mu, log_var)
        losses += loss.item()
        recon_losses += recon_loss.item()
        kld_losses += kld.item()
        loss.backward()
        enc_optimizer.step()
        dec_optimizer.step()
    return losses/len(trainloader), recon_losses/len(trainloader), kld_losses/len(trainloader)

def vae_loss(y_true, y_pred, mu, log_var):
    bce = nn.BCELoss(reduction='sum')
    recon_loss = bce(y_pred, y_true)
    kld = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kld, recon_loss, kld

def sample_z(mu, var):
    eps = torch.randn_like(var).to(DEVICE)
    z = mu + var * eps
    return z

def sample_recon(trainloader, encoder, decoder, epoch, device):
    encoder.eval()
    decoder.eval()
    dataiter = iter(trainloader)
    img, _ = dataiter.next()
    grid = torchvision.utils.make_grid(img)
    writer.add_image('images', grid, epoch)
    img = img.to(device)
    x = img.view((8, 3*32*32))
    with torch.no_grad():
        mu, log_var = encoder(x)
        z = sample_z(mu, torch.exp(0.5 * log_var))
        recon_x = decoder(z)
        recon_img = recon_x.view((8, 3, 32, 32))
    recon_grid = torchvision.utils.make_grid(recon_img.cpu().detach())
    writer.add_image('recon_images', recon_grid, epoch)

if __name__ == '__main__':
    transform = torchvision.transforms.ToTensor()
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader= DataLoader(testset, batch_size=8, shuffle=False)

    writer = SummaryWriter()

    encoder = Encoder()
    encoder.to(DEVICE)
    decoder = Decoder()
    decoder.to(DEVICE)

    enc_optimizer = optim.Adam(encoder.parameters(), lr=LEARNING_RATE)
    dec_optimizer = optim.Adam(decoder.parameters(), lr=LEARNING_RATE)

    criterion = vae_loss

    for epoch in range(1, NUM_EPOCHS + 1):
        trainloss, recon_loss, kld = train(trainloader, encoder, decoder, enc_optimizer, dec_optimizer, criterion, DEVICE)
        writer.add_scalar('Loss/train', trainloss, epoch)
        writer.add_scalar('Loss/recon', recon_loss, epoch)
        writer.add_scalar('Loss/kld', kld, epoch)
        sample_recon(testloader, encoder, decoder, epoch, DEVICE)
    