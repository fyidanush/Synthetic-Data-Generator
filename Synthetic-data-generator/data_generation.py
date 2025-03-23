import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

def train_gan(real_data, epochs=1000):
    input_dim = real_data.shape[1]
    output_dim = real_data.shape[1]

    generator = Generator(input_dim, output_dim)
    discriminator = Discriminator(input_dim)

    optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0002)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002)

    criterion = nn.BCELoss()

    for epoch in range(epochs):
        # Train discriminator
        optimizer_d.zero_grad()
        real_labels = torch.ones(real_data.shape[0], 1)
        fake_labels = torch.zeros(real_data.shape[0], 1)

        real_output = discriminator(real_data)
        d_loss_real = criterion(real_output, real_labels)

        noise = torch.randn(real_data.shape[0], input_dim)
        fake_data = generator(noise)
        fake_output = discriminator(fake_data.detach())
        d_loss_fake = criterion(fake_output, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_d.step()

        # Train generator
        optimizer_g.zero_grad()
        fake_output = discriminator(fake_data)
        g_loss = criterion(fake_output, real_labels)
        g_loss.backward()
        optimizer_g.step()

    return generator