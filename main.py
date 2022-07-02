import torch
import loader
import GANclass
import train
import utilities

lr = 2*1e-4
num_epochs=1
batch_size = 64
train_loader = loader.trainLoader(batch_size)
test_loader = loader.testLoader(batch_size)
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

model = GANclass.gan(device,lr, batch_size).to(device)
model.apply(utilities.weights_init)

g_optimizer = torch.optim.Adam(model.generator.parameters(),lr = lr)
d_optimizer = torch.optim.Adam(model.discriminator.parameters(),lr = lr)

train_outputs, d_loss, g_loss= train.training(model, train_loader,num_epochs,g_optimizer,d_optimizer)

utilities.cost_graph(g_loss,d_loss)


