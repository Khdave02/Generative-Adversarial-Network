import torch
import loader
import GANclass
import train
import utilities

lr = 2*1e-4
num_epochs=5
batch_size = 64
train_loader = loader.trainLoader(batch_size)
test_loader = loader.testLoader(batch_size)
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

model = GANclass.gan(device,lr, batch_size).to(device)
model.apply(utilities.weights_init)

g_optimizer = torch.optim.Adam(model.generator.parameters(),lr = lr)
d_optimizer = torch.optim.Adam(model.discriminator.parameters(),lr = lr)

train_outputs, d_loss, g_loss= train.training(model, train_loader,num_epochs,g_optimizer,d_optimizer)
# test_outputs, test_loss= testing(model,test_loader,num_epochs)


# test_outputs, test_loss= test.testing(model,test_loader,num_epochs)
# utilities.view_images(test_outputs,10)
utilities.cost_graph(d_loss,"d_loss")
utilities.cost_graph(g_loss,"g_loss")
# utilities.cost_graph(test_loss,"test_loss")

