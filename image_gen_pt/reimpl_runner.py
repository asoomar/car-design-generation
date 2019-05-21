from __future__ import print_function
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
import torchvision.utils as vutils
from torch.autograd import Variable
import gan

nz = 100
ngpu = 0
learning_rate = .05
beta1 = .5
num_updates = 200

def load_image(img_path):
    img = Image.open(img_path)
    trans = transforms.ToTensor()
    return trans(img)

def show_image(img):
    #print(type(img))
    trans = transforms.ToPILImage()
    out = trans(img)
    out.show()

def save_img(img, location):
    print("Saving image")
    vutils.save_image(img, location, normalize=True)


def getOptimizer(optimizer_choice, z, learning_rate):
    if optimizer_choice == 'Adam':
        return torch.optim.Adam(z, learning_rate)
    return torch.optim.Adam(z, learning_rate)


def recover(actual_z):

    mse_loss = nn.MSELoss()
    #mse_loss_ = nn.MSELoss()

    #z_approx = torch.FloatTensor(1, nz, 1, 1).uniform_(-1, 1)
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    torch.manual_seed = random.randint(1, 10000)
    z_approx = torch.randn(1, nz, 1, 1, device=device)

    z_approx = Variable(z_approx)
    z_approx.requires_grad = True

    optimizer = optim.Adam([z_approx], lr=learning_rate, betas=(.5,.999))
    show_image(gan.generate(z_approx)[0])
    for i in range(num_updates):
        g_z_approx = gan.generate(z_approx)[0]
        assert g_z_approx.shape == actual_z.shape
        mse_g_z = mse_loss(g_z_approx, actual_z)

        #mse_z = mse_loss_(z_approx, z)
        if i % 15 == 0:
            print("[Iter {}] MSE: {}"
                  .format(i, mse_g_z.item()))
            show_image(g_z_approx)

            # bprop
        optimizer.zero_grad()
        mse_g_z.backward()
        optimizer.step()

    #best_image = gan.generate(g_z_approx)[0]
    show_image(gan.generate(z_approx)[0])
    return z_approx



def main(img_path, optimizer_choice, name):

    gan.initNetG()



    actual_z = load_image(img_path=img_path)
    show_image(actual_z)

    #recover z_approx
    optimized_z = recover(actual_z=actual_z)
    #print(optimized_z)








if __name__ == '__main__':
    #path_to_image = input("enter relative path to image")
    path_to_image = "./test_images/formatted/test_image.jpg"
    print("image path: " + path_to_image)
    name = path_to_image.split("/")[-1]
    print('name: ' + name)
    optimizer = "Adam"
    #optimizer = input("choose optimizer from : Adam, AdaGrad, etc :: Note: only Adam currently implemented ")
    main(img_path=path_to_image, optimizer_choice=optimizer, name=name)