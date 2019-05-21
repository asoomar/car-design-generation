
import torch
from PIL import Image
from torch.nn import Parameter
import torchvision.transforms as transforms
import torchvision.utils as vutils
import random
from torch.autograd import Variable
import gan
import matplotlib.pyplot as plt


nz = 100
ngpu = 0

def load_image(img_path):
    img = Image.open(img_path)
    trans = transforms.ToTensor()
    return trans(img)

def show_image(img):
    print(type(img))
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


def main(img_path, optimizer_choice, name):
    print("running program")
    ##initialize gan model to generate stuff


    #define loss function
    loss_fn = torch.nn.MSELoss(reduction='mean')

    #load in actual image from file path into tensor
    actual_image = load_image(img_path)
    show_image(actual_image)
    #FOR TESTING

    #
    #
    # # print("shape of actual image: ")
    # # print(actual_image.size())
    # # show_image(actual_image)
    #
    # device1 = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    #
    # z = torch.randn(1, nz, 1, 1, device=device1, requires_grad=True)
    # z_hat = z.clone()
    # optimizer = torch.optim.Adam([z], .0005)
    #
    # #print(torch.eq(z, y).all().item()) # 1 means same, 0 means different
    # gen_image = gan.generate(z)[0]
    # # print("shape of generated image: ")
    # # print(gen_image.size())
    # # show_image(gen_image)
    #
    # loss = loss_fn(gen_image, actual_image)
    # loss = Variable(loss, requires_grad=True)
    #
    # print("LOSS INFO")
    # print(loss.item())
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()
    #
    # if torch.eq(z, z_hat).all().item() == 1:
    #     print("Tensors are same")
    # else:
    #     print("tensors are diff")


    #TESTING




    #outer loop -> defines the number of restarts
    num_restarts = 3
    num_updates = 5000
    learning_rate = .005
    global_min_loss = 1e10
    best_z = []
    for i in range(num_restarts):

        #generate vector to pass to model
        device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
        #gnerate vector here
        z = torch.randn(1, nz, 1, 1, device=device, requires_grad=False)
        local_min_loss = 1e10
        for j in range(num_updates):
            update_index = random.randint(0, 99)
            to_mod = learning_rate * random.randint(1,5) * 10

            np_arr1 = z.numpy()
            np_arr1[:, update_index,:,:] = np_arr1[:, update_index,:,:] + to_mod
            new_tensor_add = torch.tensor(np_arr1, device=device)
            add_gen = gan.generate(new_tensor_add)[0]
            add_loss = loss_fn(add_gen, actual_image)

            np_arr2 = z.numpy()
            np_arr2[:, update_index, :, :] = np_arr2[:, update_index, :, :] - to_mod
            new_tensor_sub = torch.tensor(np_arr2, device=device)
            sub_gen = gan.generate(new_tensor_sub)[0]
            sub_loss = loss_fn(sub_gen, actual_image)

            if add_loss.item() < local_min_loss:
                z = new_tensor_add
                local_min_loss = add_loss.item()
            elif sub_loss.item() < local_min_loss:
                z = new_tensor_sub
                local_min_loss = sub_loss.item()
            print("Restart #" + str(i) + " run #" + str(j) + " local_loss: " + str(local_min_loss))

        if local_min_loss < global_min_loss:
            best_z = z
            global_min_loss = local_min_loss

        print("End of round " + str(i) + ": global loss is " + str(global_min_loss))
    best_image = gan.generate(best_z)[0]
    show_image(best_image)
    # save_img(best_image, "./generated_images/" + name)










if __name__ == '__main__':
    #path_to_image = input("enter relative path to image")
    path_to_image = "./test_images/formatted/test_image.jpg"
    print("image path: " + path_to_image)
    name = path_to_image.split("/")[-1]
    print('name: ' + name)
    optimizer = "Adam"
    #optimizer = input("choose optimizer from : Adam, AdaGrad, etc :: Note: only Adam currently implemented ")
    main(img_path=path_to_image, optimizer_choice=optimizer, name=name)