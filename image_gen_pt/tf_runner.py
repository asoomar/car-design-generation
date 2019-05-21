
import torch
from PIL import Image
from torch.nn import Parameter
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import tensorflow as tf
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

def gan_estimator():
    sess = tf.Session()

    # Set up palceholders
    A = tf.placeholder(tf.float32, shape=(hparams.n_input, hparams.num_measurements), name='A')
    y_batch = tf.placeholder(tf.float32, shape=(hparams.batch_size, hparams.num_measurements), name='y_batch')

    # Create the generator
    # TODO update car_model_def to new version
    z_batch = tf.Variable(tf.random_normal([1, 100]), name='z_batch')
    x_hat_batch, restore_dict_gen, restore_path_gen = car_model_def.dcgan_gen(z_batch, hparams)

    z = torch.randn()

    # Create the discriminator
    # TODO update car_model_def to new version
    prob, restore_dict_discrim, restore_path_discrim = car_model_def.dcgan_discrim(x_hat_batch, hparams)

    # measure the estimate
    if hparams.measurement_type == 'project':
        y_hat_batch = tf.identity(x_hat_batch, name='y2_batch')
    else:
        measurement_is_sparse = (hparams.measurement_type in ['inpaint', 'superres'])
        y_hat_batch = tf.matmul(x_hat_batch, A, b_is_sparse=measurement_is_sparse, name='y2_batch')

    # define all losses
    m_loss1_batch = tf.reduce_mean(tf.abs(y_batch - y_hat_batch), 1)
    m_loss2_batch = tf.reduce_mean((y_batch - y_hat_batch) ** 2, 1)
    zp_loss_batch = tf.reduce_sum(z_batch ** 2, 1)
    d_loss1_batch = -tf.log(prob)
    d_loss2_batch = tf.log(1 - prob)

    # define total loss
    total_loss = tf.reduce_mean(m_loss2_batch)

    # Set up gradient descent
    var_list = [z_batch]
    global_step = tf.Variable(0, trainable=False, name='global_step')
    learning_rate = utils.get_learning_rate(global_step, hparams)
    with tf.variable_scope(tf.get_variable_scope(), reuse=False):
        opt = utils.get_optimizer(learning_rate, hparams)
        update_op = opt.minimize(total_loss, var_list=var_list, global_step=global_step, name='update_op')
    opt_reinit_op = utils.get_opt_reinit_op(opt, var_list, global_step)

    # Intialize and restore model parameters
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    restorer_gen = tf.train.Saver(var_list=restore_dict_gen)
    restorer_discrim = tf.train.Saver(var_list=restore_dict_discrim)
    restorer_gen.restore(sess, restore_path_gen)
    restorer_discrim.restore(sess, restore_path_discrim)

    def estimator(A_val, y_batch_val, hparams):
        """Function that returns the estimated image"""
        best_keeper = utils.BestKeeper(hparams)

        if hparams.measurement_type == 'project':
            feed_dict = {y_batch: y_batch_val}
        else:
            feed_dict = {A: A_val, y_batch: y_batch_val}

        for i in range(hparams.num_random_restarts):
            sess.run(opt_reinit_op)
            for j in range(hparams.max_update_iter):

                _, lr_val, total_loss_val, \
                m_loss1_val, \
                m_loss2_val, \
                zp_loss_val, \
                d_loss1_val, \
                d_loss2_val = sess.run([update_op, learning_rate, total_loss,
                                        m_loss1,
                                        m_loss2,
                                        zp_loss,
                                        d_loss1,
                                        d_loss2], feed_dict=feed_dict)
                logging_format = 'rr {} iter {} lr {} total_loss {} m_loss1 {} m_loss2 {} zp_loss {} d_loss1 {} d_loss2 {}'
                print
                logging_format.format(i, j, lr_val, total_loss_val,
                                      m_loss1_val,
                                      m_loss2_val,
                                      zp_loss_val,
                                      d_loss1_val,
                                      d_loss2_val)

            x_hat_batch_val, total_loss_batch_val = sess.run([x_hat_batch, total_loss_batch], feed_dict=feed_dict)
            best_keeper.report(x_hat_batch_val, total_loss_batch_val)
        return best_keeper.get_best()

    return estimator


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
    num_restarts = 10000
    num_updates = 2
    min_loss = 1e10
    best_z = 0
    for i in range(num_restarts):

        #generate vector to pass to model
        device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
        z = torch.randn(1, nz, 1, 1, device=device, requires_grad=True)
        learning_rate = .1
        optimizer = getOptimizer(optimizer_choice, [z], learning_rate)
        gen_image = gan.generate(z)[0]
        loss = loss_fn(gen_image, actual_image)
        if(loss.item() < min_loss):
            min_loss = loss.item()
            best_z = z

        #inner loop to run gradient descent on the z vector
        # for j in range(num_updates):
        #     gen_image = gan.generate(z)
        #     show_image(gen_image)
        #     loss = loss_fn(gen_image, actual_image)
        #     print("Loss for run " + str(i) + " round " + str(j) + " : " + str(loss.item()))
        #
        #     if loss.item() < min_loss:
        #         min_loss = loss.item()
        #         best_z = z
        #     loss = Variable(loss, requires_grad=True)
        #
        #
        #     optimizer.zero_grad()
        #     z.retain_grad()
        #     loss.backward()
        #     print(z.grad)
        #     optimizer.step()

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