from __future__ import print_function
import os
import random
import torch
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
# from torch.autograd import Variable
import argparse
#import functions
import networks.TFVAEGAN_model as model
import datasets.image_util as util
import classifiers.classifier_images as classifier
from utils.logger import init_loggers
from utils.options import parse 

def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')
    args = parser.parse_args()
    opt = parse(args.opt)

    return opt

opt = parse_options()

# making directory for logger
os.makedirs(opt['log'],exist_ok=True)

# initialize loggers and variables
logger= init_loggers(opt)
dataset_name = opt["datasets"]["name"]
manual_seed = opt['manual_seed']
att_size = opt["network"]["gan"]["att_size"]
res_size = opt["network"]["gan"]["res_size"]
batch_size = opt["train"]["batch_size"]
num_class = opt["network"]["gan"]["num_class"]
lambda_gan = opt["network"]["gan"]["lambda"]
recons_weight = opt["network"]["decoder"]["recons_weight"]
gamma_d = opt["network"]["gan"]["gamma_d"]
gamma_g = opt["network"]["gan"]["gamma_g"]
a1_feedback = opt["network"]["feedback"]["a1"]
num_epoch = opt["train"]["num_epoch"]
critic_iter = opt["network"]["gan"]["critic_iter"]
noise_gan = opt["network"]["gan"]["noise"]
latent_dim = opt["network"]["gan"]["latent_dim"]
syn_num = opt["network"]["gan"]["syn_num"]
lr_classifier = opt["network"]["classifier"]["lr"]
gzsl = opt["network"]["classifier"]["gzsl"]
feedback_loop = opt["network"]["feedback"]["feedback_loop"]
a2 = opt["network"]["feedback"]["a2"]

logger.info(f"Random Seed: {manual_seed}")
random.seed(manual_seed)
torch.manual_seed(manual_seed)
cuda = False
if torch.cuda.is_available():
    cuda = True
    torch.cuda.manual_seed_all(manual_seed)
cudnn.benchmark = True
# load data
data = util.DATA_LOADER(opt)
logger.info(f"# of training samples: {data.ntrain}")

netE = model.Encoder(opt)
netG = model.Generator(opt)
netD = model.Discriminator_D1(opt)
# Init models: Feedback module, auxillary module
netF = model.Feedback(opt)
netDec = model.AttDec(opt,att_size)

print(netE)
print(netG)
print(netD)
print(netF)
print(netDec)

###########
# Init Tensors
input_res = torch.FloatTensor(batch_size, res_size)
input_att = torch.FloatTensor(batch_size, att_size) # att_size
# class-embedding size
noise = torch.FloatTensor(batch_size, att_size)
one = torch.tensor(1, dtype=torch.float)

mone = one * -1
##########
# Cuda
if cuda:
    netD.cuda()
    netE.cuda()
    netF.cuda()
    netG.cuda()
    netDec.cuda()
    input_res = input_res.cuda()
    noise, input_att = noise.cuda(), input_att.cuda()
    one = one.cuda()
    mone = mone.cuda()

def loss_fn(recon_x, x, mean, log_var):
    '''
    Binary Cross Entropy loss + KL Divergence loss
    '''
    BCE = torch.nn.functional.binary_cross_entropy(recon_x+1e-12, x.detach(),reduction='sum')
    BCE = BCE.sum()/ x.size(0)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())/ x.size(0)
    return (BCE + KLD)
           
def sample():
    batch_feature, batch_att = data.next_seen_batch(batch_size)
    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)

def WeightedL1(pred, gt):
    '''
    WGAN loss
    '''
    wt = (pred-gt).pow(2)
    wt /= wt.sum(1).sqrt().unsqueeze(1).expand(wt.size(0),wt.size(1))
    loss = wt * (pred-gt).abs()
    return loss.sum()/loss.size(0)
    
def generate_syn_feature(generator,classes, attribute,num,netF=None,netDec=None):
    
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass*num, res_size)
    syn_label = torch.LongTensor(nclass*num) 
    syn_att = torch.FloatTensor(num, att_size)
    syn_noise = torch.FloatTensor(num, att_size) # replaced nz with att_size

    if cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()
    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]
        syn_att.copy_(iclass_att.repeat(num, 1))
        syn_noise.normal_(0, 1)
        # syn_noisev = Variable(syn_noise,volatile=True)
        syn_noisev = syn_noise
        # syn_attv = Variable(syn_att,volatile=True)
        syn_attv = syn_att
        fake = generator(syn_noisev,c=syn_attv)
        if netF is not None:
            dec_out = netDec(fake) # only to call the forward function of decoder
            dec_hidden_feat = netDec.getLayersOutDet() #no detach layers
            feedback_out = netF(dec_hidden_feat)
            fake = generator(syn_noisev, a1=a2, c=syn_attv, feedback_layers=feedback_out)
        output = fake
        syn_feature.narrow(0, i*num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i*num, num).fill_(iclass)

    return syn_feature, syn_label

lr_gan = opt["network"]["gan"]["lr"]
betas = opt["train"]["betas"]

optimizer = optim.Adam(netE.parameters(), lr=lr_gan)
optimizerD = optim.Adam(netD.parameters(), lr=lr_gan,betas=betas)
optimizerG = optim.Adam(netG.parameters(), lr=lr_gan,betas=betas)
optimizerF = optim.Adam(netF.parameters(), lr=opt["network"]["feedback"]["lr"], betas=betas)
optimizerDec = optim.Adam(netDec.parameters(), lr=opt["network"]["decoder"]["lr"], betas=betas)


def calc_gradient_penalty(netD,real_data, fake_data, input_att, lambda_gan):
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(real_data.size())
    if cuda:
        alpha = alpha.cuda()
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    if cuda:
        interpolates = interpolates.cuda()
    # interpolates = Variable(interpolates, requires_grad=True)
    interpolates.requires_grad_(True)
    # disc_interpolates = netD(interpolates, Variable(input_att))
    disc_interpolates = netD(interpolates, input_att)
    ones = torch.ones(disc_interpolates.size())
    if cuda:
        ones = ones.cuda()
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gan
    return gradient_penalty

best_gzsl_acc = 0
best_zsl_acc = 0

logger.info(f'Start training from epoch: {0}, iter: {0}')
for epoch in range(0,num_epoch):
    for loop in range(0,feedback_loop):
        for i in range(0, data.ntrain, batch_size):
            ######### Discriminator training ##############
            for p in netD.parameters(): #unfreeze discrimator
                p.requires_grad = True

            for p in netDec.parameters(): #unfreeze deocder
                p.requires_grad = True
            # Train D1 and Decoder (and Decoder Discriminator)
            gp_sum = 0 #lAMBDA VARIABLE
            for iter_d in range(critic_iter):
                sample()
                netD.zero_grad()          
                # input_resv = Variable(input_res)
                input_resv = input_res
                # input_attv = Variable(input_att)
                input_attv = input_att

                netDec.zero_grad()
                recons = netDec(input_resv)
                R_cost = recons_weight*WeightedL1(recons, input_attv) 
                R_cost.backward()
                optimizerDec.step()
                criticD_real = netD(input_resv, input_attv) # input_attv: semantic embeddings
                criticD_real = gamma_d*criticD_real.mean()
                criticD_real.backward(mone)
                if noise_gan:
                    # Encoder phase   
                    # input_resv: visual features x
                    # input_attv: semantic embeddings a
                    means, log_var = netE(input_resv, input_attv)
                    std = torch.exp(0.5 * log_var)
                    eps = torch.randn([batch_size, latent_dim]).cpu()
                    if cuda:
                        eps = eps.cuda()
                    # Latent code z
                    z = eps * std + means #torch.Size([64, 312])
                else:
                    noise.normal_(0, 1)
                    # z = Variable(noise)
                    z = noise

                #  Generator Phase
                if loop == 1:
                    fake = netG(z, c=input_attv)
                    dec_out = netDec(fake)
                    dec_hidden_feat = netDec.getLayersOutDet()
                    feedback_out = netF(dec_hidden_feat) # taking feedback from the feedback module
                    fake = netG(z, a1=a1_feedback, c=input_attv, feedback_layers=feedback_out)
                else:
                    fake = netG(z, c=input_attv)

                criticD_fake = netD(fake.detach(), input_attv)
                criticD_fake = gamma_d*criticD_fake.mean()
                criticD_fake.backward(one)
                
                # gradient penalty
                gradient_penalty = gamma_d*calc_gradient_penalty(netD, input_res, fake.data, input_att, lambda_gan)
                # if opt.lambda_mult == 1.1:
                gp_sum += gradient_penalty.data
                gradient_penalty.backward()         
                Wasserstein_D = criticD_real - criticD_fake
                D_cost = criticD_fake - criticD_real + gradient_penalty #add Y here and #add vae reconstruction loss
                optimizerD.step()

            gp_sum /= (gamma_d * lambda_gan * critic_iter)
            if (gp_sum > 1.05).sum() > 0:
                lambda_gan *= 1.1
            elif (gp_sum < 1.001).sum() > 0:
                lambda_gan /= 1.1

            #############Generator training ##############
            # Train Generator and Decoder
            for p in netD.parameters(): #freeze discrimator
                p.requires_grad = False
            if recons_weight > 0:
                for p in netDec.parameters(): #freeze decoder
                    p.requires_grad = False

            netE.zero_grad()
            netG.zero_grad()
            netF.zero_grad()
            # input_resv = Variable(input_res)
            # input_attv = Variable(input_att)
            input_resv = input_res
            input_attv = input_att
            means, log_var = netE(input_resv, input_attv)
            std = torch.exp(0.5 * log_var)
            eps = torch.randn([batch_size, latent_dim]).cpu()
            if cuda:
                eps = eps.cuda()
            z = eps * std + means #torch.Size([64, 312])
            if loop == 1:
                recon_x = netG(z, c=input_attv)
                dec_out = netDec(recon_x)
                dec_hidden_feat = netDec.getLayersOutDet()
                feedback_out = netF(dec_hidden_feat)
                recon_x = netG(z, a1=a1_feedback, c=input_attv, feedback_layers=feedback_out)
            else:
                recon_x = netG(z, c=input_attv)

            vae_loss_seen = loss_fn(recon_x, input_resv, means, log_var) # minimize E 3 with this setting feedback will update the loss as well
            errG = vae_loss_seen
            
            if noise_gan:
                criticG_fake = netD(recon_x,input_attv).mean()
                fake = recon_x 
            else:
                noise.normal_(0, 1)
                # noisev = Variable(noise)
                noisev = noise
                if loop == 1:
                    fake = netG(noisev, c=input_attv)
                    dec_out = netDec(recon_x) #Feedback from Decoder encoded output
                    dec_hidden_feat = netDec.getLayersOutDet()
                    feedback_out = netF(dec_hidden_feat) # taking feedback from the feedback module
                    fake = netG(noisev, a1=a1_feedback, c=input_attv, feedback_layers=feedback_out)
                else:
                    fake = netG(noisev, c=input_attv)
                criticG_fake = netD(fake,input_attv).mean()

            G_cost = -criticG_fake
            errG += gamma_g * G_cost
            netDec.zero_grad()
            recons_fake = netDec(fake)
            R_cost = WeightedL1(recons_fake, input_attv)
            errG += recons_weight * R_cost
            errG.backward()
            # write a condition here
            optimizer.step()
            optimizerG.step()
            if loop == 1:
                optimizerF.step()
            if recons_weight > 0: # not train decoder at feedback time
                optimizerDec.step() 

    logger.info('[%d/%d]  Loss_D: %.4f Loss_G: %.4f, Wasserstein_dist:%.4f, vae_loss_seen:%.4f'% (epoch, num_epoch, D_cost.item(), G_cost.item(), Wasserstein_D.item(),vae_loss_seen.item()))

    netG.eval()
    netDec.eval()
    netF.eval()

    with torch.no_grad():
        syn_feature, syn_label = generate_syn_feature(netG,data.unseenclasses, data.attribute, syn_num,netF=netF,netDec=netDec)
    
    # Generalized zero-shot learning
    if gzsl:   
        # Concatenate real seen features with synthesized unseen features
        train_X = torch.cat((data.train_feature, syn_feature), 0)
        train_Y = torch.cat((data.train_label, syn_label), 0)
        nclass = num_class

        # Train GZSL classifier
        gzsl_cls = classifier.CLASSIFIER(train_X, train_Y, data, nclass, cuda, lr_classifier, 0.5, \
                25, syn_num, generalized=True, netDec=netDec, dec_size=att_size, dec_hidden_size=4096)
        if best_gzsl_acc < gzsl_cls.H:
            best_acc_seen, best_acc_unseen, best_gzsl_acc = gzsl_cls.acc_seen, gzsl_cls.acc_unseen, gzsl_cls.H
        logger.info('GZSL: seen=%.4f, unseen=%.4f, h=%.4f' % (gzsl_cls.acc_seen, gzsl_cls.acc_unseen, gzsl_cls.H))

    # Zero-shot learning
    # Train ZSL classifier
    zsl_cls = classifier.CLASSIFIER(syn_feature, util.map_label(syn_label, data.unseenclasses), \
                    data, data.unseenclasses.size(0), cuda, lr_classifier, 0.5, 25, syn_num, \
                    generalized=False, netDec=netDec, dec_size=att_size, dec_hidden_size=4096)
    acc = zsl_cls.acc
    if best_zsl_acc < acc:
        best_zsl_acc = acc
    logger.info('ZSL: unseen accuracy=%.4f' % (acc))

    # reset G to training mode
    netG.train()
    netDec.train()
    netF.train()

logger.info('End of training.')

logger.info(f'Dataset {dataset_name}')
logger.info(f'the best ZSL unseen accuracy is {best_zsl_acc}')
if gzsl:
    logger.info(f'Dataset {dataset_name}')
    logger.info(f'the best GZSL seen accuracy is {best_acc_seen}')
    logger.info(f'the best GZSL unseen accuracy is {best_acc_unseen}')
    logger.info(f'the best GZSL H is {best_gzsl_acc}')
