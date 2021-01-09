import argparse
import os
import torch

class Options():

    def __init__(self):
        ##
        #
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        ##
        # Base
        self.parser.add_argument('--model_name', type=str, default='GAN',
                            choices=['GAN', 'CGAN', 'infoGAN', 'ACGAN', 'EBGAN', 'BEGAN', 'WGAN', 'WGAN_GP', 'DRAGAN',
                                     'LSGAN'],
                            help='The type of GAN')
        self.parser.add_argument('--dataset', default='cifar10', help='folder | cifar10 | mnist ')
        self.parser.add_argument('--dataroot', default='', help='path to dataset')
        self.parser.add_argument('--batch_size', type=int, default=14, help='input batch size')
        self.parser.add_argument('--channels', type=int, default=1, help='input channel size')
        self.parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
        self.parser.add_argument('--droplast', action='store_true', default=True, help='Drop last batch size.')
        self.parser.add_argument('--input_size', type=int, default=32, help='input image size.')
        self.parser.add_argument('--LAMBDA_IG', type=float, default=3.0, help='lambda IG')
        self.parser.add_argument('--nc', type=int, default=3, help='input image channels')
        self.parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
        self.parser.add_argument('--ngf', type=int, default=64)
        self.parser.add_argument('--ndf', type=int, default=64)
        self.parser.add_argument('--extralayers', type=int, default=0, help='Number of extra layers on gen and disc')
        self.parser.add_argument('--device', type=str, default='gpu', help='Device: gpu | cpu')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment')
        self.parser.add_argument('--model', type=str, default='ganomaly', help='chooses which model to use. ganomaly')
        self.parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        self.parser.add_argument('--display_id', type=int, default=0, help='window id of the web display')
        self.parser.add_argument('--display', action='store_true', help='Use visdom.')
        self.parser.add_argument('--outf', default='./output', help='folder to output images and model checkpoints')
        self.parser.add_argument('--manualseed', default=-1, type=int, help='manual seed')
        self.parser.add_argument('--anomaly_class', default='car', help='Anomaly class idx for mnist and cifar datasets')
        self.parser.add_argument('--proportion', type=float, default=0.1, help='Proportion of anomalies in test set.')
        self.parser.add_argument('--task', type=str, default='classify', help='classfy or anomaly_detect')
        self.parser.add_argument('--metric', type=str, default='roc', help='Evaluation metric.')
        self.parser.add_argument('--Sig_It_Num', type=int, default=3, help='Sig_It_Num')
        self.parser.add_argument('--Clst_Sample', type=int, default=1000, help='clst_sample')
        self.parser.add_argument('--save_dir', type=str, default='models',help='Directory name to save the model')
        self.parser.add_argument('--result_dir', type=str, default='results',
                            help='Directory name to save the generated images')
        self.parser.add_argument('--out_dir', type=str, default='Out_g_ig_ano', help='out put dir')
        self.parser.add_argument('--log_dir', type=str, default='logs', help='Directory name to save training logs')
        self.parser.add_argument('--sl_range', type=int, default=1, help='sl_range')
        self.parser.add_argument('--wk_range', type=int, default=1, help='wk_range')
        self.parser.add_argument('--Warm_Up_Ite_Num', type=int, default=1, help='Warm_Up_Ite_Num')
        self.parser.add_argument('--Max_Ite', type=int, default=1, help='Max_Ite')
        self.parser.add_argument('--IG_Start_It', type=int, default=1, help='IG_Start_It')
        self.parser.add_argument('--IG_GAP', type=int, default=1, help='IG_GAP')
        self.parser.add_argument('--lrG', type=float, default=0.0002)
        self.parser.add_argument('--lrD', type=float, default=0.0002)
        self.parser.add_argument('--lrIG', type=float, default=0.0002)
        self.parser.add_argument('--var_lr', type=float, default=0.0001)
        self.parser.add_argument('--var_solver', type=str, default='SGD')
        #self.parser.add_argument('--beta1', type=float, default=0.5)
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--beta2', type=float, default=0.999)
        self.parser.add_argument('--gpu_mode', type=bool, default=True)
        self.parser.add_argument('--G_w_Reg', type=bool, default=False)
        self.parser.add_argument('--use_G_var', type=bool, default=False)
        self.parser.add_argument('--Lambda_G_Reg', type=float, default=0.001)
        self.parser.add_argument('--Var_loss_Reg', type=float, default=1)
        self.parser.add_argument('--diag_Reg_val', type=float, default=0.1)
        self.parser.add_argument('--W_penaty_det', type=float, default=20)
        self.parser.add_argument('--use_clip_var', type=bool, default=False)
        #self.parser.add_argument('--Var_loss_sqrt_penality', type=bool, default=True)
        self.parser.add_argument('--Var_loss_penaty', type=str, default='quadruple', help='quadruple, square, adding_noise and det') ##or square or quadruple
        self.parser.add_argument('--llk_way', type=str, default='eig',
                                 help='det or eig for now')
        self.parser.add_argument('--kl_Reg', type=float, default=1.0)
        self.parser.add_argument('--adding_noise_var', type=float, default=1)
        self.parser.add_argument('--clip_Reg', type=float, default=0.0)

        self.parser.add_argument('--benchmark_mode', type=bool, default=True)
        self.parser.add_argument('--K', type=int, default=40, help='K is cluster number')
        self.parser.add_argument('--h_dim', type=int, default=40, help='h_dim is dimension of hidden layers')
        self.parser.add_argument('--z_dim', type=int, default=36, help='h_dim is dimension of hidden layers')
        self.parser.add_argument('--z_disc_dim', type=bool, default=5, help='h_dim is dimension of hidden layers')
        self.parser.add_argument('--use_cuda', type=bool, default=torch.cuda.is_available(), help='whether to use cuda')
        self.parser.add_argument('--epoch', type=int, default=1, help='The number of epochs to run')
        self.parser.add_argument('--isize', type=int, default=32, help='input image size.')
        ##
        # Train
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--iter_gan', type=int, default=100,
                                 help='frequency of returning to gan training')
        self.parser.add_argument('--llk_step', type=int, default=100,
                                 help='frequency of returning to gan training')

        self.parser.add_argument('--llk_step_long', type=int, default=100,
                                 help='frequency of returning to gan training')
        self.parser.add_argument('--loss_check_step', type=int, default=100,
                                 help='frequency of checking loss')
        self.parser.add_argument('--log_write_step', type=int, default=100,
                                 help='frequency of write logs')
        self.parser.add_argument('--fcounter_step', type=int, default=5,
                                 help='frequency of reopen fcounter')
        self.parser.add_argument('--points', type=int, default=80,
                                 help='points used for simulation tests')
        self.parser.add_argument('--save_ig_freq', type=int, default=100,
                                 help='frequency of saving ig_step')
        self.parser.add_argument('--val_ind', type=int, default=336,
                                 help='last index of test data used for validation')
        self.parser.add_argument('--test_ind', type=int, default=672,
                                 help='last index of test data used for test')
        self.parser.add_argument('--save_image_freq', type=int, default=100, help='frequency of saving real and fake images')
        self.parser.add_argument('--save_test_images', action='store_true', help='Save test images for demo.')
        self.parser.add_argument('--load_weights', action='store_true', help='Load the pretrained weights')
        self.parser.add_argument('--resume', default='', help="path to checkpoints (to continue training)")
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--IGM_Loss_way', type=str, default='alt', help='alt or notalt')
        #self.parser.add_argument('--iter', type=int, default=0, help='Start from iteration i')
        #self.parser.add_argument('--niter', type=int, default=15, help='number of epochs to train for')
        self.parser.add_argument('--sample_num', type=int, default=100, help='sample numbers for GAN')
        self.parser.add_argument('--rbf_a', type=float, default=1.0, help='rbf kernel normalization parameter, which can be 1., 1 or 0.7')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        self.parser.add_argument('--auc_saved_bar', type=float, default=0.6, help='auc_saved_bar')

        self.parser.add_argument('--w_bce', type=float, default=1, help='alpha to weight bce loss.')
        self.parser.add_argument('--w_rec', type=float, default=50, help='alpha to weight reconstruction loss')
        self.parser.add_argument('--w_enc', type=float, default=1, help='alpha to weight encoder loss')
        self.parser.add_argument('--restore_ckt', type=str, default='1547921663.694149', help='select a time point from saved model')
        #self.parser.add_argument('--diag_reg', type=bool, default=False)
        #self.parser.add_argument('--visible_gpus', type=str, default='3,4', help='multiple gpus')
        #self.parser.add_argument('--gpu_mode', type=bool, default=True)
        self.isTrain = True
        self.opt = None

    def parse(self):
        """ Parse Arguments.
        """

        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if self.opt.device == 'gpu':
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        # print('------------ Options -------------')
        # for k, v in sorted(args.items()):
        #     print('%s: %s' % (str(k), str(v)))
        # print('-------------- End ----------------')

        # save to the disk
        if self.opt.name == 'experiment_name':
            self.opt.name = "%s/%s" % (self.opt.model, self.opt.dataset)
        expr_dir = os.path.join(self.opt.outf, self.opt.name, 'train')
        test_dir = os.path.join(self.opt.outf, self.opt.name, 'test')

        if not os.path.isdir(expr_dir):
            os.makedirs(expr_dir)
        if not os.path.isdir(test_dir):
            os.makedirs(test_dir)

        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
