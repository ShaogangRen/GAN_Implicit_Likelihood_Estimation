import torch
from torch import nn
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
import pickle,time
from collections import OrderedDict
import matplotlib.pyplot as plt
from utils import *
from options import Options
from evaluate import evaluate
from visualizer import Visualizer
import itertools
from scipy.io import savemat
import importlib

""" Training
"""

""" ===================== TRAINING ======================== """
def reset_grad(params):
    for p in params:
        if p.grad is not None:
            data = p.grad.data
            p.grad = Variable(data.new().resize_as_(data).zero_())

""" ==================== GENERATOR ======================== """

'''=============== var_net ==================='''
class Var_Net(nn.Module):
    def __init__(self, K=20, output_dim=2, use_cuda=True):
        super(Var_Net, self).__init__()
        #self.opt = opt
        #self.K = int(opt.K/opt.ngpu) ##we cannot apply mult-gpu simply to Var_Net
        self.K = K
        self.output_dim = output_dim
        self.use_cuda = use_cuda
        self.W = nn.Parameter(0.1*torch.rand(self.K, self.output_dim).cuda() if self.use_cuda else
                              0.1*torch.rand(self.K, self.output_dim), requires_grad=True)

    def forward(self, z, cnts, lams): ##z has batch_size x z_dim and cnts has K x z_dim
        z_list = torch.unbind(z)
        V_z = []  ###the following loop will generate a list with batch_size and each one has K element
        for z_i in z_list: ##z_list has batch_size, K cluters for cnts
            t1 = torch.sum(torch.pow(cnts - z_i.repeat(self.K, 1), 2), 1) ### K x z_dim - K x z_dim yield K x z_dim
            tmp = torch.exp(-torch.mul(t1, lams)) ## K x z_dim * (K)
            V_z.append(tmp) ##so, V_z has batch_size x z_dim x K

        beta_z = torch.add(torch.stack(V_z) @ torch.mul(self.W,self.W), 0.000001) ##
        sigma_z = torch.pow(torch.sqrt(beta_z), -1) ##batch_size x self.ch_dim, w and h
        sigma_z = sigma_z.view(z.shape[0], self.output_dim)
        return sigma_z

    def clip_var(self): ##clamp in pytorch is the same as clip in numpy
        self.W.data.clamp_(min=self.opt.clip_Reg) ##0.0001

def center_lambda(z_data, K, a=1.0): ##a=1.5 or 0.7
    kmeans = KMeans(n_clusters=K, random_state=0).fit(z_data)
    cnts = kmeans.cluster_centers_

    #a = 1.0 ###a parameter needs fine-tuned RBF kernel
    cl_cnt = np.zeros((K,), dtype = float)
    for i in range(len(kmeans.labels_)):
        cl_cnt[kmeans.labels_[i]] += 1.0
    #print(cl_cnt)
    lams = np.zeros((K,), dtype = float)
    for i in range(len(kmeans.labels_)):
        lams[kmeans.labels_[i]] += np.sqrt(np.sum(np.square( cnts[kmeans.labels_[i],:]
                                    - z_data[i,:])))/cl_cnt[kmeans.labels_[i]]
    for i in range(K):
        lams[i] = np.power(a*lams[i], -2)*0.5
    #print(lams)
    return cnts, lams


class generator(nn.Module):
    def __init__(self, z_dim= 30, output_dim=1, hidden_dim = 48 ):
        super(generator, self).__init__()
        self.z_dim = z_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.fc = nn.Sequential(
            nn.Linear(self.z_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, self.output_dim)
        )
        initialize_weights(self)

    def forward(self, input):
        x = self.fc(input)
        return x

class discriminator(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, hidden_dim = 48):
        super(discriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.BatchNorm1d(256), ##?
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),  ##?
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        initialize_weights(self)

    def forward(self, input):
        x = self.fc(input)
        return x

class inv_generator(nn.Module):
    def __init__(self, input_dim=1, output_dim = 30, hidden_dim = 48):
        super(inv_generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, self.output_dim)
        )
        initialize_weights(self)

    def forward(self, input):
        x = self.fc(input)
        return x

class InvGAN(object):

    @staticmethod
    def name():
        """Return name of the class.
        """
        return 'InvGAN'

    def __init__(self, opt, train_hist,dataloader = None):
        # parameters
        self.opt = opt
        self.visualizer = Visualizer(opt)
        self.sample_num = 100
        self.train_hist = train_hist
        self.best_auc = .0
        self.best_auc_arr = []
        self.save_invgan_time_arr = []
        self.dataloader = dataloader
        if self.dataloader is not None:
            data = self.dataloader['train'].__iter__().__next__()[0]

        data = importlib.import_module("ALAD.data.{}".format(opt.dataset))
        # # networks init
        self.hidden_dim = self.opt.h_dim
        data_train = data.get_train()
        self.xdata = data_train[0]
        self.y = data_train[1]
        self.Batch_Max = self.xdata.shape[0]//self.opt.batch_size

        data_test = data.get_test()
        self.xdata_test = data_test[0]
        self.y_test = data_test[1]
        self.Batch_Max_test = self.xdata_test.shape[0]//self.opt.batch_size

        x_, _ = self.get_batch_data(1)
        print('dimenstions')
        print(x_.shape[0])
        print(x_.shape[1])
        print(self.opt.z_dim)
        print(self.hidden_dim)

        self.G = generator(z_dim=self.opt.z_dim, output_dim=x_.shape[1], hidden_dim= self.hidden_dim)
        self.G = torch.nn.DataParallel(self.G, device_ids=self.opt.gpu_ids).cuda()
        self.D = discriminator(input_dim=x_.shape[1], output_dim=1, hidden_dim= self.hidden_dim)
        self.var_net = Var_Net(self.opt.K, output_dim=x_.shape[1])
        self.D = torch.nn.DataParallel(self.D, device_ids=opt.gpu_ids).cuda()
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=opt.lrG, betas=(opt.beta1, opt.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=opt.lrD, betas=(opt.beta1, opt.beta2))

        if self.opt.gpu_mode:
            self.G.cuda()
            self.D.cuda()
            self.BCE_loss = nn.BCELoss().cuda()
        else:
            self.BCE_loss = nn.BCELoss()

        print('---------- Networks architecture -------------')
        print_network(self.G)
        print_network(self.D)
        print('-----------------------------------------------')

        self.IG = inv_generator(input_dim=x_.shape[1], output_dim= self.opt.z_dim, hidden_dim= self.hidden_dim)
        self.IG = torch.nn.DataParallel(self.IG, device_ids=self.opt.gpu_ids).cuda()
        self.IG_Var_solver = optim.Adam(itertools.chain(self.IG.parameters(), self.var_net.parameters()), lr=opt.lrIG)
        self.Q_solver = optim.Adam(itertools.chain(self.IG.parameters(), self.G.parameters()), lr=opt.lrIG)

        self.Xi = []
        if self.opt.gpu_mode:
            self.IG.cuda()
            self.BCE_loss = nn.BCELoss().cuda()
        else:
            self.BCE_loss = nn.BCELoss()

        print('---------- Networks architecture -------------')
        print_network(self.IG)
        print_network(self.var_net)
        print('-----------------------------------------------')

        # fixed noise
        self.sample_z_ = torch.rand((self.opt.batch_size, self.opt.z_dim))
        if self.opt.gpu_mode:
            self.sample_z_ = self.sample_z_.cuda()

    def mean_var(self, zz):
        mu =  torch.mean(zz, 0)
        std_z = torch.std(zz, 0)
        sig = std_z
        return mu, sig

    def get_batch_data(self, it):
        idx = it % self.Batch_Max
        return self.xdata[self.opt.batch_size * idx:self.opt.batch_size * (idx + 1), :], \
               self.y[self.opt.batch_size * idx:self.opt.batch_size * (idx + 1)]

    def get_batch_data_test(self, it):
        idx = it % self.Batch_Max_test
        return self.xdata_test[self.opt.batch_size * idx:self.opt.batch_size * (idx + 1), :], \
               self.y_test[self.opt.batch_size * idx:self.opt.batch_size * (idx + 1)]

    def train(self):
        print('training IG start!!')
        self.save_dir = os.path.join(self.opt.save_dir+'_'+str(self.opt.dataset)+'_'+str(self.opt.anomaly_class)+'_'+str(self.opt.iter_gan) + '_' + str(self.opt.h_dim) + '_' + str(self.opt.z_dim),
                                self.opt.dataset, self.opt.model_name)
        #print('model saved directory=%s' % (self.save_dir))
        start_time = time.time()
        out_dir = self.opt.out_dir +'_3L_h'+str(self.hidden_dim)+'_'+ self.opt.dataset + '/'
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        self.y_real_, self.y_fake_ = torch.ones(self.opt.batch_size, 1), torch.zeros(self.opt.batch_size, 1)
        if self.opt.gpu_mode:
            self.y_real_, self.y_fake_ = self.y_real_.cuda(), self.y_fake_.cuda()
        fcounter = 1

        z_clst = torch.randn(self.opt.Clst_Sample,self.opt.z_dim)
        self.cnts, self.lambdas = center_lambda(z_clst.data.cpu().numpy(), self.opt.K, self.opt.rbf_a)
        self.cnts = torch.tensor(self.cnts, requires_grad=False).cuda().float()
        self.lambdas = torch.tensor(self.lambdas, requires_grad=False).cuda().float()
        findex = 0

        for it in range(self.opt.iter_gan):
            #x_ = toy_data.inf_train_gen(self.opt.dataset, batch_size=self.opt.batch_size)
            x_, _ = self.get_batch_data(findex)
            findex = findex + 1

            x_ = torch.from_numpy(x_).type(torch.float32)
            #x_ = x_.view(self.opt.batch_size, self.opt.channels, self.opt.input_size, self.opt.input_size)
            if self.opt.use_cuda:
                x_ = x_.cuda()
            z_ = Variable(torch.randn(self.opt.batch_size, self.opt.z_dim))
            if self.opt.use_cuda:
                z_ = z_.cuda()
            # update D network
            self.D_optimizer.zero_grad()
            D_real = self.D(x_)
            D_real_loss = self.BCE_loss(D_real, self.y_real_)

            G_ = self.G(z_)
            D_fake = self.D(G_)
            D_fake_loss = self.BCE_loss(D_fake, self.y_fake_)

            D_loss = D_real_loss + D_fake_loss
            #self.train_hist['D_loss'].append(D_loss.item())
            D_loss.backward()
            self.D_optimizer.step()

            # update G network
            self.G_optimizer.zero_grad()
            epoch_start_time = time.time()
            #self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)

            #print('outer iter = %d' % it)
            z_ = torch.randn(self.opt.batch_size, self.opt.z_dim)
            if self.opt.use_cuda:
                z_ = z_.cuda()
            G_ = self.G(z_)

            D_fake = self.D(G_)
            G_loss = self.BCE_loss(D_fake, self.y_real_)
            G_loss.backward()
            self.G_optimizer.step()

            '''================================  train encoder and sigma ============================='''
            inv_start = 4000
            if it >= inv_start:
                x_, _ = self.get_batch_data(findex)
                findex = findex + 1
                x_ = torch.from_numpy(x_).type(torch.float32)
                if self.opt.use_cuda:
                    x_ = x_.cuda()
                z_hat = self.IG(x_)
                sig_x = self.var_net(z_hat, self.cnts, self.lambdas)
                X_t = self.G(z_hat)
                recon_loss = torch.mean(torch.div(torch.pow((X_t - x_), 2), 2.0*torch.pow(sig_x, 2))  + torch.log(sig_x))  ##we need to make sure use u_q or z_hat?
                u_hat, var_hat = self.mean_var(z_hat)
                kl_loss = torch.mean(torch.log(var_hat) + (1+ (0-u_hat)**2)/(2*var_hat**2))
                loss_x = recon_loss + kl_loss
                loss = loss_x #+ loss_z
                self.IG_Var_solver.zero_grad()
                loss.backward()
                self.IG_Var_solver.step()

                ''''=== add MI ===='''
                ###==== Q Learn ====
                z_ = torch.randn(self.opt.batch_size, self.opt.z_dim)
                if self.opt.use_cuda:
                    z_ = z_.cuda()
                X_z = self.G(z_)
                z_hat_z = self.IG(X_z)
                recon_loss_z = torch.mean(torch.pow((z_ - z_hat_z), 2))
                loss_z = recon_loss_z
                self.Q_solver.zero_grad()
                loss_z.backward()
                self.Q_solver.step()

                if (it > inv_start+2000) and ((it + 1) % 100 == 0):
                    print("Iteration: [%2d] [%4d] D_loss: %.8f, G_loss: %.8f" %
                          ((it + 1), self.opt.iter_gan, D_loss.item(),
                           G_loss.item()))
                    #self.train_hist['G_loss'].append(G_loss.item())
                    log_str = 'Train: ite ={} loss = {}\n'.format((it + 1), loss.data.cpu().numpy())
                    print(log_str)
                    '''==== test likelihood ========'''
                    #x_ = toy_data.inf_train_gen(self.opt.dataset, batch_size=self.opt.batch_size)
                    x_, _ = self.get_batch_data(findex)
                    findex = findex + 1
                    x_ = torch.from_numpy(x_).type(torch.float32)
                    if self.opt.use_cuda:
                        x_ = x_.cuda()
                    z_hat = self.IG(x_)
                    #llk_est = log_likelihood(self.cnts, self.lambdas, self.G, self.var_net, z_hat, x_)
                    llk_est, _, _, _ = G_LLK_Geom(self.G, self.var_net, self.cnts, self.lambdas,
                                                                     self.opt.batch_size, z_hat)
                    mean_llk_est = np.mean(llk_est)
                    print('Train: ite ={} llk_mean = {}\n'.format((it + 1), mean_llk_est))

                if (it > inv_start+2000) and (it % self.opt.test_interv == 0):
                    findex_test = 0
                    if 'kdd' in self.opt.dataset:
                        test_N = 1000
                    else:
                        test_N = self.Batch_Max_test

                    for i_test in range(test_N):
                        x_, y_ = self.get_batch_data_test(findex_test)
                        findex_test = findex_test + 1

                        x_ = torch.from_numpy(x_).type(torch.float32)
                        if self.opt.use_cuda:
                            x_ = x_.cuda()
                        z_hat = self.IG(x_)
                        # llk_est = log_likelihood(self.cnts, self.lambdas, self.G, self.var_net, z_hat, x_)
                        llk_est, _, _, _ = G_LLK_Geom(self.G, self.var_net, self.cnts, self.lambdas,
                                                      self.opt.batch_size, z_hat)
                        if i_test == 0:
                            llk_test_all = llk_est
                            y_test_grnd_all = y_
                        else:
                            llk_test_all = np.concatenate((llk_test_all, llk_est), axis=0)
                            y_test_grnd_all = np.concatenate((y_test_grnd_all, y_), axis=0)
                    print('Ite ={}  test llk_mean = {}\n'.format((it + 1), np.mean(llk_test_all)))
                    self.eval_anomaly(y_test_grnd_all, llk_test_all)
                    filename = out_dir + "/llk_g_ig_ano_3L_{}.mat".format(it)
                    savemat(filename, {'llk_est': llk_test_all, 'llk_grdtrue': y_test_grnd_all})

    def eval_anomaly(self, y_test_grnd_all, llk_test_all):
        gt_labels = torch.from_numpy(y_test_grnd_all)
        an_scores = torch.from_numpy(-llk_test_all) ### neg-log-likelihood
        an_scores[an_scores == float('-Inf')] = -1000000
        an_scores[an_scores == float('Inf')] = 1000000
        an_scores[an_scores != an_scores] = 0  ## if score is nan
        an_scores[an_scores==0]=torch.mean(an_scores)
        an_scores = (an_scores - torch.min(an_scores)) / (torch.max(an_scores) - torch.min(an_scores))
        print('the length of labels=%d and the length of scores=%d' % (gt_labels.shape[0],an_scores.shape[0]))
        if 'arrhythmia' in self.opt.dataset:
            threshold = 0.15
        else:
            threshold = 0.2
        auc = evaluate(gt_labels, an_scores, metric='roc', threshold= threshold)
        print('roc={}'.format(auc))
        auc = evaluate(gt_labels, an_scores, metric='auprc', threshold= threshold)
        print('auprc={}'.format(auc))
        auc = evaluate(gt_labels, an_scores, metric='f1_score', threshold= threshold)
        print('f1_score={}'.format(auc))


    def save_weights(self, epoch):
        """Save netG and netD weights for the current epoch.

        Args:
            epoch ([int]): Current epoch number.
        """
        weight_dir = os.path.join(self.opt.outf, self.opt.name, 'train', 'weights')
        if not os.path.exists(weight_dir):
            os.makedirs(weight_dir)

        torch.save({'epoch': epoch + 1, 'state_dict': self.G.state_dict()},
                   '%s/netG.pth' % (weight_dir))
        torch.save({'epoch': epoch + 1, 'state_dict': self.G.state_dict()},
                   '%s/netD.pth' % (weight_dir))

    def save_invgan(self):
        save_time = time.time()
        # save_dir = os.path.join(self.opt.save_dir+'_'+str(self.opt.h_dim)+'_'+str(self.opt.z_dim), self.opt.dataset, self.opt.model_name)

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if self.opt.task == 'anomaly_detect':
            torch.save(self.IG.module.state_dict(), os.path.join(self.save_dir, self.opt.model_name+'_'+self.opt.task+'_'+str(self.opt.anomaly_class)+'_'+str(save_time) + '_IG.pkl'))
            # not so sure if we can save an array yet. I know why it is not working since I only save results when auc is larger than 80%.
            # torch.save([self.var_net.state_dict(),self.cnts.state_dict(),self.lambdas.state_dict()], os.path.join(self.save_dir,
            #                                                      self.opt.model_name + '_' + self.opt.task + '_' + str(
            #                                                          self.opt.anomaly_class) + '_' + str(
            #                                                          save_time) + '_VN.pkl'))
            torch.save(self.var_net.state_dict(), os.path.join(self.save_dir,
                                                                 self.opt.model_name + '_' + self.opt.task + '_' + str(
                                                                     self.opt.anomaly_class) + '_' + str(
                                                                     save_time) + '_VN.pkl'))
            torch.save([self.cnts,self.lambdas], os.path.join(self.save_dir,
                                                                 self.opt.model_name + '_' + self.opt.task + '_' + str(
                                                                     self.opt.anomaly_class) + '_' + str(
                                                                     save_time) + '_cnts_lambdas.pkl'))
        elif self.opt.task == 'classification':
            torch.save(self.IG.module.state_dict(), os.path.join(self.save_dir,
                                                                 self.opt.model_name + '_' + self.opt.task +'_' + str(
                                                                     save_time) + '_IG.pkl'))
            # torch.save([self.var_net.state_dict(),self.cnts.state_dict(),self.lambdas.state_dict()], os.path.join(self.save_dir,
            #                                                      self.opt.model_name + '_' + self.opt.task+ '_' + str(
            #                                                          save_time) + '_VN.pkl'))

            torch.save(self.var_net.state_dict(), os.path.join(self.save_dir,
                                                                 self.opt.model_name + '_' + self.opt.task + '_' + str(
                                                                     save_time) + '_VN.pkl'))
            torch.save([self.cnts,self.lambdas], os.path.join(self.save_dir,
                                                                 self.opt.model_name + '_' + self.opt.task + '_' + str(
                                                                     save_time) + '_cnts_lambdas.pkl'))

        with open(os.path.join(self.save_dir, self.opt.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)
        return save_time

    def save_gan(self):
        #save_dir = os.path.join(self.opt.save_dir, self.opt.dataset, self.opt.model_name)
        # save_dir = os.path.join(self.opt.save_dir+'_'+self.opt.dataset + '_' + str(self.opt.h_dim) + '_' + str(self.opt.z_dim),
        #                         self.opt.dataset, self.opt.model_name)

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        torch.save(self.G.module.state_dict(), os.path.join(self.save_dir, self.opt.model_name + '_G.pkl'))
        torch.save(self.D.module.state_dict(), os.path.join(self.save_dir, self.opt.model_name + '_D.pkl'))

        with open(os.path.join(self.save_dir, self.opt.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)


    def load_invgan(self,save_time):
        #save_time = self.save_invgan_time_arr[time_ind]
        #save_dir = os.path.join(self.opt.save_dir, self.opt.dataset, self.opt.model_name)
        # save_dir = os.path.join(self.opt.save_dir + '_' + str(self.opt.h_dim) + '_' + str(self.opt.z_dim),
        #                         self.opt.dataset, self.opt.model_name)

        if self.opt.task == 'anomaly_detect':
            self.IG.module.load_state_dict(torch.load(os.path.join(self.save_dir,
                                                                 self.opt.model_name + '_' + self.opt.task + '_' + str(
                                                                     self.opt.anomaly_class) +'_' + str(
                                                                     save_time) + '_IG.pkl'),map_location=torch.device('cuda:'+str(self.opt.gpu_ids[0]) if torch.cuda.is_available() else 'cpu')))
            self.var_net.load_state_dict(torch.load(os.path.join(self.save_dir,
                                                                   self.opt.model_name + '_' + self.opt.task + '_' + str(
                                                                     self.opt.anomaly_class) +'_' + str(
                                                                       save_time) + '_VN.pkl'),map_location=torch.device('cuda:'+str(self.opt.gpu_ids[0]) if torch.cuda.is_available() else 'cpu')))
        elif self.opt.task == 'classification':
            self.IG.module.load_state_dict(torch.load(os.path.join(self.save_dir,
                                                                 self.opt.model_name + '_' + self.opt.task +'_' + str(
                                                                     save_time) + '_IG.pkl'),map_location=torch.device('cuda:'+str(self.opt.gpu_ids[0]) if torch.cuda.is_available() else 'cpu')))
            self.var_net.load_state_dict(torch.load(os.path.join(self.save_dir,
                                                                   self.opt.model_name + '_' + self.opt.task + '_' + str(
                                                                       save_time) + '_VN.pkl'),map_location=torch.device('cuda:'+str(self.opt.gpu_ids[0]) if torch.cuda.is_available() else 'cpu')))

        self.IG.z_dim = self.opt.z_dim
        self.G.output_dim = self.IG.module.input_dim

    def load_gan(self):
        #save_dir = os.path.join(self.opt.save_dir, self.opt.dataset, self.opt.model_name)
        # save_dir = os.path.join(self.opt.save_dir + '_' + str(self.opt.h_dim) + '_' + str(self.opt.z_dim),
        #                         self.opt.dataset, self.opt.model_name)
        self.G.module.load_state_dict(torch.load(os.path.join(self.save_dir, self.opt.model_name + '_G.pkl'),map_location=torch.device('cuda:'+str(self.opt.gpu_ids[0]) if torch.cuda.is_available() else 'cpu')))
        self.D.module.load_state_dict(torch.load(os.path.join(self.save_dir, self.opt.model_name + '_D.pkl'),map_location=torch.device('cuda:'+str(self.opt.gpu_ids[0]) if torch.cuda.is_available() else 'cpu')))
        #self.D.module.load_state_dict(torch.load(os.path.join(self.save_dir, self.opt.model_name + '_D.pkl')))

    # ##
    def test(self,mode): ##mode refers to whether it is a validation or a test
        print('mode=%s' % mode)
        """ Test ganomaly model.

        Args:
            dataloader ([type]): Dataloader for the test set

        Raises:
            IOError: Model weights not found.
        """
        # Create big error tensor for the test set.
        self.an_scores = torch.zeros(size=(len(self.dataloader['test'].dataset),), dtype=torch.float32,
                                     device='cuda')
        self.gt_labels = torch.zeros(size=(len(self.dataloader['test'].dataset),), dtype=torch.long, device='cuda')
        self.gt = torch.empty(size=(self.opt.batch_size,), dtype=torch.long, device='cuda')
        X_Buf = []
        time_o = time.time()
        epoch_iter = 0
        self.times = []
        for it, (x_t, labels) in enumerate(self.dataloader['test']):
            if mode == 'validation' and it*self.opt.batch_size >= self.opt.val_ind: ##there are about 15876 test data. we use 336 of them for validation
                break
            elif mode == 'test' and (it*self.opt.batch_size > self.opt.test_ind or it*self.opt.batch_size < self.opt.val_ind): ##since we have tested them in validation, we skip them.
                continue
            epoch_iter += self.opt.batch_size
            self.gt.data.resize_(labels.size()).copy_(labels)
            if it == self.dataloader['test'].dataset.__len__() // self.opt.batch_size:
                break
            if self.opt.use_cuda:
                x_t = x_t.cuda() # +gausssion noise
                x_t = x_t.view(-1,self.opt.channels,self.opt.input_size,self.opt.input_size)
                #print(x_t[0])
                z = self.IG(x_t)
                if self.opt.use_cuda:
                    z = z.cuda()
                llk_est,_,_,_ = G_LLK_Geom(self.G, self.var_net, self.cnts, self.lambdas, self.opt.batch_size, z,'',self.opt.llk_way, self.opt.diag_Reg_val, self.opt.use_cuda)
                print('iter={} and the final batch test loglikelihood on test data={}\n'.format(it,llk_est))
                time_i = time.time()
                self.an_scores[it*self.opt.batch_size : it*self.opt.batch_size+llk_est.size] = torch.from_numpy(llk_est)
                self.gt_labels[it*self.opt.batch_size : it*self.opt.batch_size+llk_est.size] = self.gt.reshape(llk_est.size)
                self.times.append(time_o - time_i)

        self.times = np.array(self.times)
        self.times = np.mean(self.times[:100] * 1000)
        # Scale error vector between [0, 1]

        if mode == 'validation':
            self.an_scores = self.an_scores[0:self.opt.val_ind]
            self.gt_labels = self.gt_labels[0:self.opt.val_ind]
        elif mode == 'test':
            self.an_scores = self.an_scores[self.opt.val_ind:self.opt.test_ind]
            self.gt_labels = self.gt_labels[self.opt.val_ind:self.opt.test_ind]

        self.an_scores[self.an_scores == float('-Inf')] = -1000000
        self.an_scores[self.an_scores == float('Inf')] = 1000000
        self.an_scores[self.an_scores != self.an_scores] = 0  ## if score is nan
        #self.an_scores[self.an_scores == float('Inf')] = sys.float_info.max/2
        self.an_scores[self.an_scores==0]=torch.mean(self.an_scores)
        self.an_scores = (self.an_scores - torch.min(self.an_scores)) / (torch.max(self.an_scores) - torch.min(self.an_scores))
        # auc, eer = roc(self.gt_labels, self.an_scores)
        print('the length of labels=%d and the length of scores=%d' % (self.gt_labels.size()[0],self.an_scores.size()[0]))
        auc = evaluate(self.gt_labels, self.an_scores, metric=self.opt.metric)
        print('anamoly class=%s and gpu_id=%s and auc=%f' % (self.opt.anomaly_class,self.opt.gpu_ids,auc))
        performance = OrderedDict([('Avg Run Time (ms/batch)', self.times), ('AUC', auc)])
        if self.opt.display_id > 0 and self.opt.phase == 'test':
            counter_ratio = float(epoch_iter) / len(self.dataloader['test'].dataset)
            self.visualizer.plot_performance(self.opt.epoch, counter_ratio, performance)
        return performance

    def random_walk(self):
        z = np.random.normal(0, 1, (self.opt.batch_size, self.opt.z_dim))
        N = 10000  ##step
        z_path = []
        tz = torch.from_numpy(z)
        if self.opt.use_cuda:
            tz = tz.cuda().float()
        for it in range(N):
            rem_metric, _, _, _= G_Jacobian_eig(self.G, self.var_net, self.cnts, self.lambdas, tz)
            for ii in range(len(rem_metric)):
                m = rem_metric[ii].data.cpu().numpy()
                #print(m)
                L, U = np.linalg.eigh(m)
                #print(L)
                # print(L.shape)
                # print(U.shape)
                # T = np.matmul(U, np.sqrt(L))
                # print(T.shape)
                v = np.multiply(np.matmul(U, np.sqrt(L)), np.random.normal(0, 1, L.shape[0]))
                z[ii, :] = z[ii, :] + 0.0000000001 * v
            z_path.append(z[0])
            tz = torch.from_numpy(z)
            if self.opt.use_cuda:
                tz = tz.cuda().float()
            img_smpl = self.G(tz).data.cpu().numpy()
            samples = np.reshape(img_smpl, [self.opt.batch_size, self.opt.isize, self.opt.isize, self.opt.channels])
            print('N=%d and it=%d' % (N,it))
            if not os.path.exists(self.opt.dataset+"_randwk_out"):
                os.mkdir(self.opt.dataset+"_randwk_out")
            save_images(samples, [4, 4],
                        './{}/train_{:07d}_b.png'.format(self.opt.dataset+"_randwk_out", it))
        fname = "random_out/path.p"
        pickle.dump(z_path, open(fname, "wb"))

    def est_llk_trend(self,iter):
        time_o = time.time()
        epoch_iter = 0
        self.times = []
        # self.llk_mean = torch.zeros(size=(10,), dtype=torch.float32,
        #                              device='cuda')
        self.llk_mean = np.zeros(shape=[20],dtype=float)
        for it, (x_t, labels) in enumerate(self.dataloader['test']):
            epoch_iter += self.opt.batch_size
            if it > 0: ##looks that we only need one batch
                break
            for var_it in range(20):
                if self.opt.use_cuda:
                    #gaussian(self.W, 0. 0, self.opt.adding_noise_var)
                    x_t = gaussian(x_t, 0.0, 100*var_it).cuda() ## gausssion noise
                    x_t = x_t.view(-1, self.opt.channels, self.opt.input_size, self.opt.input_size)
                    # print(x_t[0])
                    z = self.IG(x_t)
                    if self.opt.use_cuda:
                        z = z.cuda()
                    llk_est, _, _, _ = G_LLK_Geom(self.G, self.var_net, self.cnts, self.lambdas, self.opt.batch_size, z, '',
                                                  self.opt.llk_way, self.opt.diag_Reg_val, self.opt.use_cuda)
                    print('iter={} and the final batch test loglikelihood on test data={}\n'.format(var_it, llk_est))
                    llk_est[llk_est == float('-Inf')] = -1000000
                    llk_est[llk_est == float('Inf')] = 1000000
                    llk_est[llk_est != llk_est] = 0  ## if score is nan
                    # self.an_scores[self.an_scores == float('Inf')] = sys.float_info.max/2
                    llk_est[llk_est == 0] = np.mean(llk_est)
                    self.llk_mean[var_it] = np.mean(llk_est)
                    time_i = time.time()
                    self.times.append(time_o - time_i)
            # Measure inference time.
        print('llk_mean result')
        print(self.llk_mean)
        figure = plt.figure(1)
        plt.rc('text', usetex=False)
        plt.axis([0, 19, -80.0, -40.0])
        #self.llk_mean = self.llk_mean.cpu.numpy()

        plt.plot([0,1,2,3,4,5,6,7,8,9],self.llk_mean,'rs--',label=self.opt.dataset)
        plt.xlabel('iteration')
        plt.ylabel('average loglikelihood')
        plt.legend(loc='lower right')
        plt.show()
        figure.savefig(self.opt.dataset+'_loglikelihood_trend_'+str(iter)+'.png', dpi=figure.dpi)
        self.times = np.array(self.times)
        self.times = np.mean(self.times[:100] * 1000)
        # Scale error vector between [0, 1]


"""main"""
def main():
    # parse arguments
    opt = Options().parse()
    # from celeb_data import get_loader
    # data_loader = get_loader(
    #     './data/celebA', 'train', opt.batch_size, opt.isize)

    opt.use_clip_var = False
    #print('opt.use_clip_var = %s' % opt.use_clip_var)
    train_hist = {}
    train_hist['D_loss'] = []
    train_hist['G_loss'] = []
    train_hist['IG_loss'] = []
    train_hist['per_epoch_time'] = []
    train_hist['total_time'] = []
    if opt is None:
        exit()
    ##
    # LOAD DATA
    dataloader = None #load_data(opt)
    #gan = GAN(opt,train_hist,dataloader)
        # launch the graph in a session
    #gan.train()
    #gan.load()
    igan = InvGAN(opt,train_hist,dataloader)
    igan.train()
    print(" [*] Training finished!")

    # """ ===================== TRAINING ======================== """
    # best_auc = 0.0
    # # visualize learned generator
    # #gan.visualize_results(opt.epoch)
    # best_auc_arr = igan.best_auc_arr
    # save_invgan_time_arr = igan.save_invgan_time_arr
    #
    # for time in save_invgan_time_arr:
    #     igan.load_invgan(time)
    #     res = igan.test('test')
    #     igan.save_weights(igan.opt.epoch)
    #     igan.visualizer.print_current_performance(res, best_auc)
    # print(" [*] Testing finished!")
    # if res['AUC'] > best_auc:
    #     best_auc = res['AUC']
    print(">> Training model %s.[Done]" % igan.name())

if __name__ == '__main__':
    main()





