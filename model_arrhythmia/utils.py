import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.autograd as autograd
from scipy.stats import norm
import os, gzip, torch
import scipy.misc
import imageio
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from torchvision import datasets, transforms

fname = "data_two.p"

# X_dim = mnist.train.images.shape[1]
# y_dim = mnist.train.labels.shape[1]
c = 0
lr = 1e-3


#os.environ["CUDA_VISIBLE_DEVICES"] = '3'
#cudnn.benchmark = True
#device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')


# use_cuda = torch.cuda.is_available()
# if use_cuda:
#     gpu = 5

# def get_data_batch(X_Buf, mb_size):
#     while True:
#         x_, y_ = mnist.train.next_batch(mb_size)
#         if len(X_Buf) < mb_size:
#             for i in range(mb_size):
#                 if not y_[i, 0]:
#                     X_Buf.append(x_[i, :])
#             continue
#         else:
#             X = Variable(torch.from_numpy(np.concatenate(X_Buf[0:mb_size])))
#             X = X.view(mb_size, -1)
#             X_Buf = X_Buf[mb_size:]
#             break
#     return X, X_Buf

def get_data_batch_iter(X_Buf, mb_size,iterr):
    while True:
        #x_, y_ = mnist.train.next_batch(mb_size)
        try:
            x_, y_ = next(iterr)
        except StopIteration:
            X = None
            break

        if len(X_Buf) < mb_size:
            for i in range(mb_size):
                X_Buf.append(x_[i, :])
            continue
        else:
            X = Variable(torch.from_numpy(np.concatenate(X_Buf[0:mb_size])))
            X = X.view(mb_size, -1)
            X_Buf = X_Buf[mb_size:]
            break
    return X, X_Buf

def get_data_label_batch_iter(X_Buf,Y_Buf, mb_size,iterr):
    while True:
        #x_, y_ = mnist.train.next_batch(mb_size)
        try:
            x_, y_ = next(iterr)
        except StopIteration:
            X = None
            Y = None
            break

        if len(X_Buf) < mb_size:
            for i in range(mb_size):
                X_Buf.append(x_[i, :])
                Y_Buf.append(y_[i])
            continue
        else:
            X = Variable(torch.from_numpy(np.concatenate(X_Buf[0:mb_size])))
            X = X.view(mb_size, -1)
            X_Buf = X_Buf[mb_size:]
            Y = Variable(torch.from_numpy(np.array(Y_Buf[0:mb_size])))
            Y_Buf = Y_Buf[mb_size:]
            break
    return X, X_Buf, Y, Y_Buf

def center_lambda(z_data, K):
    kmeans = KMeans(n_clusters=K, random_state=0).fit(z_data)
    cnts = kmeans.cluster_centers_

    a = 1.0
    cl_cnt = np.zeros((K,), dtype = float)
    for i in range(len(kmeans.labels_)):
        cl_cnt[kmeans.labels_[i]] += 1.0
    ##print(cl_cnt)
    lams = np.zeros((K,), dtype = float)
    for i in range(len(kmeans.labels_)):
        lams[kmeans.labels_[i]] += np.sqrt(np.sum(np.square( cnts[kmeans.labels_[i],:]
                                    - z_data[i,:])))/cl_cnt[kmeans.labels_[i]]
    for i in range(K):
        lams[i] = np.power(a*lams[i], -2)*0.5
    ##print(lams)
    return cnts, lams


"""=========jacobian==========="""

def G_Jacobian_Det(netG, z,use_cuda=True):
    interpolates = z
    if use_cuda:
        interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    G = netG(interpolates)
    PixN = G.size()[1]
    for i in range(PixN):
        pixel = G[:,i]
        gradients = autograd.grad(outputs=pixel, inputs=interpolates,
                                  grad_outputs=torch.ones(G.size()[0]).cuda() if use_cuda else torch.ones(G.size()[0]), create_graph=True, retain_graph=True,
                                  only_inputs=True)[0]
        gradients= gradients.unsqueeze(2)
        if i == 0:
            grad = gradients
        else:
            grad = torch.cat((grad, gradients), 2)

    det_penalty = 0
    det_abs_sum = 0
    LogDet = []
    ImN = grad.size()[0]
    for i in range(ImN):
        m = grad[i, :, :]
        #det_penalty += torch.abs(torch.det(torch.mm(m, torch.t(m))))
        LogDet.append(torch.log(torch.det(torch.mm(m, torch.t(m)))))
        tmp =  torch.det(torch.mm(m, torch.t(m)))
        det_abs_sum +=  torch.abs(tmp)
        det_penalty += (tmp- 1.0)**2
    return det_penalty, det_abs_sum, LogDet


def G_Jacobian_Eigen_Penalty(netG, z,mb_size, Z_dim,use_cuda=True):
    ub = Variable(torch.Tensor([20.0]))
    lb = Variable(torch.Tensor([1.0]))
    if use_cuda:
        ub = ub.cuda()
        lb = lb.cuda()
    delta = Variable(torch.randn(mb_size, Z_dim))
    if use_cuda:
        delta = delta.cuda()
    eps = 0.01
    delta = (delta/delta.norm(2))*eps
    z_t = z + delta
    Q = torch.sqrt(torch.sum((netG(z_t) - netG(z))**2))/torch.sqrt(torch.sum((z-z_t)**2))
    Lmax = (torch.max(Q, ub) - ub)**2
    Lmin = (torch.min(Q, lb) - lb)**2
    L = Lmax + Lmin
    return L

def IG_Loss_ZZ(netG, netIG, mb_size,Z_dim,z,use_cuda=True):
    interpolates = z
    if use_cuda:
        interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    G = netG(interpolates)
    z_hat = netIG(G)
    zN = z_hat.size()[1]
    #grad_list = []
    for i in range(zN):
        z_ = z_hat[:,i]
        ##print(pixel.size())
        gradients = autograd.grad(outputs=z_, inputs=interpolates,
                                  grad_outputs=torch.ones(G.size()[0]).cuda() if use_cuda else torch.ones(G.size()[0]),
                                  create_graph=True, retain_graph=True,
                                  only_inputs=True)[0]
        gradients= gradients.unsqueeze(2)
        if i == 0:
            grad = gradients
        else:
            grad = torch.cat((grad, gradients), 2)

    IG_det_penalty = 0
    ImN = grad.size()[0]
    for i in range(ImN):
        m = grad[i, :, :]
        teye = torch.eye(Z_dim).cuda() if use_cuda else torch.eye(Z_dim)
        tmp = m - teye
        IG_det_penalty += torch.trace(torch.mm(torch.t(tmp), tmp))

    """======== IG eigen penalty """
    ub = torch.Tensor([20.0])
    lb = torch.Tensor([1.0]) #Variable(
    if use_cuda:
        ub = ub.cuda()
        lb = lb.cuda()
    delta =torch.randn(mb_size, Z_dim) # Variable()
    if use_cuda:
        delta = delta.cuda()
    eps = 0.1
    delta = (delta/delta.norm(2))*eps
    z_t = z + delta
    Q = torch.sqrt(torch.sum((netIG(netG(z_t)) - z_hat)**2))/torch.sqrt(torch.sum((z_t - z)**2))
    Lmax = (torch.max(Q, ub) - ub)**2
    Lmin = (torch.min(Q, lb) - lb)**2
    IG_L = Lmax + Lmin
    return z_hat, IG_det_penalty, IG_L

def IGM_Loss_Geom_alt(netG, netIG, XVar, cnts, lambdas, Z_dim,z,adding_noise_var,use_G_var=True,use_cuda=True):
    #adding_noise_mat = np.random.normal(0.0, adding_noise_var, XVar.W.size())
    #W_bar = gaussian(XVar.W, 0.0, adding_noise_var)
    if use_G_var:
        V = XVar(z, cnts, lambdas)
        G = netG(z)
        G_var = torch.mul(V, torch.randn(V.size()).cuda())
        z_hat = netIG(G + G_var)
    # else:
    #     G = netG(z)
    #     z_hat, _ = netIG(G)

    interpolates = z
    if use_cuda:
        interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    G = netG(interpolates)
    #z_hat_G = netIG(G)

    #adding_noise_mat = np.random.normal(0.0, adding_noise_var, XVar.W.size())
    #W_bar = gaussian(XVar.W, 0.0, adding_noise_var)
    V = XVar(interpolates, cnts, lambdas)
    z_hat_GV = netIG(G+torch.mul(V, torch.randn(V.size()).cuda()))

    zN = z_hat_GV.size()[1]
    #z_hat = netIG(G)
    for i in range(zN):
        z_ = z_hat_GV[:,i]
        ##print(pixel.size())
        gradients = autograd.grad(outputs=z_, inputs=interpolates,
                                  grad_outputs=torch.ones(z_hat_GV.size()[0]).cuda() if use_cuda else torch.ones(z_hat_GV.size()[0]),
                                  create_graph=True, retain_graph=True,
                                  only_inputs=True)[0]
        gradients= gradients.unsqueeze(2)
        if i == 0:
            grad_GV = gradients
        else:
            grad_GV = torch.cat((grad_GV, gradients), 2)

    IG_det_penalty = 0
    teye = torch.eye(Z_dim).cuda() if use_cuda else torch.eye(Z_dim)
    ImN = grad_GV.size()[0]
    for i in range(ImN):
        gv = grad_GV[i, :, :]
        tmp = torch.mm((gv - teye),torch.t((gv - teye)))
        IG_det_penalty += torch.trace(tmp)
    return z_hat, IG_det_penalty



def IGM_Loss_Geom(netG, netIG, XVar, cnts, lambdas, Z_dim,z,adding_noise_var,use_G_var=True,use_cuda=True):
    #adding_noise_mat = np.random.normal(0.0, adding_noise_var, XVar.W.size())
    #W_bar = gaussian(XVar.W, 0.0, adding_noise_var)
    if use_G_var:
        V = XVar(z, cnts, lambdas)
        G = netG(z)
        G_var = torch.mul(V, torch.randn(V.size()).cuda())
        z_hat = netIG(G + G_var)
    else:
        z_hat = netG(z)
        z_hat = netIG(z_hat)

    interpolates = z
    if use_cuda:
        interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    G = netG(interpolates)
    z_hat_G = netIG(G)

    #adding_noise_mat = np.random.normal(0.0, adding_noise_var, XVar.W.size())
    #W_bar = gaussian(XVar.W, 0.0, adding_noise_var)
    V = XVar(interpolates, cnts, lambdas)
    z_hat_V = netIG(V)

    zN = z_hat_G.size()[1]
    #z_hat = netIG(G)
    for i in range(zN):
        z_ = z_hat_G[:,i]
        ##print(pixel.size())
        gradients = autograd.grad(outputs=z_, inputs=interpolates,
                                  grad_outputs=torch.ones(z_hat_G.size()[0]).cuda() if use_cuda else torch.ones(z_hat_G.size()[0]),
                                  create_graph=True, retain_graph=True,
                                  only_inputs=True)[0]
        gradients= gradients.unsqueeze(2)
        if i == 0:
            grad_G = gradients
        else:
            grad_G = torch.cat((grad_G, gradients), 2)

    for i in range(zN):
        z_ = z_hat_V[:,i]
        v_gradients = autograd.grad(outputs=z_, inputs=interpolates,
                                  grad_outputs=torch.ones(z_hat_V.size()[0]).cuda() if use_cuda else torch.ones(z_hat_V.size()[0]),
                                  create_graph=True, retain_graph=True,
                                  only_inputs=True)[0]
        v_gradients= v_gradients.unsqueeze(2)
        if i == 0:
            grad_V = v_gradients
        else:
            grad_V = torch.cat((grad_V, v_gradients), 2)

    IG_det_penalty = 0
    teye = torch.eye(Z_dim).cuda() if use_cuda else torch.eye(Z_dim)
    ImN = grad_G.size()[0]
    for i in range(ImN):
        gg = grad_G[i, :, :]
        vg = grad_V[i, :, :]
        tmp = torch.mm(gg, torch.t(gg)) + torch.mm(vg, torch.t(vg)) - torch.t(gg) - gg + teye
        IG_det_penalty += torch.trace(tmp)

    return z_hat,IG_det_penalty


def G_LLK_Geom(netG, XVar, cnts, lambdas, mb_size, z, iter_str='',
               llk_way='eig', adding_noise_var=0, diag_Reg_val=0.1,
               use_cuda=True):
    z_v = z.data.cpu().numpy()
    loglk = np.squeeze(np.sum(np.log(norm.pdf(z_v, 0.0, 1.0)),1))
    #print('loglk size')
    #print(loglk.size)
    rem_metric, JgJg_L, JsJs_L, sumeigvas_L = G_Jacobian_eig(netG,  XVar, cnts, lambdas,
                                                             z,iter_str, diag_Reg_val,
                                                             use_cuda)
    #print("sumeigvas_L size")
    #print(sumeigvas_L[0].size)
    llk_est = [loglk[i] - 0.5*sumeigvas_L[i] for i in range(mb_size)]
    #print(llk_est.shape())
    #llk_est_mean = torch.mean(torch.stack(llk_est))
    #print("llk_est 0 size ")
    #print(llk_est[0].size)

    llk_est = np.array(llk_est)
    #print("llk_est size")
    #print(llk_est.size)
    return llk_est, rem_metric, JgJg_L, JsJs_L


def Js_Jacobian(XVar, cnts, lambdas, z, use_cuda=True):
    interpolates = z
    if use_cuda:
        interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    V = XVar(interpolates, cnts, lambdas)
    V = V.view(V.size()[0], V.size()[1]*V.size()[2] * V.size()[3])

    PixN = V.size()[1]
    for i in range(PixN):
        sig = V[:,i]
        v_gradients = autograd.grad(outputs=sig, inputs=interpolates,
                                  grad_outputs=torch.ones(V.size()[0]).cuda() if use_cuda else torch.ones(V.size()[0]), create_graph=True, retain_graph=True,
                                  only_inputs=True)[0]
        v_gradients= v_gradients.unsqueeze(2)
        if i == 0:
            v_grad = v_gradients
        else:
            v_grad = torch.cat((v_grad, v_gradients), 2)

    jsjs_reg = 0
    ImN = v_grad.size()[0]
    for i in range(ImN):
        vg = v_grad[i, :, :]
        JsJs = torch.mm(vg, torch.t(vg))
        jsjs_reg += torch.pow(torch.det(JsJs)-1,2)
    return jsjs_reg


def G_Jacobian(netG,  XVar, cnts, lambdas, z,  iter_str='', adding_noise_var=0, diag_Reg_val=0.1, use_cuda=True):
    interpolates = z
    if use_cuda:
        interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    G = netG(interpolates)
    V = XVar(interpolates, cnts, lambdas)

    G = G.view(G.size()[0],G.size()[1]*G.size()[2]*G.size()[3])
    V = V.view(V.size()[0], V.size()[1]*V.size()[2] * V.size()[3])
    PixN = G.size()[1]
    for i in range(PixN):
        pixel = G[:,i]
        g_gradients = autograd.grad(outputs=pixel, inputs=interpolates,
                                  grad_outputs=torch.ones(G.size()[0]).cuda() if use_cuda else torch.ones(G.size()[0]), create_graph=True, retain_graph=True,
                                  only_inputs=True)[0]
        g_gradients= g_gradients.unsqueeze(2)
        if i == 0:
            g_grad = g_gradients
        else:
            g_grad = torch.cat((g_grad, g_gradients), 2)

    PixN = V.size()[1]
    for i in range(PixN):
        sig = V[:,i]
        v_gradients = autograd.grad(outputs=sig, inputs=interpolates,
                                  grad_outputs=torch.ones(V.size()[0]).cuda() if use_cuda else torch.ones(V.size()[0]), create_graph=True, retain_graph=True,
                                  only_inputs=True)[0]
        v_gradients= v_gradients.unsqueeze(2)
        if i == 0:
            v_grad = v_gradients
        else:
            v_grad = torch.cat((v_grad, v_gradients), 2)

    ImN = g_grad.size()[0]
    rem_metric = []
    JsJs_L = []
    JgJg_L = []
    M_penal = 0
    Log_Det = []
    for i in range(ImN):
        gg = g_grad[i, :, :]
        vg = v_grad[i, :, :]
        JgJg = torch.mm(gg, torch.t(gg))
        JsJs = torch.mm(vg, torch.t(vg))
        JsJs_L.append(JsJs)
        JgJg_L.append(JgJg)
        #list(JsJs.size())[0]
        diag_reg = torch.diag(
            torch.zeros(size=(list(JsJs.size())[0],), dtype=torch.float32, device='cuda') + diag_Reg_val)
        if diag_Reg_val !=0:
            #diag_reg = torch.diag(torch.zeros(size=(list(JsJs.size())[0],), dtype=torch.float32, device='cuda') + diag_Reg_val)
            # diag_reg = torch.diag(
            #     torch.ones(size=(list(JsJs.size())[0],), dtype=torch.float32, device='cuda') * diag_Reg_val)
            #diag_reg = torch.diag(gaussian(diag_reg,0.0,1e-3))
            jgjg_jsjs_sum = JgJg + JsJs + diag_reg
            rem_metric.append(jgjg_jsjs_sum)
            ##
            dets = torch.det(jgjg_jsjs_sum)
            tmp = torch.log(dets)

            if torch.isnan(tmp) or torch.isinf(tmp):
                print('G_Jacobian while loop where tmp after nan=%f' % tmp)
        else:
            jgjg_jsjs_sum = JgJg + JsJs
            rem_metric.append(jgjg_jsjs_sum)
            ##
            dets = torch.det(jgjg_jsjs_sum)
            tmp = torch.log(dets)
            #if tmp == float('nan'):
            if torch.isnan(tmp) or torch.isinf(tmp):
                print('tmp after nan=%f' % tmp)

        if torch.isnan(tmp):
            if iter_str !='':
                torch.save([JsJs,JgJg],'jgjs'+iter_str+'nan.pkl')


        if tmp == float('-Inf'):
            if iter_str != '':
                torch.save([JsJs, JgJg], 'jgjs'+iter_str+'neginf.pkl')

        if torch.isinf(tmp):
            if iter_str != '':
                torch.save([JsJs, JgJg], 'jgjs'+iter_str+'inf.pkl')

        #M_penal += tmp
        M_penal += torch.pow((dets-1),2)
        Log_Det.append(tmp)
        #M, Mg, Ms, _, Log_Det
    return rem_metric, JgJg_L, JsJs_L, M_penal, Log_Det


def G_LLK_Geom_G(netG, XVar, cnts, lambdas, mb_size, z, iter_str='',
               llk_way='eig', adding_noise_var=0, diag_Reg_val=0.1,
               use_cuda=True, grad_x_z=True):
    z_v = z.data.cpu().numpy()
    loglk = np.squeeze(np.sum(np.log(norm.pdf(z_v, 0.0, 1.0)), 1))
    #print('loglk')
    #print(loglk)

    interpolates = z
    if use_cuda:
        interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    G = netG(interpolates)
    PixN = G.size()[1]
    for i in range(PixN):
        pixel = G[:,i]
        g_gradients = autograd.grad(outputs=pixel, inputs=interpolates,
                                  grad_outputs=torch.ones(G.size()[0]).cuda() if use_cuda else torch.ones(G.size()[0]), create_graph=True, retain_graph=True,
                                  only_inputs=True)[0]
        g_gradients= g_gradients.unsqueeze(2)
        if i == 0:
            g_grad = g_gradients
        else:
            g_grad = torch.cat((g_grad, g_gradients), 2)

    ImN = g_grad.size()[0]
    rem_metric = []
    JgJg_L = []
    sumeigvas_L = []

    print('ImN={}'.format(ImN))
    print('mb_size={}'.format(mb_size))

    for i in range(ImN):
        gg = g_grad[i, :, :]
        #if grad_x_z == True:
        JgJg = torch.mm(gg, torch.t(gg))
        JgJg_L.append(JgJg)
        jgjg_jsjs_sum = JgJg
        rem_metric.append(jgjg_jsjs_sum)
        ith_rem = jgjg_jsjs_sum.data.cpu().numpy()
        eigvas, eigves = np.linalg.eigh(ith_rem)
        sum_eigvas = np.sum(np.log(eigvas))
        if diag_Reg_val !=0 and np.isinf(sum_eigvas):
            jgjg_jsjs_sum = JgJg
            rem_metric.append(jgjg_jsjs_sum)
            ith_rem = jgjg_jsjs_sum.data.cpu().numpy()
            eigvas, eigves = np.linalg.eigh(ith_rem)
            sum_eigvas = np.sum(np.log(eigvas))

            while np.isinf(sum_eigvas):
                ith_rem = jgjg_jsjs_sum.data.cpu().numpy()
                eigvas, eigves = np.linalg.eigh(ith_rem)
                sum_eigvas = np.sum(np.log(eigvas))
        sumeigvas_L.append(sum_eigvas)

    llk_est = [loglk[i] - 0.5 * sumeigvas_L[i] for i in range(mb_size)]
    #llk_est_mean = torch.mean(torch.stack(llk_est))
    llk_est = np.array(llk_est)
    return llk_est #, llk_est_mean


'''
in order to avoid to reduce the infinities from the calculations of determinants
'''
def G_Jacobian_eig(netG,  XVar, cnts, lambdas, z,  iter_str='', diag_Reg_val=0.1, use_cuda=True):
    #tmp_writer = open('jgjs.pkl','w')
    #print('the value of diag_Reg_val in G_Jacobian of util.py=%f' % diag_Reg_val)
    interpolates = z
    if use_cuda:
        interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    G = netG(interpolates)
    ##we can always add W_bar to XVar, namely, var_net. If no need to add noise, Var_Net will not add.
    ##W_bar = gaussian(XVar.W, 0.0, adding_noise_var)
    V = XVar(interpolates, cnts, lambdas)

    #G = G.view(G.size()[0],G.size()[1]*G.size()[2]*G.size()[3])
    #V = V.view(V.size()[0], V.size()[1]*V.size()[2] * V.size()[3])
    PixN = G.size()[1]
    for i in range(PixN):
        pixel = G[:,i]
        g_gradients = autograd.grad(outputs=pixel, inputs=interpolates,
                                  grad_outputs=torch.ones(G.size()[0]).cuda() if use_cuda else torch.ones(G.size()[0]), create_graph=True, retain_graph=True,
                                  only_inputs=True)[0]
        g_gradients= g_gradients.unsqueeze(2)
        if i == 0:
            g_grad = g_gradients
        else:
            g_grad = torch.cat((g_grad, g_gradients), 2)

    PixN = V.size()[1]
    for i in range(PixN):
        sig = V[:,i]
        v_gradients = autograd.grad(outputs=sig, inputs=interpolates,
                                  grad_outputs=torch.ones(V.size()[0]).cuda() if use_cuda else torch.ones(V.size()[0]), create_graph=True, retain_graph=True,
                                  only_inputs=True)[0]
        v_gradients= v_gradients.unsqueeze(2)
        if i == 0:
            v_grad = v_gradients
        else:
            v_grad = torch.cat((v_grad, v_gradients), 2)
    ImN = g_grad.size()[0]
    rem_metric = []
    JsJs_L = []
    JgJg_L = []
    sumeigvas_L = []
    for i in range(ImN):
        gg = g_grad[i, :, :]
        vg = v_grad[i, :, :]

        #if grad_x_z == True:
        #print(gg.size())
        JgJg = torch.mm(gg, torch.t(gg))
        JsJs = torch.mm(vg, torch.t(vg))
        # else:
        #     JgJg = torch.mm(torch.t(gg), gg)
        #     JsJs = torch.mm(torch.t(vg), vg)
        #print('JgJg')
        #print(JgJg)
        #print('JsJs')
        #print(JsJs)

        JsJs_L.append(JsJs)
        JgJg_L.append(JgJg)
        jgjg_jsjs_sum = JgJg + JsJs
        rem_metric.append(jgjg_jsjs_sum)
        ith_rem = jgjg_jsjs_sum.data.cpu().numpy()
        eigvas, eigves = np.linalg.eigh(ith_rem)
        # #print('eigvas')
        # #print(eigvas)
        sum_eigvas = np.sum(np.log(eigvas))
        diag_reg = torch.diag(
            torch.zeros(size=(list(JsJs.size())[0],), dtype=torch.float32, device='cuda') + diag_Reg_val)
        if diag_Reg_val !=0 and np.isinf(sum_eigvas):
            jgjg_jsjs_sum = JgJg + JsJs + diag_reg
            rem_metric.append(jgjg_jsjs_sum)
            ith_rem = jgjg_jsjs_sum.data.cpu().numpy()
            eigvas, eigves = np.linalg.eigh(ith_rem)
            sum_eigvas = np.sum(np.log(eigvas))

            while np.isinf(sum_eigvas):
            #while torch.isneginf(tmp):
                jgjg_jsjs_sum = jgjg_jsjs_sum + diag_reg
                ith_rem = jgjg_jsjs_sum.data.cpu().numpy()
                eigvas, eigves = np.linalg.eigh(ith_rem)
                sum_eigvas = np.sum(np.log(eigvas))
                ##print('in the G_Jacobian while loop where tmp is -inf when diag_Reg_val is not 0 where sum_eigvas=%f' % sum_eigvas)

        sumeigvas_L.append(sum_eigvas)
    return rem_metric, JgJg_L, JsJs_L,sumeigvas_L

def G_Jacobian_usediag(netG,  XVar, cnts, lambdas, z,  iter_str='', adding_noise_var=0, diag_Reg_val=0.1, use_cuda=True):
    #tmp_writer = open('jgjs.pkl','w')
    #print('the value of diag_Reg_val in G_Jacobian of util.py=%f' % diag_Reg_val)
    interpolates = z
    if use_cuda:
        interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    G = netG(interpolates)
    ##we can always add W_bar to XVar, namely, var_net. If no need to add noise, Var_Net will not add.
    ##W_bar = gaussian(XVar.W, 0.0, adding_noise_var)
    V = XVar(interpolates, cnts, lambdas)

    G = G.view(G.size()[0],G.size()[1]*G.size()[2]*G.size()[3])
    V = V.view(V.size()[0], V.size()[1]*V.size()[2] * V.size()[3])
    PixN = G.size()[1]
    for i in range(PixN):
        pixel = G[:,i]
        g_gradients = autograd.grad(outputs=pixel, inputs=interpolates,
                                  grad_outputs=torch.ones(G.size()[0]).cuda() if use_cuda else torch.ones(G.size()[0]), create_graph=True, retain_graph=True,
                                  only_inputs=True)[0]
        g_gradients= g_gradients.unsqueeze(2)
        if i == 0:
            g_grad = g_gradients
        else:
            g_grad = torch.cat((g_grad, g_gradients), 2)

    PixN = V.size()[1]
    for i in range(PixN):
        sig = V[:,i]
        v_gradients = autograd.grad(outputs=sig, inputs=interpolates,
                                  grad_outputs=torch.ones(V.size()[0]).cuda() if use_cuda else torch.ones(V.size()[0]), create_graph=True, retain_graph=True,
                                  only_inputs=True)[0]
        v_gradients= v_gradients.unsqueeze(2)
        if i == 0:
            v_grad = v_gradients
        else:
            v_grad = torch.cat((v_grad, v_gradients), 2)

    ImN = g_grad.size()[0]
    rem_metric = []
    JsJs_L = []
    JgJg_L = []
    M_penal = 0
    Log_Det = []
    for i in range(ImN):
        gg = g_grad[i, :, :]
        vg = v_grad[i, :, :]
        JgJg = torch.mm(gg, torch.t(gg))
        JsJs = torch.mm(vg, torch.t(vg))
        JsJs_L.append(JsJs)
        JgJg_L.append(JgJg)
        #list(JsJs.size())[0]
        diag_reg = torch.diag(
            torch.zeros(size=(list(JsJs.size())[0],), dtype=torch.float32, device='cuda') + diag_Reg_val)
        if diag_Reg_val !=0:
            #diag_reg = torch.diag(torch.zeros(size=(list(JsJs.size())[0],), dtype=torch.float32, device='cuda') + diag_Reg_val)
            # diag_reg = torch.diag(
            #     torch.ones(size=(list(JsJs.size())[0],), dtype=torch.float32, device='cuda') * diag_Reg_val)
            #diag_reg = torch.diag(gaussian(diag_reg,0.0,1e-3))
            jgjg_jsjs_sum = JgJg + JsJs + diag_reg
            rem_metric.append(jgjg_jsjs_sum)
            ##
            dets = torch.det(jgjg_jsjs_sum)
            tmp = torch.log(dets)

            while tmp == float('-Inf'):
            #while torch.isneginf(tmp):
                jgjg_jsjs_sum = jgjg_jsjs_sum + diag_reg
                dets = torch.det(jgjg_jsjs_sum)
                tmp = torch.log(dets)
                #print('in the G_Jacobian while loop where tmp is -inf when diag_Reg_val is not 0 where tmp=%f' % tmp)

            while torch.isinf(tmp):
                jgjg_jsjs_sum = jgjg_jsjs_sum - diag_reg
                dets = torch.det(jgjg_jsjs_sum)
                tmp = torch.log(dets)
                #print('in the G_Jacobian while loop where tmp is inf when diag_Reg_val is not 0 where tmp=%f' % tmp)
            if torch.isnan(tmp):
                jgjg_jsjs_sum = diag_reg
                dets = torch.det(jgjg_jsjs_sum)
                tmp = torch.log(dets)
                #print('G_Jacobian while loop where tmp after nan=%f' % tmp)
        else:
            jgjg_jsjs_sum = JgJg + JsJs
            rem_metric.append(jgjg_jsjs_sum)
            ##
            dets = torch.det(jgjg_jsjs_sum)
            tmp = torch.log(dets)
            #if tmp == float('nan'):
            if torch.isnan(tmp):
                jgjg_jsjs_sum = diag_reg
                dets = torch.det(jgjg_jsjs_sum)
                tmp = torch.log(dets)
                #print('tmp after nan=%f' % tmp)

        if torch.isnan(tmp):
            ##this means that determinant is 0
            #print('G_Jacobian ImN loop i=%d logdeterminant = %f' % (i, tmp))
            #print('G_Jacobian here the determinant is nan')
            #print('G_Jacobian the det = %f' % torch.det(jgjg_jsjs_sum).cpu().numpy())
            #print('G_Jacobian jgjg_jsjs_sum')

            #print(jgjg_jsjs_sum.data.cpu().numpy())
            # for j in range(len(rem_metric)):
            #     #print(rem_metric[j].data[0])
            #print('JgJg')
            #print(JgJg.data.cpu().numpy())
            # for j in range(list(JgJg.size())[0]):
            #     #print(JgJg_L[j].data[0])
            #print('JsJs')
            #print(JsJs.data.cpu().numpy())
            # for j in range(list(JsJs.size())[0]):
            #     #print(JsJs_L[j].data[0])
            #print('diag_reg')
            #print(diag_reg.data.cpu().numpy())
            if iter_str !='':
                torch.save([JsJs,JgJg],'jgjs'+iter_str+'nan.pkl')


        if tmp == float('-Inf'):
        #if torch.isneginf(tmp):
            #print('ImN loop i=%d logdeterminant = %f' % (i,tmp))
            #print('here the determinant is negative infinity')
            #print('jgjg_jsjs_sum')
            #print(jgjg_jsjs_sum.data[0])
            # for j in range(len(rem_metric)):
            #     #print(rem_metric[j].data[0])
            #print('JgJg')
            #print(JgJg.data[0])
            # for j in range(list(JgJg.size())[0]):
            #     #print(JgJg_L[j].data[0])
            #print('JsJs')
            #print(JsJs.data[0])
            # for j in range(list(JsJs.size())[0]):
            #     #print(JsJs_L[j].data[0])
            #print('diag_reg')
            #print(diag_reg.data[0])
            if iter_str != '':
                torch.save([JsJs, JgJg], 'jgjs'+iter_str+'neginf.pkl')

        if torch.isinf(tmp):
            #print('ImN loop i=%d logdeterminant = %f' % (i,tmp))
            #print('here the determinant is negative infinity')
            #print('jgjg_jsjs_sum')
            #print(jgjg_jsjs_sum.data[0])
            # for j in range(len(rem_metric)):
            #     #print(rem_metric[j].data[0])
            #print('JgJg')
            #print(JgJg.data[0])
            # for j in range(list(JgJg.size())[0]):
            #     #print(JgJg_L[j].data[0])
            #print('JsJs')
            #print(JsJs.data[0])
            # for j in range(list(JsJs.size())[0]):
            #     #print(JsJs_L[j].data[0])
            #print('diag_reg')
            #print(diag_reg.data[0])
            if iter_str != '':
                torch.save([JsJs, JgJg], 'jgjs'+iter_str+'inf.pkl')

        #M_penal += tmp
        M_penal += torch.pow((dets-1),2)
        Log_Det.append(tmp)
        #M, Mg, Ms, _, Log_Det
    return rem_metric, JgJg_L, JsJs_L, M_penal, Log_Det

def gaussian(ins, mean, stddev):
    #noise = Variable(ins.data.new(ins.size()).normal_(mean, stddev))
    noise = torch.tensor(ins.data.new(ins.size()).normal_(mean, stddev))
    return ins + noise

def load_mnist(dataset):
    data_dir = os.path.join("./data", dataset)

    def extract_data(filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
        return data

    data = extract_data(data_dir + '/train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
    trX = data.reshape((60000, 28, 28, 1))

    data = extract_data(data_dir + '/train-labels-idx1-ubyte.gz', 60000, 8, 1)
    trY = data.reshape((60000))

    data = extract_data(data_dir + '/t10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28)
    teX = data.reshape((10000, 28, 28, 1))

    data = extract_data(data_dir + '/t10k-labels-idx1-ubyte.gz', 10000, 8, 1)
    teY = data.reshape((10000))

    trY = np.asarray(trY).astype(np.int)
    teY = np.asarray(teY)

    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    y_vec = np.zeros((len(y), 10), dtype=np.float)
    for i, label in enumerate(y):
        y_vec[i, y[i]] = 1

    X = X.transpose(0, 3, 1, 2) / 255.
    # y_vec = y_vec.transpose(0, 3, 1, 2)

    X = torch.from_numpy(X).type(torch.FloatTensor)
    y_vec = torch.from_numpy(y_vec).type(torch.FloatTensor)
    return X, y_vec

def load_celebA(dir, transform, batch_size, shuffle):
    # transform = transforms.Compose([
    #     transforms.CenterCrop(160),
    #     transform.Scale(64)
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    # ])

    # data_dir = 'data/celebA'  # this path depends on your computer
    dset = datasets.ImageFolder(dir, transform)
    data_loader = torch.utils.data.DataLoader(dset, batch_size, shuffle)

    return data_loader


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    #print(net)
    #print('Total number of parameters: %d' % num_params)

def save_images(images, size, image_path):
    return imsave(images, size, image_path)

def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')

def generate_animation(path, num):
    images = []
    for e in range(num):
        img_name = path + '_epoch%03d' % (e+1) + '.png'
        images.append(imageio.imread(img_name))
    imageio.mimsave(path + '_generate_animation.gif', images, fps=5)

def loss_plot(hist, path = 'Train_hist.png', model_name = ''):
    x = range(len(hist['D_loss']))

    y1 = hist['D_loss']
    y2 = hist['G_loss']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(path, model_name + '_loss.png')

    plt.savefig(path)

    plt.close()

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

def KL_z_z_hat(IG, data, gen_size):
    data_size=data.dataset.train_data.shape[0]
    batch_total = int(data_size / data.batch_size)
    ind_arr = []
    for i in range(gen_size):
        s_ind = np.random.randint(0,batch_total)
        while s_ind in ind_arr:
            s_ind = np.random.randint(0, batch_total)
        ind_arr.append(s_ind)

    z_hat = []
    for it, (x_t, labels) in enumerate(data):
        if it in ind_arr:
            z_hat_it = IG(x_t)
            z_hat.append(z_hat_it)
    #print('the size of data=%d, gen_size=%d and ind_arr length=%d and z_hat length=%d' % (data_size, gen_size, len(ind_arr),len(z_hat)))
    z_hat = torch.cat(z_hat,0)
    z_dist = torch.distributions.normal.Normal(torch.tensor(float(0)).cuda(), torch.tensor(float(1)).cuda())
    kl_out_arr = []
    for i in range(z_hat.shape[1]):
        z_hat_dist = torch.distributions.normal.Normal(torch.mean(z_hat[:, i]), torch.std(z_hat[:, i]))
        kl_out = torch.distributions.kl_divergence(z_dist,z_hat_dist)
        kl_out_arr.append(kl_out)
    ##print('kl divergence=%f' % torch.mean(torch.stack(kl_out_arr,0)))
    return torch.mean(torch.stack(kl_out_arr,0))

def KL_z_z_hat_v2(IG, data, gen_size,kl_reg_iterr,opt):
    # ind_arr = []
    # for i in range(gen_size):
    #     s_ind = np.random.randint(0,batch_total)
    #     while s_ind in ind_arr:
    #         s_ind = np.random.randint(0, batch_total)
    #     ind_arr.append(s_ind)
    # z_hat = []
    # for it, (x_t, labels) in enumerate(data):
    #     if it in ind_arr:
    #         z_hat_it = IG(x_t)
    #         z_hat.append(z_hat_it)
    # if len(z_hat)==0:
    #     #print('debug')
    ##print('the size of data=%d, gen_size=%d and ind_arr length=%d and z_hat length=%d' % (data_size, gen_size, len(ind_arr), len(z_hat)))
    data_size=data.dataset.train_data.shape[0]
    batch_total = int(data_size / data.batch_size)
    X_Buf = []
    z_hat = []
    for it in range(gen_size):
        x_t, X_Buf = get_data_batch_iter(X_Buf, data.batch_size, kl_reg_iterr)
        if x_t is None and it < gen_size:
            kl_reg_iterr = iter(data)
            continue
        x_t = x_t.view(-1, opt.channels, opt.input_size, opt.input_size)
        z_hat_it = IG(x_t)
        z_hat.append(z_hat_it)
    #print('the size of data=%d, gen_size=%d and z_hat length=%d' % (data_size, gen_size, len(z_hat)))
    z_hat = torch.cat(z_hat,0).cuda()
    #z_dist = torch.distributions.normal.Normal(torch.tensor(float(0)).cuda(), torch.tensor(float(1)).cuda())
    kl_out_arr = []
    for i in range(z_hat.shape[1]):
        #z_hat_dist = torch.distributions.normal.Normal(torch.mean(z_hat[:, i]), torch.std(z_hat[:, i]))
        z_mean = torch.mean(z_hat[:, i])
        z_std = torch.std(z_hat[:, i])
        kl_out = float(torch.log(z_std))+torch.div(torch.add(torch.tensor(1.0), float(torch.pow(z_mean, 2))),
                  2 * float(torch.pow(z_std, 2))) - torch.tensor(1 / 2)
        kl_out_arr.append(kl_out)
    #print('kl divergence calculated=%f' % torch.mean(torch.stack(kl_out_arr,0))) ##based on equation from https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
    return torch.mean(torch.stack(kl_out_arr,0))




