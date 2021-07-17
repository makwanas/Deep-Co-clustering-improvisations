from __future__ import division
import os
import tensorflow as tf
import core.paegmm.kddcup10.kddcup10_pae_gmm as paegmm
import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

if __name__ == '__main__':
    print("-----------Path----------")
    print(os.getcwd())
    # data_path
    Coil20 =['/nfs/stak/users/busarapn/Final_project_cs535/Deep-Co-Clustering-master/DeepCC/Data/Yale_32x32.mat',10,'Yale_32x32']

    filename   = Coil20[0]
    num_clus_r = Coil20[1]
    num_clus_c = Coil20[1]

    ae_config = [1024, 500, 200, 100, 40]

    ae_col_config = [165, 100, 40]

    gmm_config = [[num_clus_r, 5], 48, 160, 80, 40, num_clus_r]

    accuracy     = []
    NMI          = []
    loss         = []
    loss_col     = []

    rounds = 1
    epochs = 1000
    epochs_pretrain = 1000
    for k in range(rounds):
        tf.reset_default_graph()
        machine = paegmm.KddcupPaeGmm(1024, num_clus_r, num_clus_c, ae_config, ae_col_config, gmm_config, 0)
        data = machine.get_data(filename)
        acc, nmi, loss, los_col = machine.run(data, epochs, epochs_pretrain)

        accuracy      = np.append(accuracy, acc)
        NMI           = np.append(NMI, nmi)
        loss          = np.append(loss, loss)
        loss_col      = np.append(loss_col, los_col)

    f_acc = open('relu_accuracy.txt', 'w')
    f_acc.write(str(accuracy))
    f_acc.close()

    f_nmi = open('relu_nmi.txt', 'w')
    f_nmi.write(str(NMI))
    f_nmi.close()

    f_loss = open('relu_loss.txt', 'w')
    f_loss.write(str(loss))
    f_loss.close()

    f_loss_col = open('relu_loss_col.txt', 'w')
    f_loss_col.write(str(loss_col))
    f_loss.close()
