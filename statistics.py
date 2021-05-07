import torch
import numpy as np
import matplotlib.pyplot as plt

dy, dx = 50, 50
rep = 15
epochs = 50
save = True

acc = np.zeros((epochs, rep, 4))
x = np.arange(epochs)
# 00 for 75/25 split
# 0 for 50/50 split

for r in range(rep):
    acc[:, r, 0] = np.loadtxt("02test_acc"+str(r)+".txt")*100 #phase
    acc[:, r, 1] = np.loadtxt("rbm_test_acc"+str(r)+".txt")*100 #gauss
    acc[:, r, 2] = np.loadtxt("01test_acc"+str(r)+".txt")*100 #mag
    acc[:, r, 3] = np.loadtxt("00test_acc"+str(r)+".txt")*100 #mag+phase+gauss


if save:
    dsvp1 = np.std(acc[epochs-1,:,0])
    dsvp2 = np.std(acc[epochs-1,:,1])
    dsvp3 = np.std(acc[epochs-1,:,2])
    dsvp4 = np.std(acc[epochs-1,:,3])

    dsp = np.zeros((2, 4))
    dsp[1, 0] = dsvp1
    dsp[1, 1] = dsvp2
    dsp[1, 2] = dsvp3
    dsp[1, 3] = dsvp4

    dsp[0, 0] = acc[epochs-1,:,0].mean(-1)
    dsp[0, 1] = acc[epochs-1,:,1].mean(-1)
    dsp[0, 2] = acc[epochs-1,:,2].mean(-1)
    dsp[0, 3] = acc[epochs-1,:,3].mean(-1)

    np.savetxt('res_75.txt', dsp)
    np.savetxt('wil_75.txt', acc[epochs-1,:,:].reshape((rep, 4)))

acc = np.mean(acc, axis=1)

print("Acc", np.round(acc[epochs-1, 0],2), np.round(acc[epochs-1, 1],2), np.round(acc[epochs-1, 2],2), np.round(acc[epochs-1, 3],2))


for _ in range(2):
    fig, ax = plt.subplots()
    #ax.errorbar(x, acc, dsvp, linestyle='-', linewidth=.5, label='MultFRRBM', marker='.', color='red', ecolor='black')
    ax.plot(x, acc[:,0], linestyle='-', linewidth=1, label='MultFRRBM-P',color='red') 
    ax.plot(x, acc[:,1], linestyle=':', linewidth=1, label='GaussianRBM',color='blue')
    ax.plot(x, acc[:,2], linestyle='--', linewidth=1, label='MultFRRBM-M', color='green')
    ax.plot(x, acc[:,3], linestyle='-.', linewidth=1, label='MultFRRBM-PM', color='black') #, marker='+'

    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    ax.set_xlabel("Epochs", fontsize=14)
    ax.set_ylabel("Mean Accuracy", fontsize=14)
    ax.legend(loc='lower right', fontsize=14)
    ax.grid(True)

    axins=ax.inset_axes([0.2, 0.2, .3, .3])
    axins.plot(x[47:], acc[47:,0], linestyle='-', linewidth=.75, color='red')
    axins.plot(x[47:], acc[47:,1], linestyle=':', linewidth=.75, color='blue')
    axins.plot(x[47:], acc[47:,2], linestyle='--', linewidth=.75, color='green')
    axins.plot(x[47:], acc[47:,3], linestyle='-.', linewidth=.75, color='black')
    axins.set_xticklabels('')
    #axins.set_yticklabels('')
    ax.indicate_inset_zoom(axins, linewidth=0.75)

    plt.savefig("mean_acc_75.eps", bbox_inches='tight')
    #plt.savefig("mean_acc_50.eps", bbox_inches='tight')
    plt.show()

