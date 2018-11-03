import numpy as np
from read_pdb_file import read_pdb
import matplotlib.pyplot as plt

def plot_distribution_range_xyz():
    num=[]
    x=[]
    y=[]
    z=[]
    for i in range(1, 3001):
        print(i)
        X_list, Y_list, Z_list, atomtype_list = read_pdb('../data/training_data/%s_pro_cg.pdb' % str(i).zfill(4))
        num.append((len(X_list)))
        x.append(max(X_list)-min(X_list))
        y.append(max(Y_list) - min(Y_list))
        z.append(max(Z_list) - min(Z_list))

    fig = plt.figure(dpi=128, figsize=(12, 6))
    fig.autofmt_xdate()
    fig.suptitle('Spatial Range VS Number of Atom within a Protein')

    ax = plt.subplot(131)
    ax.set_title('X Range VS Number of Atoms')
    ax.set_ylabel('X')
    ax.set_xlabel('The Numbers of Atoms within a Protein')
    ax.scatter(num, x, label='Protein')

    plt.legend()
    plt.grid(True)


    ax = plt.subplot(132)
    ax.set_title('Y Range VS Number of Atoms')
    ax.set_ylabel('Y')
    ax.set_xlabel('The Numbers of Atoms within a Protein')
    ax.scatter(num, y, label='Protein')
    plt.legend()
    plt.grid(True)


    ax = plt.subplot(133)
    ax.set_title('Z Range VS Number of Atoms')
    ax.set_ylabel('Z')
    ax.set_xlabel('The Numbers of Atoms within a Protein')
    ax.scatter(num, z, label='Protein')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.show()

#plot_distribution_range_xyz()


x=[1,2,3,4,5,6,7,8,9,10]
y1=[0.45, 0.324, 0.302, 0.288, 0.278, 0.269, 0.260, 0.256, 0.253, 0.25]
y2=[0.36, 0.3, 0.382, 0.345, 0.286,0.32, 0.31, 0.303, 0.286, 0.278]
plt.figure(figsize=(8, 5))
plt.plot(x, y1, label='train_loss')
plt.plot(x, y2, label='valid_loss')
plt.title('Loss vs Epochs in Training and Validation Set for LSTM')
plt.xlabel('Epochs')
plt.ylabel('Loss(MSE)')
x_label = range(1, 11)
plt.xticks(x_label)
plt.legend()
plt.grid()
plt.savefig('../data/result/lstm_validation.jpg', dpi=200)