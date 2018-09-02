from pandas.core.frame import DataFrame as df
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pickle
import string
from matplotlib.backends.backend_pdf import PdfPages

network_names = ['100-40-4', '28*6-4', '14*28-4']

network_id = 0
batch_size = 30
itemchoose = 0
dataitem_infix = {'Accuracy', 'Loss'}
network_suffix = {30: '_30', 1: ''}


with open('network_1.bin', 'rb') as f:
    d1 = pickle.load(f)
with open('network_2.bin', 'rb') as f:
    d2 = pickle.load(f)
with open('network_3.bin', 'rb') as f:
    d3 = pickle.load(f)

with open('network_1' + network_suffix[batch_size] + '.bin', 'rb') as f:
    d1_10 = pickle.load(f)
with open('network_2' + network_suffix[batch_size] + '.bin', 'rb') as f:
    d2_10 = pickle.load(f)
with open('network_3' + network_suffix[batch_size] + '.bin', 'rb') as f:
    d3_10 = pickle.load(f)

d1.pop('network')
d2.pop('network')
d3.pop('network')
d1_10.pop('network')
d2_10.pop('network')
d3_10.pop('network')

data = dict()
data[network_names[0] + '_train_loss'] = d1.pop('train_l_r')
data[network_names[0] + '_train_accuracy'] = d1.pop('train_a_r')
data[network_names[0] + '_test_loss'] = d1.pop('test_l_r')
data[network_names[0] + '_test_accuracy'] = d1.pop('test_a_r')
data[network_names[1] + '_train_loss'] = d2.pop('train_l_r')
data[network_names[1] + '_train_accuracy'] = d2.pop('train_a_r')
data[network_names[1] + '_test_loss'] = d2.pop('test_l_r')
data[network_names[1] + '_test_accuracy'] = d2.pop('test_a_r')
data[network_names[2] + '_train_loss'] = d3.pop('train_l_r')
data[network_names[2] + '_train_accuracy'] = d3.pop('train_a_r')
data[network_names[2] + '_test_loss'] = d3.pop('test_l_r')
data[network_names[2] + '_test_accuracy'] = d3.pop('test_a_r')

data[network_names[0] + '_train_loss'+ network_suffix[batch_size]] = d1_10.pop('train_l_r')
data[network_names[0] + '_train_accuracy'+ network_suffix[batch_size]] = d1_10.pop('train_a_r')
data[network_names[0] + '_test_loss'+ network_suffix[batch_size]] = d1_10.pop('test_l_r')
data[network_names[0] + '_test_accuracy'+ network_suffix[batch_size]] = d1_10.pop('test_a_r')
data[network_names[1] + '_train_loss'+ network_suffix[batch_size]] = d2_10.pop('train_l_r')
data[network_names[1] + '_train_accuracy'+ network_suffix[batch_size]] = d2_10.pop('train_a_r')
data[network_names[1] + '_test_loss'+ network_suffix[batch_size]] = d2_10.pop('test_l_r')
data[network_names[1] + '_test_accuracy'+ network_suffix[batch_size]] = d2_10.pop('test_a_r')
data[network_names[2] + '_train_loss'+ network_suffix[batch_size]] = d3_10.pop('train_l_r')
data[network_names[2] + '_train_accuracy'+ network_suffix[batch_size]] = d3_10.pop('train_a_r')
data[network_names[2] + '_test_loss'+ network_suffix[batch_size]] = d3_10.pop('test_l_r')
data[network_names[2] + '_test_accuracy'+ network_suffix[batch_size]] = d3_10.pop('test_a_r')

data_frame = df(data=data, index=np.arange(0,50)*2500/50)

plt.rc('lines', linewidth=2)
plt.rc('font', size=12)
plt.ioff()


for network_id in [0, 1, 2]:
    for batch_size in [1, 30]:
        for infix in dataitem_infix:
            fig, ax = plt.subplots()
            ax.plot(data_frame.index, data_frame[network_names[network_id] + '_train_' + infix.lower() + network_suffix[batch_size]], label='Train')
            ax.plot(data_frame.index, data_frame[network_names[network_id] + '_test_' + infix.lower() + network_suffix[batch_size]], label='Test')
            if infix is 'Accuracy':
                ax.set_ylim([0,1])
            else:
                ax.set_ylim([0,2])

            plt.title(infix + ' for ' + network_names[network_id] + ' network, batchsize=' + str(batch_size))
            plt.xlabel('Iterations')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.show()
#            with PdfPages('figs/'+network_names[network_id]+'_'+infix.lower()+network_suffix[batch_size]+ '.pdf') as pp:
#                pp.savefig(fig, bbox_inches='tight')
            plt.close(fig)

for network_id in [0, 1, 2]:
    fig, ax = plt.subplots()
    ax.plot(data_frame.index,
            data_frame[network_names[network_id] + '_test_accuracy' + network_suffix[1]],
            label='Size=1, Test')
    ax.plot(data_frame.index,
            data_frame[network_names[network_id] + '_test_accuracy' + network_suffix[30]],
            label='Size=30, Test')
    ax.plot(data_frame.index,
            data_frame[network_names[network_id] + '_train_accuracy' + network_suffix[1]],
            label='Size=1, Train')
    ax.plot(data_frame.index,
            data_frame[network_names[network_id] + '_train_accuracy' + network_suffix[30]],
            label='Size=30, Train')
    ax.set_ylim([0, 1])

    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
#    with PdfPages('figs/batch_' + network_names[network_id] + '.pdf') as pp:
#        pp.savefig(fig, bbox_inches='tight')
    plt.close(fig)

