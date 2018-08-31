import csv
import numpy as np
import re

# the function of read data from given csv file
def read_data(file_name):
    data_set = []
    # regularization to not read the heading column
    pattern = re.compile(r'^[-+]?[-0-9]\d*\.\d*|[-+]?\.?[0-9]\d*$')
    with open(file_name, newline='') as csvfile:
        data_file = csv.reader(csvfile)
        for row in data_file:
            temp = []
            for x in list(row):
                result = pattern.match(x)
                if result:
                    temp.append(float(x))
            data_set.append(temp)

    return np.array(data_set,dtype=np.float64)

# for loop to calculate the w_hat and b_hat of the second neural network
for i in range(97,102):
    # ASCII for controlling characters from a to e
    # formatted output
    a_b = read_data('../data/question_1/%c/%c_b.csv'%(i,i))
    a_w = read_data('../data/question_1/%c/%c_w.csv'%(i,i))
    b1=np.transpose(a_b[:1].reshape(5,1))
    b2=np.transpose(a_b[1:2].reshape(5,1))
    b3=np.transpose(a_b[2:3].reshape(5,1))
    print(b1)
    print(b2)
    print(b3)
    w1=np.transpose(a_w[:5].reshape(5,5))
    w2=np.transpose(a_w[5:10].reshape(5,5))
    w3=np.transpose(a_w[10:15].reshape(5,5))
    print(w1)
    print(w2)
    print(w3)
    w_hat=np.transpose(np.dot(np.dot(w1,w2),w3))
    b_hat=np.dot(np.dot(b1,w2),w3)+np.dot(b2,w3)+b3
    # output the results into Question_1
    np.savetxt('../Question_1/%c-w.csv'%i, w_hat, delimiter = ',')
    np.savetxt('../Question_1/%c-b.csv'%i, b_hat, delimiter = ',')