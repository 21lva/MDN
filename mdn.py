import matplotlib.pyplot as plt
import numpy as np

def DataSet(n_samples=1000,n_output=3):
    """
    dimension : 2
    """
    noise = np.random.normal(size=n_samples)
    xdata = np.random.uniform(-10.0,10.0,n_samples)
    ydata = 5*np.sin(0.5*xdata)+0.5*xdata+noise
    return xdata,ydata

class MDN():
    def __init__(self,sess,dim_input=1,n_hidden=20,dim_output=1,n_component=3,learning_rate=0.001):

        self.sess = sess
        self.dim_input = dim_input
        self.n_hidden = n_hidden
        self.dim_output = dim_output
        self.learning_rate = learning_rate

        tf.Variable_scope("NN"):
            self.input = tf.placeholder(tf.float32,shape = [None,self.dim_input])
            self.output = tf.placeholder(tf.float32,shape=[None,self.dim_output])
            
            self.hidden = tf.layers.Dense(self.input,self.n_hidden,activation=tf.nn.sigmoid)
            
            self.mixing_pred = tf.layers.Dense(self.hidden,n_component,activation = tf.nn.softmax)
            self.mean_pred = tf.layers.Dense(self.hidden,n_component*dim_output)
            self.std_pred = tf.layers.Dense(self.hidden,n_component*dim_output,activation = tf.nn.relu)

        tf.Variable_scope("prob"):
            mother = np.reciprocal(self.std_pred*np.sqrt(2*np.pi))
            exp = np.exp((-0.5)*np.square(self.output-self.mean_pred)*np.reciprocal(np.square(tf.self.std_pred)))
            N = exp*mother
            self.prob = tf.math.reduce_sum(N*self.mixing_pred)
            self.loss = (-1)*np.log(self.prob)

        self.opt= tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train = self.opt.minimize(self.loss)

        self.sess.run(tf.global_variables_initializer())
    
    def train_epoch(self,X,Y):
        self.sess.run(train,feed_dict={self.input:X,self.output:Y})

    def predict(self,Input):
        Mixing_coef= self.sess.run(self.mixing_pred,feed_dict={self.input:Input})
        Mean = self.sess.run(self.mean_pred,feed_dict={self.input:Input})
        Std= self.sess.run(self.std_pred,feed_dict={self.input:Input})

        return 





x,y = DataSet()
plt.figure()
plt.scatter(x,y)
plt.show()

    
    
