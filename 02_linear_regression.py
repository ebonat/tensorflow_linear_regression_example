# TensorFlow-Tutorials
# https://github.com/nlintz/TensorFlow-Tutorials

# Linear Regression
# https://github.com/nlintz/TensorFlow-Tutorials/blob/master/01_linear_regression.py

import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
    
def main():    
#     create a X_training numpy array between -1 and 1 with 101 values 
    X_train = np.linspace(start =-1, stop=1, num=101)
#     print(X_train)
#     print()
    
#     create a Y_training numpy array which is approximately linear but with some random noise
#     the slope is equal 2 (y = m * x + b; m = 2
    Y_train = 2 * X_train + np.random.randn(*X_train.shape) * 0.33     
#     print(Y_train)
#     print()
     
#     show the scatter plot for X_train an Y_train
    plt.scatter(X_train, Y_train,  color='black')
    plt.title('Linear Regression - TensorFlow')
    plt.xlabel('X_train')
    plt.ylabel('Y_train')
    plt.tight_layout()
#     plt.show()
      
#      TENSORFLOW DECLARATIONS
#     create X and Y symbolic variables
    X = tf.placeholder("float") 
    Y = tf.placeholder("float")
 
#     create a shared variable for the weight matrix        
    w = tf.Variable(0.0, name="weights")
         
#     create the y multiplication model as X * w
    y_model = tf.multiply(X, w)
      
#     use square error for cost function
    cost = tf.square(Y - y_model)   
     
#     construct an optimizer to minimize cost and fit line to the data
    train_optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost) 
     
#     TENSORFLOW  WORKS
#     launch the graph in a session
    with tf.Session() as session:
         
#         need to initialize the variables w
        tf.global_variables_initializer().run()
      
        for i in range(100):
            for (x, y) in zip(X_train, Y_train):
                session.run(train_optimizer, feed_dict={X: x, Y: y})
                 
#         w predicted variable value
        w_predicted = session.run(w)
         
#     frees all resources associated with the session
    session.close()
     
    print("Predicted slope: {} ".format(w_predicted))
     
#     build the predicted model
    Y_predict = w_predicted * X_train    

#     plot the line
    plt.plot(X_train, Y_predict, color='red', linewidth=3)
    plt.show()
    
#     need to show the final regression analysis parameters!

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print("Program Runtime: " + str(round(end_time - start_time, 1)) + " seconds" + "\n")