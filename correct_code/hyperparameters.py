# Part III Correct code (Please note that only partial code is included for the emphasis of topics and the specific part needs to be changed and has been changed is marked with a sign and descriptions)

# @hyperparameters examples

#1
learning_rate=0.01
epochs=450 #the value is changed
my_batch_size=12 

my_model = build_model(learning_rate)
trained_weight, trained_bias, epochs, rmse = train_model(my_model, my_feature, 
                                                         my_label, epochs,
                                                         my_batch_size)
plot_the_model(trained_weight, trained_bias, my_feature, my_label)
plot_the_loss_curve(epochs, rmse)

#output
#Epoch 1/450
#1/1 [==============================] - 0s 3ms/step - loss: 206.7690 - root_mean_squared_error: 14.3795
#Epoch 2/450
#1/1 [==============================] - 0s 3ms/step - loss: 199.3452 - root_mean_squared_error: 14.1190
#Epoch 3/450
#1/1 [==============================] - 0s 3ms/step - loss: 194.0909 - root_mean_squared_error: 13.9317
#...
#Epoch 448/450
#1/1 [==============================] - 0s 2ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
#Epoch 449/450
#1/1 [==============================] - 0s 2ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
#Epoch 450/450
#1/1 [==============================] - 0s 2ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
#As you can see, our output loss is greatly reduced by increasing number of epochs. There's one trick that the training may have not been finished when the loss curve has not converged or the loss value is quite large. The most efficient way to resolve the issue is that you increase the number of epochs to have the model trained sufficiently.

#2
learning_rate=0.01 #this value is changed
epochs=500
my_batch_size=12

my_model = build_model(learning_rate)
trained_weight, trained_bias, epochs, rmse = train_model(my_model, my_feature, 
                                                         my_label, epochs,
                                                         my_batch_size)
plot_the_model(trained_weight, trained_bias, my_feature, my_label)

#output
#Epoch 1/500
#1/1 [==============================] - 0s 1ms/step - loss: 960.9570 - root_mean_squared_error: 30.9993
#Epoch 2/500
#1/1 [==============================] - 0s 1ms/step - loss: 944.8501 - root_mean_squared_error: 30.7384
#...
#Epoch 284/500
#1/1 [==============================] - 0s 1ms/step - loss: 62.5916 - root_mean_squared_error: 7.9115
#...
#Epoch 499/500
#1/1 [==============================] - 0s 1ms/step - loss: 0.9666 - root_mean_squared_error: 0.9832
#Epoch 500/500
#1/1 [==============================] - 0s 1ms/step - loss: 0.9638 - root_mean_squared_error: 0.9817

#That learning rate value only serves as a reference to show you that we usually set a small value for learning rate. But please note that there's always an ideal combination among those parameters and you should keep tuning them to find a better one.

#3
learning_rate=0.05
epochs=100
my_batch_size=1 # Wow, a batch size of 1 works!

my_model = build_model(learning_rate)
trained_weight, trained_bias, epochs, rmse = train_model(my_model, my_feature, 
                                                         my_label, epochs,
                                                         my_batch_size)
plot_the_model(trained_weight, trained_bias, my_feature, my_label)
plot_the_loss_curve(epochs, rmse)

#output
#Epoch 1/125
#12/12 [==============================] - 0s 1ms/step - loss: 1050.8781 - root_mean_squared_error: 32.4172
#Epoch 2/125
#12/12 [==============================] - 0s 969us/step - loss: 764.8110 - root_mean_squared_error: 27.6552
#Epoch 3/125
#12/12 [==============================] - 0s 969us/step - loss: 576.5870 - root_mean_squared_error: 24.0122
#...
#Epoch 124/125
#12/12 [==============================] - 0s 1ms/step - loss: 1.1272 - root_mean_squared_error: 1.0617
#Epoch 125/125
#12/12 [==============================] - 0s 1ms/step - loss: 1.0034 - root_mean_squared_error: 1.0017

#As you can see, the loss value is reduced and please remeber that the loss value in examples above may be further improved if you try many times to find an ideal combination among those hyperparameters. All corrected codes in this file is to give you an overview about how those parameters can be changed to make a difference
#Finally, there's a summary about hyparameter tuning:
#Most machine learning problems require a lot of hyperparameter tuning.However, we can't give a rule of thumb for all models as every model is different. Therefore, you must experiment enough to find the best set of parameters stated in your model.
#What I can provide are some general principles you may mostly use in your future ML programming jobs

#Training loss should steadily decrease, steeply at first, and then more slowly until the slope of the curve reaches or approaches zero.
#If the training loss does not converge, train for more epochs.
#If the training loss decreases too slowly, increase the learning rate. Note that setting the training loss too high may also prevent training loss from converging.
#If the training loss varies wildly (that is, the training loss jumps around), decrease the learning rate.
#Lowering the learning rate while increasing the number of epochs or the batch size is often a good combination.
#Setting the batch size to a very small batch number can also cause instability. First, try large batch size values. Then, decrease the batch size until you see degradation.
#For real-world datasets consisting of a very large number of examples, the entire dataset might not fit into memory. In such cases, you'll need to reduce the batch size to enable a batch to fit into memory.


#Special thanks to google education for inspiring me on knowledge in this chapter
