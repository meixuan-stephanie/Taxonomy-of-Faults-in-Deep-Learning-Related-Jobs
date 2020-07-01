# Part III Correct code (Please note that only partial code is included for the emphasis of topics and the specific part needs to be changed and has been changed is marked with a sign and descriptions)

# @hyperparameters examples
learning_rate=0.01
epochs=450 #the value is changed
my_batch_size=12 

my_model = build_model(learning_rate)
trained_weight, trained_bias, epochs, rmse = train_model(my_model, my_feature, 
                                                         my_label, epochs,
                                                         my_batch_size)
plot_the_model(trained_weight, trained_bias, my_feature, my_label)
plot_the_loss_curve(epochs, rmse)

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

