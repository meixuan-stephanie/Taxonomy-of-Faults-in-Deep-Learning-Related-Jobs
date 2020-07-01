# Part II Wrong code(Please note thet only partial code is included for the emphasis of topics)

# @hyperparameters examples

learning_rate=0.01
epochs=10
my_batch_size=12

my_model = build_model(learning_rate)
trained_weight, trained_bias, epochs, rmse = train_model(my_model, my_feature, 
                                                         my_label, epochs,
                                                         my_batch_size)
plot_the_model(trained_weight, trained_bias, my_feature, my_label)
plot_the_loss_curve(epochs, rmse)

#output
#Epoch 1/10
#1/1 [==============================] - 0s 1ms/step - loss: 1063.3500 - root_mean_squared_error: 32.6090
#Epoch 2/10
#1/1 [==============================] - 0s 1ms/step - loss: 1046.4030 - root_mean_squared_error: 32.3482
#...
#Epoch 9/10
#1/1 [==============================] - 0s 955us/step - loss: 985.6248 - root_mean_squared_error: 31.3947
#Epoch 10/10
#1/1 [==============================] - 0s 1ms/step - loss: 979.1053 - root_mean_squared_error: 31.2907

#Examine the output above,the loss is not coverging to stay steady which indicates that some hyperparameters are suboptimal. Since the traing loss didnot converge by inspection. One possible solution is to train for more epochs.
#Please refer to the same topic in another file named correct_code.py
