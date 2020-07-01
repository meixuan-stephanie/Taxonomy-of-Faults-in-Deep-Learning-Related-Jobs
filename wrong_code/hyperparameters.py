# Part II Wrong code
#Please note thet only partial code is included for the emphasis of topics and the specific part needs to be changed and has been changed is marked with a sign and descriptions

# @hyperparameters examples

#1
learning_rate=0.01
epochs=10 #need to be fixed
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
#Please refer to the same topic in another folder named correct_code

#You may want your models to converge more quickly when your epochs number is set large
#however, setting the learning rate too large often makes it impossible for a model to converge

#2
learning_rate=100 #needs to be fixed
epochs=500 

my_model = build_model(learning_rate)
trained_weight, trained_bias, epochs, rmse = train_model(my_model, my_feature, 
                                                         my_label, epochs,
                                                         my_batch_size)
plot_the_model(trained_weight, trained_bias, my_feature, my_label)
plot_the_loss_curve(epochs, rmse)

#output
#Epoch 1/500
#1/1 [==============================] - 0s 1ms/step - loss: 627.3366 - root_mean_squared_error: 25.0467
#Epoch 2/500
#1/1 [==============================] - 0s 1ms/step - loss: 6686614.0000 - root_mean_squared_error: 2585.8489
#...
#Epoch 24/500
#1/1 [==============================] - 0s 1ms/step - loss: 0.8938 - root_mean_squared_error: 0.9454
#Epoch 25/500
#1/1 [==============================] - 0s 1ms/step - loss: 0.8908 - root_mean_squared_error: 0.9438
#Epoch 26/500
#1/1 [==============================] - 0s 967us/step - loss: 0.8883 - root_mean_squared_error: 0.9425
#...
#Epoch 79/500
#1/1 [==============================] - 0s 1ms/step - loss: 182627.7500 - root_mean_squared_error: 427.3497
#Epoch 80/500
#1/1 [==============================] - 0s 1ms/step - loss: 220037.1875 - root_mean_squared_error: 469.0812
#Epoch 81/500
#1/1 [==============================] - 0s 1ms/step - loss: 244864.9531 - root_mean_squared_error: 494.8383
#Epoch 82/500
#1/1 [==============================] - 0s 948us/step - loss: 246719.9844 - root_mean_squared_error: 496.7092
#Epoch 83/500
#1/1 [==============================] - 0s 1ms/step - loss: 227311.1250 - root_mean_squared_error: 476.7716
#Epoch 84/500
#1/1 [==============================] - 0s 965us/step - loss: 197832.0625 - root_mean_squared_error: 444.7832
#...
#Epoch 497/500
#1/1 [==============================] - 0s 1ms/step - loss: 170417.5156 - root_mean_squared_error: 412.8166
#Epoch 498/500
#1/1 [==============================] - 0s 2ms/step - loss: 170417.7031 - root_mean_squared_error: 412.8168
#Epoch 499/500
#1/1 [==============================] - 0s 1ms/step - loss: 170417.7500 - root_mean_squared_error: 412.8168
#Epoch 500/500
#1/1 [==============================] - 0s 2ms/step - loss: 170417.8125 - root_mean_squared_error: 412.8169

#Examine the output above, you may find that the loss value is kind of fluctuating and the final result of loss value is still large. YThus, we may kniow that the model is not our expoectation although learning speed is high.
#Please refer to the same topic in another folder named correct_code

#Sometimes your batch size value may also have an effect on the final loss value
#3
learning_rate=0.05
epochs=100
my_batch_size= 12  # can to be changed

my_model = build_model(learning_rate)
trained_weight, trained_bias, epochs, rmse = train_model(my_model, my_feature, 
                                                        my_label, epochs,
                                                        my_batch_size)
plot_the_model(trained_weight, trained_bias, my_feature, my_label)
plot_the_loss_curve(epochs, rmse)

#output
#Epoch 1/100
#1/1 [==============================] - 0s 2ms/step - loss: 1055.5636 - root_mean_squared_error: 32.4894
#...
#Epoch 99/100
#1/1 [==============================] - 0s 1ms/step - loss: 1.5894 - root_mean_squared_error: 1.2607
#Epoch 100/100
#1/1 [==============================] - 0s 1ms/step - loss: 1.5487 - root_mean_squared_error: 1.2445
#you may find that the loss value can be further reduced
#Please refer to the same topic in another folder named correct_code
