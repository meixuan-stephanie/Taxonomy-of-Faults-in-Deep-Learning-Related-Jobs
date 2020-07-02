# Part III Correct code (Please note that only partial code is included for the emphasis of topics and the specific part needs to be changed and has been changed is marked with a sign and some marked descriptions are explanations)

# @validation and testing related examples

#1
learning_rate = 0.08
epochs = 70
batch_size = 100

# Split the original training set into a reduced training set and a
# validation set. 
validation_split=0.2

my_feature="median_income" 
my_label="median_house_value"

my_model = None

shuffled_train_df = train_df.reindex(np.random.permutation(train_df.index)) 

my_model = build_model(learning_rate)
epochs, rmse, history = train_model(my_model, shuffled_train_df, my_feature, 
                                    my_label, epochs, batch_size, 
                                    validation_split)

plot_the_loss_curve(epochs, history["root_mean_squared_error"], 
                    history["val_root_mean_squared_error"])

#output
#Epoch 1/70
#136/136 [==============================] - 0s 2ms/step - loss: 41890.3359 - root_mean_squared_error: 204.6713 - val_loss: 31063.3320 - val_root_mean_squared_error: 176.2479
#Epoch 2/70
#136/136 [==============================] - 0s 1ms/step - loss: 24331.3672 - root_mean_squared_error: 155.9852 - val_loss: 16856.7344 - val_root_mean_squared_error: 129.8335
#...
#Epoch 68/70
#136/136 [==============================] - 0s 1ms/step - loss: 7136.9751 - root_mean_squared_error: 84.4806 - val_loss: 6526.3257 - val_root_mean_squared_error: 80.7857
#Epoch 69/70
#136/136 [==============================] - 0s 1ms/step - loss: 7137.2246 - root_mean_squared_error: 84.4821 - val_loss: 6526.9287 - val_root_mean_squared_error: 80.7894
#Epoch 70/70
#136/136 [==============================] - 0s 1ms/step - loss: 7135.6919 - root_mean_squared_error: 84.4730 - val_loss: 6526.3984 - val_root_mean_squared_error: 80.7861

x_test = test_df[my_feature]
y_test = test_df[my_label]

results = my_model.evaluate(x_test, y_test, batch_size=batch_size)

#output
#30/30 [==============================] - 0s 1ms/step - loss: 7013.0713 - root_mean_squared_error: 83.7441

#Compare the rmse when evaluated three data set,in our experiment,the rmse were similar enough
#In this improved code and with validation set added, we are able to pick the model that does best on the validation set
#and double check that model against the test set which actuaaly acts as the ultimate judge of a model's quality.


#extra 
#details for adding shuffled_train_df rather than adding validation_split solely
learning_rate = 0.08
epochs = 70
batch_size = 100

# Split the original training set into a reduced training set and a
# validation set. 
validation_split=0.2

my_feature="median_income" 
my_label="median_house_value" 

my_model = None

#shuffled_train_df = train_df.reindex(np.random.permutation(train_df.index)) 

my_model = build_model(learning_rate)
epochs, rmse, history = train_model(my_model, train_df, my_feature, 
                                    my_label, epochs, batch_size, 
                                    validation_split)

plot_the_loss_curve(epochs, history["root_mean_squared_error"], 
                    history["val_root_mean_squared_error"])

#output
#Epoch 1/70
#136/136 [==============================] - 0s 2ms/step - loss: 40433.3359 - root_mean_squared_error: 201.0804 - val_loss: 48998.8320 - val_root_mean_squared_error: 221.3568
#Epoch 2/70
#136/136 [==============================] - 0s 1ms/step - loss: 23335.4102 - root_mean_squared_error: 152.7593 - val_loss: 28723.3652 - val_root_mean_squared_error: 169.4797
#...
#Epoch 69/70
#136/136 [==============================] - 0s 1ms/step - loss: 6513.2651 - root_mean_squared_error: 80.7048 - val_loss: 9403.2158 - val_root_mean_squared_error: 96.9702
#Epoch 70/70
#136/136 [==============================] - 0s 1ms/step - loss: 6511.5566 - root_mean_squared_error: 80.6942 - val_loss: 9465.7686 - val_root_mean_squared_error: 97.2922

#As you can see, the loss and val_loss differs largely and you can't go into the next step to test the model as your model is not trained enough.
#This problem is rooted in the data itself. To solve this mysterious issue, you have to write a line of pandas code above which is shuffled_train_df... to let the loss of two sets converge

#Some tips:
#Test sets and validation sets are always repeatedly used. The more you use the same data to make decisions about
#hyperparameter settings or other possible improvements, the less confidence the model has to predict new data. 
#Therefore, it's a good idea to collect more data and keep refreshing your test set and validation set. All in all, it's always a good thing to have fewer exposures to the test set.


#Special thanks to google education for inspiring me on knowledge in this chapter
