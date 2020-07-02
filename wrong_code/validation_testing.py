# Part III Wrong code (Please note that only partial code is included for the emphasis of topics and the specific part needs to be changed and has been changed is marked with a sign and descriptions)

# @validation and testing related examples

#You may be familiar to divide your dataset into two parts which are called traing set and test set. However, you need to modify your data to better fit test set which means your test set data is still exposed.
#You can greatly reduce your chances OVERFITTING by partitioning the data set into three subsets. We are here to compare the differences in root mean square value.

#1
learning_rate = 0.08
epochs = 70
batch_size = 100

#something missing here

my_feature = "median_income"  
my_label="median_house_value" 

my_model = None

my_model = build_model(learning_rate)
weight, bias, epochs, rmse = train_model(my_model, training_df, 
                                         my_feature, my_label,
                                         epochs, batch_size)

plot_the_model(weight, bias, my_feature, my_label)
plot_the_loss_curve(epochs, rmse)

#output
#Epoch 1/70
#170/170 [==============================] - 0s 992us/step - loss: 43307.0625 - root_mean_squared_error: 208.1035
#Epoch 2/70
#170/170 [==============================] - 0s 949us/step - loss: 21817.2656 - root_mean_squared_error: 147.7067
#...
#Epoch 68/70
#170/170 [==============================] - 0s 950us/step - loss: 7014.8965 - root_mean_squared_error: 83.7550
#Epoch 69/70
#170/170 [==============================] - 0s 1ms/step - loss: 7015.0659 - root_mean_squared_error: 83.7560
#Epoch 70/70
#170/170 [==============================] - 0s 1ms/step - loss: 7015.3403 - root_mean_squared_error: 83.7576

x_test = test_df[my_feature]
y_test = test_df[my_label]

results = my_model.evaluate(x_test, y_test, batch_size=batch_size)

#output
#30/30 [==============================] - 0s 939us/step - loss: 7013.3481 - root_mean_squared_error: 83.7457

#According to the root mean square error(rmse) above,the value is extremely similar which implicts that you test data set is overfitted to the training data set.
#As a result, your model's predictive power may be inaccurate for funture unseen data.
#Please refer to the same topic in another folder named correct_code
