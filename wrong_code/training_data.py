# Part III Correct code (Please note that only partial code is included for the emphasis of topics and the specific part needs to be changed and has been changed is marked with a sign and descriptions)

# @Training data examples
#A short preview of the data set is shown below
#           longitude	latitude	housing_median_age	total_rooms	total_bedrooms	population	households	median_income	median_house_value
#0	        -114.3	      34.2	           15.0.     	5612.0	          1283.0	     1015.0	   472.0	       1.5	          66.9
#1	        -114.5.     	34.4	           19.0	      7650.0	          1901.0	     1129.0	   463.0	       1.8	          80.1
#2	        -114.6.     	33.7	           17.0	      720.0	             174.0	      333.0	   117.0	       1.7	          85.7
#3	        -114.6	      33.6	           14.0	      1501.0	           337.0	      515.0	   226.0	       3.2	          73.4
#4	        -114.6	      33.6	           20.0	      1454.0	           326.0	      624.0	   262.0	       1.9	          65.5

#1
learning_rate = 0.01
epochs = 30
batch_size = 30


my_feature = "total_rooms"  
my_label="median_house_value" 
my_model = None


my_model = build_model(learning_rate)
weight, bias, epochs, rmse = train_model(my_model, training_df, 
                                         my_feature, my_label,
                                         epochs, batch_size)

print("\nThe learned weight for your model is %.4f" % weight)
print("The learned bias for your model is %.4f\n" % bias )

plot_the_model(weight, bias, my_feature, my_label)
plot_the_loss_curve(epochs, rmse)

#output
#Epoch 1/30
#567/567 [==============================] - 1s 1ms/step - loss: 33170.7500 - root_mean_squared_error: 182.1284
#Epoch 2/30
#567/567 [==============================] - 1s 1ms/step - loss: 27754.0469 - root_mean_squared_error: 166.5955
#...
#Epoch 29/30
#567/567 [==============================] - 1s 962us/step - loss: 15484.0674 - root_mean_squared_error: 124.4350
#Epoch 30/30
#567/567 [==============================] - 1s 947us/step - loss: 15296.4248 - root_mean_squared_error: 123.6787
