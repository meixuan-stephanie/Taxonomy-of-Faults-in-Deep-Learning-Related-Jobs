# Part III Correct code (Please note that only partial code is included for the emphasis of topics and the specific part needs to be changed and has been changed is marked with a sign and descriptions)

# @Training data examples

#A short preview of the data set is shown below and basically we want to predict a house value by giving some chosen feature
#           longitude	latitude	housing_median_age	total_rooms	total_bedrooms	population	households	median_income	median_house_value
#0	        -114.3	      34.2	           15.0.     	5612.0	          1283.0	     1015.0	   472.0	       1.5	          66.9
#1	        -114.5.     	34.4	           19.0	      7650.0	          1901.0	     1129.0	   463.0	       1.8	          80.1
#2	        -114.6.     	33.7	           17.0	      720.0	             174.0	      333.0	   117.0	       1.7	          85.7
#3	        -114.6	      33.6	           14.0	      1501.0	           337.0	      515.0	   226.0	       3.2	          73.4
#4	        -114.6	      33.6	           20.0	      1454.0	           326.0	      624.0	   262.0	       1.9	          65.5

#A wrong selection of features can have a huge effect on your model's accuracy
#1
learning_rate = 0.01
epochs = 30
batch_size = 30


my_feature = "total_rooms"  # Specify the feature and the label.In this case, we chose "total rooms"
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

#You may find the loss from this model is extremely large. And the graph generated from the output has many randomly distributed points.
#As a result, you may get differnt large loss balues each time you train this model. A more detailed result will be showed by making predictions with this model

def predict_house_values(n, feature, label):
  """Predict house values based on a feature."""

  batch = training_df[feature][10000:10000 + n]
  predicted_values = my_model.predict_on_batch(x=batch)

  print("feature   label          predicted")
  print("  value   value          value")
  print("          in thousand$   in thousand$")
  print("--------------------------------------")
  for i in range(n):
    print ("%5.0f %6.0f %15.0f" % (training_df[feature][10000 + i],
                                   training_df[label][10000 + i],
                                   predicted_values[i][0] ))
  predict_house_values(10, my_feature, my_label)
  
#output
#  feature   label          predicted
#  value   value          value
#          in thousand$   in thousand$
#--------------------------------------
#    2     53             188
#    2     92             200
#    2     69             195
#    2     62             211
#    1     80             186
#    2    295             225
#    2    500             210
#    2    342             223
#    4    118             287
#    2    128             214

#According to the output above, most of the predicted values differ from the label value
#By the way, you may say only ten examples are given are not precise to state whether a model's ability is good or bad
#You may feel free to vary the example number, however, regardless of how many values you will set there in this case
#the prediction will not change so much as the feature selected doesn't correlate with this label perfectly.

#You are also free to try different features within the data set after the first feature you chose is not suitable for your model
#For example, "population" can be picked and it can give you a slightly better result compared to the previous one

#Unfortunately, the model's predictive ability is still not as good as you expect after trying using a new feature
#Brainstorm may be important when you define a synthetic feature, you can just use "rooms_per_person" by combining two features applied above.
#Remember to tune your hyperparameters respectively

training_df["rooms_per_person"] = training_df["total_rooms"] / training_df["population"]
my_feature = "rooms_per_person" #a new feature applied

# Tune the hyperparameters.
learning_rate = 0.06
epochs = 24
batch_size = 30

my_model = build_model(learning_rate)
weight, bias, epochs, mae = train_model(my_model, training_df,
                                        my_feature, my_label,
                                        epochs, batch_size)

plot_the_loss_curve(epochs, mae)
predict_house_values(15, my_feature, my_label)

#output
#feature   label          predicted
#  value   value          value
#          in thousand$   in thousand$
#--------------------------------------
#    2     53             188
#    2     92             200
#    2     69             195
#    2     62             211
#    1     80             186
#    2    295             225
#    2    500             210
#    2    342             223
#    4    118             287
#    2    128             214
#    2    187             224
#    3     80             234
#    2    112             224
#    2     95             219
#    2     69             210
