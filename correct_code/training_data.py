# Part III Correct code (Please note that only partial code is included for the emphasis of topics and the specific part needs to be changed and has been changed is marked with a sign and descriptions)

# @Training data examples
#It's important to notice and find features whose raw values correlate with the label perfectly and a concept is used called correlation matrix

#A correlation matrix indicates how each attribute's raw values relate to the other attributes' raw values. Correlation values have the following meanings:
#1.0: perfect positive correlation; that is, when one attribute rises, the other attribute rises.
#-1.0: perfect negative correlation; that is, when one attribute rises, the other attribute falls.
#0.0: no correlation; the two column's are not linearly related.
#In general, the higher the absolute value of a correlation value, the greater its predictive power.

#In this case,it's a linear regression problem and it's easy to use this method

training_df.corr()

#output
#              longitude	latitude	housing_median_age	total_rooms	total_bedrooms	population	households	median_income	median_house_value	rooms_per_person
#longitude	       1.0	      -0.9	       -0.1	            0.0	         0.1	          0.1	       0.1	        -0.0	        -0.0	             -0.1
#latitude	      -0.9	       1.0	        0.0	           -0.0        	-0.1	         -0.1	      -0.1	        -0.1	        -0.1	              0.1
#housing_median_age	-0.1	   0.0	        1.0	           -0.4	        -0.3	         -0.3	      -0.3	        -0.1	         0.1	             -0.1
#total_rooms     0.0	      -0.0	       -0.4	            1.0	         0.9	          0.9	       0.9	         0.2	         0.1	              0.1
#total_bedrooms	 0.1	      -0.1	       -0.3	            0.9	         1.0	          0.9	       1.0	        -0.0	         0.0	              0.0
#population	     0.1	      -0.1	       -0.3	            0.9	         0.9	          1.0	       0.9	        -0.0	        -0.0	             -0.1
#households	     0.1	      -0.1	       -0.3	            0.9	         1.0	          0.9	       1.0	         0.0	         0.1	             -0.0
#median_income	 -0.0	      -0.1	       -0.1	            0.2	        -0.0	         -0.0	       0.0	         1.0	         0.7	              0.2
#median_house_value	-0.0	  -0.1	        0.1	            0.1	         0.0	         -0.0	       0.1	         0.7	         1.0	              0.2
#rooms_per_person -0.1	      0.1	         -0.1	            0.1          0.0	         -0.1	      -0.0	         0.2	         0.2                1.0

#As the result shown above, the 'median_income' correlates 0.7 with the label(median_house_value), so median_income might be a good feature


#There are many tricks to be applied when choosing a good feature
#Some tips:
#Avoid rarely used discrete feature values for example,a feature called "unique_house_id" which means that feature is very specific to every house.Thus, it's a bad feature
#Prefer clear and obvious meanings for example "house_age: 27"  rather than stating "house_age: 851472000". In some cases, noisy data causes unclear values


#Special thanks to google education for inspiring me on knowledge in this chapter
