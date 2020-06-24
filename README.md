# Taxonomy-of-Real-Faults-in-Deep-Learning-Sysatems

## **Preface** ##

*The goal of this summary is to build a taxonomy of real faults in deep learning(DL) systems which could be helpful and useful to aid developers in preventing, detecting, debugging and fixing of program defects in DL related jobs. This taxonomy includes five categories in deep learning specific failures  coming from GitHub issues and Stack Overflow quessions and three categories raised from industrial side serving as a practicle supplementary. Additinally, I'd like to greatly thank two significant research papers which had a big impact on this summary.*

1. An Empirical Study on Program Failures of Deep Learning Jobs written by Ru Z., Wencong X., Hongyu Z., Yu L., Haoxiang L., Mao Y.
2. Taxonomy of Real Faults in Deep Learning Systems written by Nargiz H., Gunel J., Gabriele B., Vincenzo R., Andrea S., and Paolo T.  

*This summary includes three parts:* 
* A description of DL faults which is introduced in this README file
* Wrong code associated with mentioned DL faults 
* Corrected code derived from false version(Please note that the last two parts are written separately in an independent file for your convenience to check)

## **Part I: Possible Faults Occured in DL Systems** ##

## __\*DL Specific\*__ ##



Dimension | Category | Subcategory
----------|----------|------------
Model     | Layers   |Activation Function / Layer Properties / Missing/Redundant/Wrong Layer
_ |  Model Types&Properties
        

> __Model__



Failure Descriptions:

> @Layers

> Activation Function:

a)Binary Step

b)Linear

c)Sigmoid

d)Tanh

f)ReLU

g)Leaky ReLU

h)Parameterised ReLU

i)Exponential Linear Unit

j)Swish

k)Softmax

“A neural network without an activation function is essentially just a linear regression model.” Thus, basically the importance of activation functions are to have non-linearity in the network with non linear transformation to the inputs of the neuron.

* Wrong type of activation function

*Choosing the right activation function:
Since we have so many activation functions, we need some logic to know which activation function should be used in which situation. Although there’s no rule of thumb for good or bad choice, we might be able to make a better choice for easy and quicker convergence of the network.*
1. Sigmoid functions and their combinations generally work better in the case of classifiers
2. Sigmoids and tanh functions are sometimes avoided due to the vanishing gradient problem
3. ReLU function is a general activation function and is used in most cases these days
4. If we encounter a case of dead neurons in our networks the leaky ReLU function is the best choice
5. Always keep in mind that ReLU function should only be used in the hidden layers
6. As a rule of thumb, you can begin with using ReLU function and then move over to other activation functions in case ReLU doesn’t provide with optimal results 
* Missing softmax activation function
* Missing ReLU activation function
> Layer Properties
* Wrong input sample size for linear layer
* Wrong defined input shape
* Wrong defined output shape
* Wrong defined input and output shape
* Wrong filer size for a convolutional layer
* Bias needed in a layer
* Suboptimal number of neurons in a layer
* Wrong amount and type of pooling in convolutional layer
* Layers’ dimensions mismatch
> Missing/Redundant/Wrong Layer
* Missing dropout layer
* Missing normalisation layer
* Missing softmax layer
* Redundant softmax layer
* Wrong layer type 
* Missing flatten layer
* Missing dense layer
* Missing average pooling layer
* Wrong type of pooling layer

> @Model Types& Properties
* Wrong model initialisation
* Wrong weights initialisation
* Wrong selection of model eg. A recurrent network was used instead of a convolutional network which requires the latter
* Wrong network architecture
* Suboptimal network structure eg. too many layers are used, causing suboptimal structure which in turn leads to poor performance of the model
* Multiple initialisations of CNN


Dimension | Category | Subcategory
----------|----------|------------
Tensors & Inputs | Wrong Input | Wrong Input Format/Wrong Shape of Input Data/ Wrong Type of Input Data
_ | Wrong Tensor Shape



> __Tensor&Inputs__

Failture Descriptions:

> @Wrong Input
> Wrong Input Format

eg.  A faulty behaviour results from data with incompatible format,type or shape being used as an input to a layer or a method.
* Wrong input format
* Wrong input format for RNN
* Wrong format of passed weights
* Incompatible tensor type

> Wrong shape of Input Data
* Wrong shape of input data for a method
* Wrong shape of input data for a layer
* Wrong shape of input data

> Wrong Type of Input Data
* Wrong type of input data for a method
* Wrong type of input data for a layer
* Wrong type of input data


> @Wrong Tensor Shape
* Wrong tensor shape (missing squeeze)
* Wrong tensor shape ( wrong indexing )
* Wrong tensor shape ( wrong output padding )
* Wrong tensor shape ( other)
* Tensor shape mismatch

Dimension | Category | Subcategory
----------|----------|------------
Training| Processing of Training Data | Missing/wrong preprocessing
-| Training Data|
-| Training Process|
-| Optimiser|
-|Loss Function|
-|Validation/Testing|
-|Hyperparameters|
> __Training__

Failure Descriptions:

> @Preprocessing of Training Data

> Missing preprocessing step
* Subsampling
* Normalisation
* Input scaling
* Resize of the images
* Oversampling
* Encoding of categorical data
* Padding
* Data shuffling
* interpolation...
> Wrong preprocessing step
* Pixel encoding
* Padding
* Text segmentation
* Normalisation
* Positional encoding
* Character encoding...
> @Training Data

eg. not enough training set/unbalanced training data (one or more classes in a data set are underrepresented 
* Wrong labels for training data
* Wrong selection of features
* Unbalanced training data
* Not enough training data
* Low quality training data
* Overlapping output classes in training data
* Too many output categories
* Small range of values for a feature
* Discarding important features
>@Training Process
* Wrong management of memory resources
* Reference for non-existing checkpoint
* Model too big to fitinto available memory
* Missing data augmentation
* Redundant data augmentation
>@Optimiser

eg. selection of an unsuited optimisation function for model training

* Wrong optimisation function Eg. Adam optimiser instead of stochastic gradient descent is selected 

* Epsilon for adam optimiser
>@ Loss Function
* Wrong loss function calculation
* Missing masking of invalid values to zero
* Wrong selection of loss function
* Missing loss function
>@ Validation/Testing

eg. bad choice of performance metrics or faulty split of data into training and testing datasets
* Missing validation set
* Wrong performance metric
* Incorrect train/test data split
>@ Hyperparameters
* Suboptimal hyperparameters tuning
* Suboptimal learning rate
* Data batching required
* Suboptimal number of epochs
* Suboptimal batch size
* Wrongly implemented data batching 
* Missing regularisation (loss and weight)

Dimension | Category | Subcategory
----------|----------|------------
GPU Usage
> __GPU Usage__

Failure Descriptions:

* Missing destination GPU device
* Incorrect state sharing
* Wrong reference to GPU device
* Missing transfer of data to GPU
* Wrong tensor transfer to GPU
* GPU tensor is used instead of CPU tensor
* Wrong data parallelism on GPUs
* Calling unsupported operations on CUDA tensors
* Conversion to CUDA tensor inside the training/test loop
* Wrongly implemented data transfer function

Dimension | Category | Subcategory
----------|----------|------------
API

> __API__

Failure Descriptions:

eg. using API in a way that does not conform to the logic set out by developers of the framework

* Deprecated API
* Wrong usage of image decoding API
* Wrong usage of placeholder restoration API
* Missing argument scoping
* Wrong position of data shuffle operation
* Missing global variables initialisation
* Wrong API usage
* Missing API call
* Wrong reference to operational graph

## __\*Non-DL specific\*__ ##
Dimension | Category | Subcategory
----------|----------|------------
Execution Environment| Path Not found
-| Library Not Found
-| Permission Denied

Failure Descriptions:

>@Path Not found 

eg.File or directory cannot be found
>@Library Not Found 

eg. Python modules or dependent DLLs cannot be found on the search path
>@Permission Denied 

eg.Insufficient permission to perform actions 


Dimension | Category | Subcategory
----------|----------|------------
General Code Error| Illegal Argument
-| Type Mismatch
-| Key Not Found
-| Null Reference
-| Attribute Not Found
-| Syntax Error
-| Illegal Index
-| Undefined Variable
-| Not Implemented
-| Division by Zero

>@Illegal Argument

eg. Argument does not satisfy program or function requirement

>@Type Mismatch

eg. Applying an operation or function to an object of inappropriate type

>@Key Not Found

eg. Accessing collection items with a non-existing key

>@Null Reference 

eg.Deference on null value objects

>@Attribute Not Found

eg.Referencing a non-existentPython class field

>@Syntax Error

eg.Violation of the grammatical rules

>@Illegal Index

eg.Accessing array elements with an out-of-range or non-integer index

>@Undefined Variable

eg. Referencing a variable before its definition

>@Not Implemented

eg. Functionality is not implemented yet

>@Division by Zero

eg.Dividing a decimal value by zero


Dimension | Category | Subcategory
----------|----------|------------
Data| Corrupt Data
-| Unexpected Encoding

>@Corrupt Data

eg.Exceptional schema or contents in data

>@ Unexpected Encoding

eg. Data cannot be correctly encoded or decoded
