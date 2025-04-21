## Deep learning is used for unstructured data (don't use deep learning when problem can be solved using simple rule based system)
almost anything can be solved using deep learning as long as you can convert the input data in numbers.
## Deep learning algorithm's
1. Neural networks.
2. Fully connected neural network.
3. Convolutional neural network.
4. Recurrent neural network.
5. Transformer

## What are neural networks?

  image or text or voice or anykind of unstructured data  -------------> [[1030,94,42992]]  ------------->  ![Neural_Network_layer](https://github.com/user-attachments/assets/c25a894a-dc87-4498-8a06-d86a36f7dab1)  ---------------> Represention output (numbers).-> these represention o/p can be can be understand by humans .


patterns synonymous=> embedding, weights, feature representation. 


## Types of Learning
1. Supervised Learning: example=> input will be data and labels
2. Semi-supervised Learning: example => input will be data and some labels.
3. Unsupervised Learning: example => only data available. 
4. Transfer Learning: example: it will be taking what one neural network had learned then using that to find pattern in another data.


## Basic of TensorFlow
1. Tensor: are multi dimensional arrays,
   ==> import tensorflow as tf  ## importing
   ==> 0-D array
   tensor_zero_d=tf.constant(0)  ## 0-D
   ==> 1-D array
   tensor_one_d=tf.constant([1,2,3])  ## 1-D
   ==> 2-D array
   tensor_two_d=tf.constant([[1,2,3],[2,5,7]])  ## 2-D
   ==> 3-D array
   tensor_three_d=tf.constant([
   [[1,2,3],[2,5,7]],
   [[1,2,3],[2,5,27]],
   [[1,2,3],[2,5,7]]
   ])  ## 3-D
   == 4-D array
   tensor_four_d=tf.constant([
   [
   [[1,2,3],[2,5,7]],
   [[1,2,3],[2,5,27]],
   [[1,2,3],[2,5,7]]
   ],
   [
   [[1,2,3],[2,5,7]],
   [[1,2,3],[2,5,27]],
   [[1,2,3],[2,5,7]]
   ],
   [
   [[1,2,3],[2,5,7]],
   [[1,2,3],[2,5,27]],
   [[1,2,3],[2,5,7]]
   ],
   ])  ## 4-D tensor

2. Check tensorflow's available data type and in built methods.
   ### Practice More and more with tensorflow methods

3. CASTING
4. INITIALIZATION
5. INDEXING
6. BROADCASTING
7. ALGEBRAIC OPERATIONS
8. MATRIX OPERATIONS
9. COMMONLY USED FUNCTIONS IN ML
10. RAGGED TENSORS
11. SPARSE TENSORS
12. STRING TENSORS


### Machine Learning Development Life Cycle
1. Task  (understand the task properly).
2. Data
3. Modeling
4. Error Measurement
5. Training & Optimization
6. Performance Measurement
7. Validating & Testing
8. Corrective Measures


