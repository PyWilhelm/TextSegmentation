## Deep Learning in Text Segmentation

### Introduction

> *1 page (+ abstract)*

### Text Segmentation Review
1. The meaning of TS for some languages, e.g. Chinese, Japenese
2. Traditional Approaches: 
    - N-Gram model
    - Character-based tagging
    - Word perceptron

> *0.75 page - 1 page*

### Deep Learning for Text Segmentation
Word Segmentation Problem can be treated as a labeling task for a sequence of characters. The critical part of handeling the tagging task is the choice of features in the traditional approaches. But it depends on the knowledgement and intuition of experts. And the deep learning approaches overcome this problem and can discover multiple levels of features extraction, with higher levels representing more abstract aspects of the inputs. 

1. Common model of Deep Learning for Text Segmentation

Collobert introduce a common neural network model for many NLP problems. [fig1]
The model contains the following core components. 

    - Character embedding layer

    the characters must be transformed to a vector of features. 
    we have two methods for character embedding. Bitmap and Letter2Vec.

    - Char-window

    normally, we use a local window for dealing with the sequence of characters. assume size of window is M, for each characters ci in the sequence the embeddings of all the context of ci-m1 ... c1+m1 is concatenated into ai R d w

    - Classical or deep neural network layers

Then the ai is fed into the 2nd layer, the neural network layer which performs a linear transformation followed by a non-linear activition function g, e.g. sigmoid and tanh. [formula]

    - tag inference layer
  As results, each character will be labeled as one of [S, N] to indicate segmentation. S means segmenting after the character and N means no-segmenting. 
2. Gated Recursive Neural Network

3. Long Short Term Memory Neural Network

> *1.5 page*

### Experience
1. Data prepare, english only
2. Model and superparameter setup
3. Experience results

> *0.75 page*

### Conclusion
1. Analyse the best Model
2. Limitation: only english, no analyse for unknown words

> *0.5 page*

### Reference

> *0.5 page*
