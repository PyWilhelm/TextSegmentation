### Experiments

     We evaluate the several alternative models on a subset of Wikipedia corpus. In order to the evaluation simpler and more understandable, only English corpus is used. We use precision, recall and F1-score to evaluate the models. 

1. Prepare Data


    We use the Wikipedia Corpus in English. It provides a huge number of articles. The first phase to transform the corpus to the training labeled dataset. All tags of HTML are removed, so that the article is in raw-text format. After that the article is separated by sentences, and remove all the non-letter characters, such as white spaces, punctuations. The last phase is to set up a label for each letter. Here we use two classes to lable the data. One of them is TRUE, that means there is a segmentation after this letter, and the opposite class is FALSE. For example, the sentence "Nature language processing is cool." will be transformed to a vector of raw data ("Naturelanguageprocessingiscool") and the corresponding vector of labels("FFFFFTFFFFFFFTFFFFFFFFFTFTFFFT"). 

    The second phase is character embeddings. Actually this phase can be treated as the first hidden layer of the neural network. Because the neural network cannot process the sybomlic data directly, the letters must be transformed to distributed vectors firstly. Here two approaches are used, Bitmap and letter2vec. 

    1.1. Bitmap

    Formally, we have a character dictionary C of size |C|. [for English it simple, 0-9a-zA-Z] Each character c in C, is represented as a 0-1 vector Vc = (v0, ..., v|c|, vi=0 if Ci!=c, vi=1 if Ci==c). Therefore, for each sentence S we have a matrix M^{|C|*|S|}. The metrix is the actual input data fed to neural network. 
    This method is very simple to implement and interprete. However, it has some disavantages. The matrix is very sparse, so it wastes quite much memory space. Seconds, all characters are treated as with the equal weight, but as we known, the vowel letters and consonant letters shoule has different importance. Besides, the frequence of letters usually has a huge difference, e.g. "c" vs. "x". 

    1.2. letter2vec

2. Model and superparameter setup
    
    
3. Experience results