# Generating_counterfactual_examples

There are the pretrained classifier models in 'models/saved'.
If you want to retrain the modesl, the retraining code is in 'models/'.

I used the baseline model to generate counterfactual examples as GrowingSphere algorith.
You can see more detain in the algorithm on a paper named "Inverse Classification for Comparison-based Interpretability in Machine Learning"
I also copied GrowingSphere algorithm code and used for my task.

The dataset is in 'data/'.
There are two dataset named "HELOC", "UCI_creditcard"
The two dataset is used to train a classifier which is for loan accept or not.

For evaluating same condition, I splited the train dataset and test dataset.
I also uploaded the evaluation metric code to this repository.
The evaluation metrics that I'm suggesting are 'L1', 'L2', 'Coherence'.

To save our time, I saved the generated counterfactual examples which is made by GrowingSphere to 'data/' each dataset respectly.
So, just running the 'HELOC_L1L2_coherence.py', 'UCI_L1L2_coherence.py' file is available.

Finaly, The torch version is 1.7.0.
Else packages's version wouldn't have problem to run this code.

If you have questions, please mail me to 'hd_kim@korea.ac.kr'
Thank you!
