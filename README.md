# Game Rating Prediction
This code predict game ratings using its title. This is the code for the challenge in 'How to Do Sentiment Analysis' #3 - Intro to Deep Learning by Siraj Raval on Youtube


##Overview

This code is for the challenge at as part of [this](https://youtu.be/si8zZHkufRY) video on Youtube by Siraj Raval. It uses [TFLearn](http://tflearn.org/) to train a Sentiment Analyzer on a set of IGN [Games titles](https://www.kaggle.com/egrinstein/20-years-of-games). Once trained, given some input game title, it will be able to classify it as either positive or negative. The neural network that is built for this is a [recurrent network](https://en.wikipedia.org/wiki/Recurrent_neural_network). It uses a technique called [LSTM](http://colah.github.io/posts/2015-08-Understanding-LSTMs/).

##Dependencies

* tflearn

Use [this](http://tflearn.org/installation/) guide to install TFLearn. Or just use the Amazon Web Services prebuilt [AMI](https://aws.amazon.com/marketplace/pp/B01EYKBEQ0/ref=_ptnr_wp_blog_post) to run this in the cloud


##Usage

Run ``challenge3.py`` in terminal and it should start training


##Credits

Credits for this code goes to Siraj. Thanks for everything.
