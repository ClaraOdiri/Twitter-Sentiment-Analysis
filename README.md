# Twitter-Sentiment-Analysis 
#  How Do People in the UK Feel About The EU After Brexit - A Twitter Sentiment Analysis
An analysis of tweets to understand how people in the UK feel about the European Union after Brexit
<div align="justify">
  
**INTRODUCTION**


UK citizens voted to leave the European Union (EU) on June 23, 2016, which was officially implemented on January 31, 2020. The final outcome of the referendum was a 51.9% in favor to leave the European Union. This is popularly known as Brexit. During this period, UK residents experienced a great deal of uncertainty around the UK’s continued relationship with the EU. Many people have used social media platforms like Twitter to express their emotions about this important event. The UK’s decision to leave the EU has led to significant
political and economic discussions by the media, analysts and scholars, making it crucial to understand how the public feels about this major shift in the country’s relationship with the rest of Europe and its consequences for public policy, investment decisions, and currency markets. The topic of understanding public sentiment towards the European Union (EU) after Brexit is of great importance in the field of AI, specifically in Natural Language Processing (NLP) and sentiment analysis.<br>
Sentiment analysis has been recently considered as an important tool for detecting mental well-being in Twitter contents (Palicki et al., 2021). Sentiment analysis using NLP techniques can help to gain insights into people’s attitudes, opinions and emotions towards the EU, by analyzing text data from social media sources like Twitter. This can provide valuable information for policymakers, businesses, and individuals seeking to understand how the UK’s relationship with the EU may evolve in the future.

The main research question of this project is to analyze public sentiment towards the EU after Brexit in the UK, by collecting and analyzing text data from Twitter. The expected outcome is to provide insights into how people in the UK feel about the EU. This information can be useful for policymakers, businesses, and individuals in understanding the implications of Brexit and how they can adapt to these changes.

**BACKGROUND**


Previous studies have used various approaches, such as machine learning algorithms, lexicon-based methods, and deep learning models, to analyze sentiments of Twitter data concerning Brexit and the European Union. Some research results suggest that leaving the EU will have negative consequences for international trade (Miedviekova, 2020; Dobosh & Saban 2021), and that the costs of remaining in the EU are high (Palicki et al., 2021). Additionally, leaving the EU may have negative impacts on the UK’s economy (ˇSimunjak & Caliandro 2020; Grˇcar et al 2017), and may exacerbate existing problems in the UK’s health service (Fahy et al. 2019). It is also worth
noting that the Brexit vote was influenced by factors such as attitudes towards European integration and national identity (Brakman et al 2018).<br>
Palicki et al. (2021) conducted a study to detect psychological distress in Brexit-related tweets using a transfer learning approach. The paper proposed a sentiment analysis approach based on transfer learning to measure non-clinical psychological distress in Brexit tweets. The transfer learning was able to apply knowledge already learned in the source domain (tweets for users who self-identified with depression or anxiety) to the target domain (tweets about Brexit). The authors noted that UK residents experienced a great deal of uncertainty around the UK’s continued relationship with the EU during the Brexit period and many people used social media platforms to express their emotions about this critical event. They trained a deep learning model on a large dataset of tweets labeled for psychological distress and then fine-tuned the model on a smaller dataset of Brexit-related tweets. The authors found that
the model achieved high accuracy in detecting psychological distress in Brexitrelated tweets. The study suggests that social media can be a useful tool for monitoring mental well-being during political events, and that transfer learning can improve the accuracy of detecting psychological distress in tweets. <br>
Grˇcar et al. (2017) conducted an analysis of Twitter data collected over six weeks before the Brexit referendum in the UK in June 2016. A stance classifier was trained using a Support Vector Machine (SVM) algorithm with linear kernel to address two questions: what is the relation between the Twitter mood and the referendum outcome, and who were the most influential Twitter users in the pro- and contra-Brexit camps? The training set was annotated with four possible stances: ”Leave”, ”Remain”, ”Neutral”, and ”Unrelated”. The classifier was evaluated using two different metrics: accuracy and F1-score. The stance classifier achieved an accuracy of 72.6% and an F1-score of 0.71 on the test set, which indicates that it performed reasonably well at classifying tweets into their correct stance. The study also analyzed the influence of Twitter users by adapting the Hirsch index to their productivity (Twitter posts) and citations (retweets). They compared the influence of pro- and contra-Brexit users and detected retweet communities, comparing their polarization regarding the Brexit stance. The study found that Twitter data alone was not a reliable predictor of the outcome of the
Brexit referendum. The authors argued that previous studies which had shown a direct correlation between volume/sentiment of Twitter data and outcome of elections had many shortcomings and that their methods were no better than
random classifiers. <br>
Saif et al. (2016) conducted a study on Twitter sentiment analysis related to Brexit. The authors proposed an approach called SentiCircles, which builds a dynamic representation of words that captures their contextual semantics in order to tune their pre-assigned sentiment strength and polarity in a given sentiment lexicon. They used three datasets of tweets related to Brexit: the OMD (Diakopoulos & Shamma, 2010) and HCR (Speriosu et al., 2011) datasets to assess the performance of their approach at the tweet level only, since the
datasets provide human annotations for tweets but not for entities. Due to the lack of gold-standard datasets for evaluating entity-level sentiment, they generated an additional dataset, STS-Gold (Saif et al., 2013). They described the use of SentiCircles for lexicon-based sentiment identification at both entity-level and tweet-level using different methods. Their proposed approach outperformed other lexicon labelling methods for both entity-level and tweet-level sentiment detection. For tweet-level sentiment detection, the approach also gave a better overall result than the state-of-the-art lexicon-based approach SentiStrength on two out of three datasets. While SentiStrength uses a fixed set of lexicon words and keeps the strength of each sentiment term unchanged across different data, the SentiCircle representation effectively updated the sentiment strength of many terms dynamically based on their contextual semantics in tweets<br>
Saad et al. (2019) proposed a detailed sentiment analysis of tweets based on ordinal regression using machine learning techniques. They presented an approach to Twitter sentiment analysis by building a balancing and scoring model and afterward, classifying tweets into several ordinal classes using machine learning classifiers such as Multinomial logistic regression, Support vector regression, Decision Trees, and Random Forest. Their results indicate that the Decision Tree gave the highest accuracy at 91.81% and they concluded that the proposed model can detect ordinal regression in Twitter using machine learning methods with a good accuracy result. The performance of the model was measured using accuracy, Mean Absolute Error, and Mean Squared Error. <br>
Various work has been done in the field of Twitter sentiment analysis regarding Brexit, but despite the extensive research, a gap still exists in the literature regarding the sentiment of people in the UK towards the EU after Brexit which this study seeks to fill. This study aims to use a combination of text preprocessing techniques, Long Short-Term Memory (LSTM) neural network and SVM model to classify tweets related to the EU as positive, negative, or neutral. The accuracy, precision, recall, and F1 scores will provide insights into the performance of the models in predicting sentiments. The significance of this project lies in its contribution to a better understanding of public opinion towards the EU in the aftermath of Brexit. This study can be seen as an extension of the existing literature as it aims to expand the scope of research in the field and provide insights into a specific aspect of the broader topic of Brexit and sentiment analysis.

**OBJECTIVES**


The objective of this study is to analyze Twitter data of people in the UK to understand their sentiment of the EU after Brexit. Specifically, we aim to use a Long Short-Term Memory (LSTM) neural network and SVM model to classify tweets as positive, negative, or neutral. The accuracy, precision, recall, and F1 scores will provide insights into the performance of the models in predicting sentiments. The results of this study will provide valuable insights into how people in the UK feel about the European Union after Brexit.

**METHODOLOGY**


**Data Collection:** The data source for this project is Twitter. Tweets
written in English, by people in the UK, were collected from Twitter using the
snscrape library. Hashtags that were used to collect the data include Brexit,
EU, European Union, VoteLeave, and VoteRemain. The tweets were collected
over a period of three months from January 31, 2020, when Brexit was implemented.
About 100,000 tweets were scraped. <br>
**Data Preprocessing:** Raw tweets are full of noise, misspellings and
contain numerous abbreviations and slang words (Saad & Yang 2019). The
preprocessing steps made on the data before modelling are:
- removing duplicated tweets, <br>
- converting words into lowercase (case conversion),<br>
- removing all hyperlinks, hashtags, retweets, and username links that appeared
in the tweets,<br>
- removing puctuations and alphanumeric characters,<br>
- removing words with only three letters,<br>
- removing commonly used words that do not have special meanings, such
as pronouns, prepositions, and conjunctions (stop word removal),<br>
- reducing words to their stem or common root by removing plurals, genders,
and conjugation (stemming),<br>
- segmenting sentences into words or phrases called token by discarding
some characters (tokenization),<br>
- vectorizing the tweets using TF-IDF vectorization to transform each tweet
into a numerical feature vector, and,<br>
- using TextBlob to add equal amounts of positive, negative, and neutral
tweets to get a balanced dataset.<br>  
  
**Data Exploration:** Visualizations like wordcloud, to view the most
frequently occurring words in the dataset, were created to help explore the
data further. This provides insight into the commonly used words or phrases,
where the size of the word represents its frequency. In addition to wordclouds,
a geospatial map was also created using the folium library after extracting the
longitude and latitude from the Location column. 

  
**EXPERIMENT**
  
  
**Data Splitting:**
The dataset was split using an 80/20 split, into training and testing sets.
The training set was further split into training and validation sets using the
80/20 spilt again. <br>
**Model Development:**
A Long Short-Term Memory (LSTM) neural network using Keras was developed
as the base model. The Long Short-Term Memory (LSTM) model is a type
of Recurrent Neural Network (RNN) that is better than traditional recurrent
neural networks in terms of memory. It is capable of handling long-term dependencies
by selectively retaining or forgetting information at each time step.<br>

The basic architecture of an LSTM model includes memory cells, input gates,
output gates, and forget gates. The memory cells are responsible for storing and
updating the information at each time step. The input gates determine which
information to add to the memory cells, while the forget gates decide which
information to discard. The output gates control the information that is passed
on to the next time step. The LSTM model was trained on the preprocessed
tweets with these layers:<br>
- An Embedding layer with input dimension 5000, output dimension 128,
and input length equal to the number of features in the input data X.<br>
- An LSTM layer with 128 units and a dropout rate of 0.2 to prevent overfitting.<br>
- A Dense layer with 3 units and softmax activation.<br>
- The model is compiled with the categorical cross-entropy loss function and
the Adam optimizer.<br>
- The model’s performance was evaluated with the accuracy metric.<br>
To compare the base model, a Support vector Machine (SVM) was trained.
For classification problems, the SVM algorithm finds the hyperplane that maximally
separates the different classes in the feature space. This hyperplane is
found by maximizing the margin, which is the distance between the hyperplane
and the closest points of each class, also known as support vectors. The SVM
algorithm was trained using the svm.SVC class from the scikit-learn library with
the following parameters:
- The kernel parameter was set to linear which means the algorithm will
create a linear decision boundary.<br>
- The C parameter to control the trade-off between maximizing the margin
and minimizing the classification error.<br>
- The gamma parameter is set to auto, which means that gamma will be
set to 1/n features.<br>
  
**Model Training and Testing:**
The base LSTM model was trained for 10 epochs with a batch size of 256
and validation data provided, the accuracy metric was used to evaluate the
performance of the model on the test set. The model achieved a training loss
and accuracy of 0.0970 and 0.9726 respectively, while the validation loss and
accuracy were 0.3195 and 0.9140 respectively, suggesting that the model is performing
very poorly.
The SVM model achieved an accuracy of 0.88, which means that it correctly
classified 88% of the instances in the dataset. Looking at the individual class
metrics, we can see that class 1 (neutral) has the highest precision, recall, and
F1-score, indicating that it is the easiest class to classify. Class 0 (negative)
has the lowest recall, meaning that the model is less likely to correctly identify
instances of this class. Class 2 (positive) has the lowest precision, meaning that
the model is more likely to misclassify instances as this class.
  
**Hyperparameter Tuning:**
To improve the base LSTM model and prevent overfitting, the hyperparameter
were tuned to get the best params using a RandomSearch object.
These best params were used to build another LSTM model with the following
architecture:
-The input data is first passed through the embedding layer, which converts
the input text into a dense vector of fixed size.<br>
- The first LSTM layer has 256 units with a dropout rate of 0.2 and a
recurrent dropout rate of 0.4.<br>
- The layer returns sequences, which are then passed through another LSTM
layer with 128 units and the same dropout and recurrent dropout rates.<br>
- The output from the second LSTM layer is then passed through a dense
layer with a softmax activation function to predict the sentiment labels.<br>
- The model is optimized using the categorical cross-entropy loss function
and the Adam optimizer.<br>

  
 **RESULTS**
  
  
**Evaluation Metrics:** The accuracy of the model was used as the primary
evaluation metric. I also calculated the precision, recall, and F1-score for each
sentiment category.
The tuned LSTM model gave a test accuracy of 0.95 showing a great improvement
of the model. Table 5 shows the classification report of the LSTM
model. Noticeably, tweets categorized as positive have the highest precision at
%. This implies that tweets categorized as positive tend to be more correctly
classified. The recall of neutral tweets is the highest at %. This is caused by the fact that after the preprocessing steps, many of the tweets classified as positive
and negative became neutral sentiments.
Based on the outcomes in table
5, it can be revealed that 23% of the tweets were positive, with 12% of the
tweets expressing negative sentiment and 64% expressing neutral sentiment.

  
**CONCLUSION**
  
  
The study aims to analyze Twitter data of people in the UK to understand their
sentiment concerning the European Union after Brexit. In the context of this
work, I present an approach that aims to extract Twitter sentiment analysis by

building a deep learning model and classifying the sentiments behind the tweets.
using a new dataset and machine learning models. The LSTM neural network
and SVM model were used to classify tweets into positive, negative, or neutral
categories. However, LSTM gave the highest accuracy at 95%, which so far
indicates that the general feeling of people in the United Kingdom concerning
the European Union is neutral. Experimental results conclude that the proposed
model can detect sentiments of tweets using deep learning models. The
performance of the model was measured using the accuracy metric, precision,
recall and F1 scores.<br>
In the future, I intend to improve this approach by investigating different
deep learning techniques such as Word2Vec, BERT, GloVe and Vader. Furthermore,
I intend to attempt the use of bigrams and trigrams to capture context
and improve the accuracy of the models.

  
**REFERENCES**
  
  
Anguita, D., Boni, A. and Ridella, S. (2003) A Digital Architecture for Support
Vector Machines: Theory, Algorithm, and FGPA Implementation. IEEE
Transactions on Neural Networks, 14(5), pp.993–1009.
  
  
Blake, David P. (2018) How Bright are the Prospects for UK Trade and Prosperity
Post-Brexit? Available online: http://dx.doi.org/10.2139/ssrn.3183019
[Accessed 25/03/2023]
  
  
Brakman, S., Garretsen, H., & Kohl, T. (2018) Consequences of Brexit and
Options for a ‘Global Britain’. Papers in Regional Science, 97(1), 55-72.
  
  
Diakopoulos, N.A. & Shamma, D.A. (2010) Characterizing Debate Performance
via Aggregated Twitter Sentiment. In Proceedings of the SIGCHI conference
on human factors in computing systems (pp. 1195-1198).
  
  
Dobosh, O., & Saban, O. (2021). The Archetypal Symbols of Light and
Darkness in the Brexit Saga. Scientific Notes of Ostroh Academy National University:
Philology Series, No (11(79), 120–123.
  
  
Fahy, N., Hervey, T., Greer, S., Jarman, H., Stuckler, D., Galsworthy, M., Mc-
Kee, M. (2019). How will Brexit affect Health Services in the UK? An Updated
Evaluation. The Lancet, Vol 393, Issue 10174, 949-958.
  
  
Graves, A., Wayne, G. & Danihelka, I. (2014). Neural Turing Machines.
arXiv:1410.5401 [cs]. [online] Available online: https://arxiv.org/abs/1410.5401v2
[Accessed 03/04/2023].
  
  
Graves, A. (2012). SSupervised Sequence Labelling with Recurrent Neural Networks
Berlin, Heidelberg Springer.
  
  
Grˇcar, M., Cherepnalkoski, D., Mozetiˇc, I., & Kralj Novak, P. (2017). Stance
and Influence of Twitter Users Regarding the Brexit Referendum. Computational
Social Networks, 4(1). Available online: https://doi.org/10.1186/s40649-
017-0042-6 [Accessed 02/04/2023]
  
  
Hochreiter S. & J. Schmidhuber (1997). Long Short Term Memory. Neural
Computation 9(8); 1735 - 1780.
  
Kasture, N.R. & Poonam Bhilare (2015). An Approach for Sentiment Analysis
on Social Networking Sites. International Conference on Computing Communication
Control and Automation p 390 - 395.
  
  
Kim S. & Hovy E., (2004). Determining the Sentiment of Opinions. In Proceedings
of the 20th International Conference of Computing & Linguistics p.
1367
  
  
Liu, S., Li, F., Li, F., Cheng, X. & Shen, H. (2013). Adaptive Co-training
SVM for Sentiment Classification on Tweets. In Proceedings of the 22nd ACM
International Conference on Information & Knowledge Management p.2079 -
2088
  
  
Miedviekova, N. (2020) UK Prospects Evaluation after Brexit. Journal of European
Economy, Vol 19, No 1 (2020), 65-81.
  
  
O’Connor B., Balasubramanyan R., Routledge B.R., & Smith NA (2010).
From Tweets to Polls: Linking Text Sentiment to Public Opinion Time Series.
In Proceedings of the International AAAI Conference on Weblogs and Social
Media, May 2010, p 122 - 129
  
  
Palicki, S., Fouad, S., Adedoyin-Olowe, M. & Abdallah, Z.S. (2021) Transfer
Learning Approach for Detecting Psychological Distress in Brexit Tweets In
Proceedings of the 36th Annual ACM Symposium on Applied Computing, March
2021 Pages 967–975
  
  
Saad, S.E. & Yang, J. (2019). Twitter Sentiment Analysis Based on Ordinal
Regression. IEEE Access, 7, pp.163677–163685.
  
  
Saif, H., He, Y., Fernandez, M., & Alani, H. (2016). Contextual Semantics
for Sentiment Analysis of Twitter.Information Processing & Management,
52(1), Available online: https://doi.org/10.1016/j.ipm.2015.01.005 [Accessed
02/04/2023]
  
  
Saif, H., Fernandez, M., He, Y. & Alani, H., (2013). Evaluation Datasets for
Twitter Sentiment Analysis: a Survey and a New Dataset, the STS-Gold.1st
International Workshop on Emotion and Sentiment in Social and Expressive
Media: Approaches and Perspectives from AI (ESSEM 2013), Turin, Italy.
Scikit-learn.org. Scikit-learn 0.23.1 Documentation. Available online: https://scikitlearn.
org/stable/about.html.
  
  
ˇSimunjak, M., & Caliandro, A. (2020). Framing #Brexit on Twitter: The
EU 27’s Lesson in Message Discipline?. The British Journal of Politics and
International Relations, 3(22), 439-459.
  
  
Speriosu, M., Sudan, N., Upadhyay, S. & Baldridge, J. (2011) Twitter Polarity
Classification with Label Propagation over Lexical Links and the Follower
Graph. In Proceedings of the First workshop on Unsupervised Learning in NLP
(pp. 53-63).
</div>
