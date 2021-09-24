#  Load required Libraries 
# Dataframe
import pandas as pd

# Array
import numpy as np
import itertools

# Decompress the file
import gzip

# Visualizations
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import matplotlib.colors as colors
%matplotlib inline
from wordcloud import WordCloud, STOPWORDS

## Warnings
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

# Reading the dataset 
data = pd.read_csv('C:/Users/Moses/Dropbox/Teesside/Assignments/Amzon.csv')
# Checking the size of the dataframe
data.shape

#checking the head of the data to understand  various variables involved
data.head()
data.info()

# Finding null values in Dataset
len(data) - len(data.dropna())
data.isnull().sum()
#Checking the data type of Review data 
for i in range(0,len(data)-1):
    if type(data.iloc[i]['review']) != str:
        data.iloc[i]['review'] = str(data.iloc[i]['review'])
#rating over 3 or more considering as good and rest as bad 
data.info()

# rating score
rating_counts = data.groupby('rating').size()
rating_counts

#concatinate reviews and summary 
data['review_text'] = data[['summary', 'review']].apply(lambda x: " ".join(str(y) for y in x if str(y) != 'nan'), axis = 1)
data = data.drop(['review', 'summary'], axis = 1)
data.head()

# print positive review 
data['review_text'][1991]

# classifying good and bad rating 
good_review = len(data[data['rating'] >= 3])
bad_review = len(data[data['rating'] < 3])

#printing review and thier total numbers
print ('Good ratings : {} reviews for apple earpods'.format(good_review))
print ('Bad ratings : {} reviews for apple earpods'.format(bad_review))

# apply new classification 
data['rating_class'] = data['rating'].apply(lambda x: 'bad' if x < 3 else'good')
data.head()
data.shape
data.isnull().sum()
#desctriptive statistics 
#customers for each rating class
data['rating_class'].value_counts()
total = len(data)
print ("total reviews: ",total)
print ("Average rating ratio: ",round(data.rating.mean(),3))

#EDA
data[['rating']].describe()

# Review rating counts bar chart
plt.figure(figsize=(14,6))
data['rating'].value_counts().sort_index().plot(kind='bar')
plt.title('Distribution of Rating')
plt.xlabel('Rating')
plt.ylabel('Number of Reviews')

# text preprocessing
#import spacy
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize, regexp_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize.toktok import ToktokTokenizer
import gensim
import re
import unicodedata
tokenizer = ToktokTokenizer()
#nlp = spacy.load('en', parse=True, tag=True, entity=True)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from bs4 import BeautifulSoup
#import contractions
#from contractions import CONTRACTION_MAP
from nltk.stem import WordNetLemmatizer

# raw tokens
from nltk.tokenize import word_tokenize
raw_tokens=len([w for t in (data["review_text"].apply(word_tokenize)) for w in t])
print('Number of raw tokens: {}'.format(raw_tokens))

#lemmatization
import re, string, unicodedata
import nltk
#import contractions
from bs4 import BeautifulSoup
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer
#from contractions import CONTRACTION_MAP

def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text

# special_characters removal
def remove_special_characters(text, remove_digits=True):
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    return text


def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words


def remove_punctuation_and_splchars(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_word = remove_special_characters(new_word, True)
            new_words.append(new_word)
    return new_words

def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

stopword_list= stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('not')

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopword_list:
            new_words.append(word)
    return new_words

def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems

def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

def normalize(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation_and_splchars(words)
    words = remove_stopwords(words)
    return words

def lemmatize(words):
    lemmas = lemmatize_verbs(words)
    return lemmas

def normalize_and_lemmaize(input):
    sample = denoise_text(input)
    #sample = expand_contractions(sample)
    sample = remove_special_characters(sample)
    words = nltk.word_tokenize(sample)
    words = normalize(words)
    lemmas = lemmatize(words)
    return ' '.join(lemmas)
print(stopword_list)

data.isnull().sum()
data['cleaned_text'] = data['review_text'].map(lambda text: normalize_and_lemmaize(text))
data
data.info()
data.isnull().sum()
#cleaning the text
from nltk.tokenize import word_tokenize
clean_tokens=len([w for t in (data["cleaned_text"].apply(word_tokenize)) for w in t])
print('Number of cleaned tokens: {}\n'.format(clean_tokens))
print('Percentage of removed tokens: {0:.2f}'.format(1-(clean_tokens/raw_tokens)))

data.to_csv('C:/Users/Moses/Dropbox/Teesside/Assignments/clean_review_earpods.csv', sep=',', encoding='utf-8', index = False)

data1 = pd.read_csv('C:/Users/Moses/Dropbox/Teesside/Assignments/clean_review_earpods.csv') 
data1.info()
data1.head(5)
data1.shape
data1.isnull().sum()
#droping rows with null values 
data1.dropna(inplace=True)
data1.isnull().sum()
data1.info()

# exploratory data analysis

# plot number of positive reviews
reviews = data1["rating"].value_counts()
plt.figure(figsize=(12,8))
reviews[:20].plot(kind='bar')
plt.title("Number of Reviews from 1-5 ")
plt.xlabel('rating')
plt.ylabel('Number of Reviews')

r_5 = data1[(data1['rating_class']=="good")]
r_5


print(r_5['review_text'][770])

from nltk.tokenize import RegexpTokenizer
def RegExpTokenizer(Sent):
    tokenizer = RegexpTokenizer(r'\w+')
    return tokenizer.tokenize(Sent)

ListWords = []
for m in r_5['cleaned_text']:
    n = RegExpTokenizer(str(m))
    ListWords.append(n)
print(ListWords[10])
#all words
from nltk import FreqDist
def Bag_Of_Words(ListWords):
    all_words = []
    for m in ListWords:
        for w in m:
            all_words.append(w.lower())
    all_words1 = FreqDist(all_words)
    return all_words1

import matplotlib as mpl
from wordcloud import WordCloud
all_words4 = Bag_Of_Words(ListWords)
ax = plt.figure(figsize=(15,10))
# Generate a word cloud image
wordcloud = WordCloud(background_color='white',max_font_size=40).generate(' '.join(all_words4.keys()))

# Display the generated image:
# the matplotlib way:
import matplotlib.pyplot as plt
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
print("Combien de Mots !!!",len(all_words4))

plt.figure(figsize = (8,6))
import seaborn as sns
from sklearn.manifold import TSNE
all_words4 = Bag_Of_Words(ListWords)
count = []
Words  = []
for w in all_words4.most_common(10):
    count.append(w[1])
    Words.append(w[0])
sns.set_style("darkgrid")
sns.barplot(Words,count)

b_5 = data1[(data1['rating_class']=="bad")]
b_5

print(b_5['review_text'][284])

from nltk.tokenize import RegexpTokenizer
def RegExpTokenizer(Sent):
    tokenizer = RegexpTokenizer(r'\w+')
    return tokenizer.tokenize(Sent)

ListWords1 = []
for m in b_5['cleaned_text']:
    n = RegExpTokenizer(str(m))
    ListWords1.append(n)
print(ListWords1[1])

from nltk import FreqDist
def Bag_Of_Words(ListWords1):
    all_words1 = []
    for m in ListWords1:
        for w in m:
            all_words1.append(w.lower())
    all_words2 = FreqDist(all_words1)
    return all_words2

import matplotlib as mpl
from wordcloud import WordCloud
all_words5 = Bag_Of_Words(ListWords1)
ax = plt.figure(figsize=(15,10))
# Generate a word cloud image
wordcloud = WordCloud(background_color='white',max_font_size=40).generate(' '.join(all_words5.keys()))

# Display the generated image:
# the matplotlib way:
import matplotlib.pyplot as plt
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
print("Combien de Mots !!!",len(all_words5))

plt.figure(figsize = (8,6))

import seaborn as sns
from sklearn.manifold import TSNE
all_words5 = Bag_Of_Words(ListWords1)
count = []
Words  = []
for w in all_words5.most_common(10):
    count.append(w[1])
    Words.append(w[0])
sns.set_style("darkgrid")
sns.barplot(Words,count)

# rating 
plt.figure(figsize = (10,6))
sns.countplot(data1['rating'])
plt.title('Total Review Numbers for Each Rating', color='r')
plt.xlabel('Rating')
plt.ylabel('Number of Reviews')
plt.show()

# Customer totals for each rating class
data1['rating'].value_counts()

%matplotlib inline
plt.figure(figsize = (10,6))

data1.groupby('rating').rating.count()
data1.groupby('rating').rating.count().plot(kind='pie',autopct='%1.1f%%',startangle=90,explode=(0,0.1,0,0,0),)

word_count=[]
for s1 in data.review_text:
    word_count.append(len(str(s1).split()))

plt.figure(figsize = (8,6))


#import seaborn as sns
#import matplotlib.pyplot as plt
#sns.boxplot(x="rating",y=word_count,data = data1)
#plt.xlabel('Rating')
#plt.ylabel('Review Length')
#plt.show()
#Since there are outliers in the above boxplot we are not able to clearly visualize.So remove the outliers 
#plt.figure(figsize = (8,6))
#sns.boxplot(x="rating",y=word_count,data=data1,showfliers=False)
#plt.xlabel('Rating')
#plt.ylabel('Review Length')
#plt.show()

# Total numbers of ratings in the home and kitchen product reviews
plt.figure(figsize = (8,6))
sns.countplot(data['rating_class'])
plt.title('Total Review Numbers for Each Rating Class', color='r')
plt.xlabel('Rating Class')
plt.ylabel('Number of Reviews')
plt.show()

# Customer totals for each rating class
data1['rating_class'].value_counts()

plt.figure(figsize = (15,8))

#clean text length
plt.figure(figsize = (15,8))

review_length = data1["cleaned_text"].dropna().map(lambda x: len(x))
plt.figure(figsize=(12,8))
review_length.loc[review_length < 2000].hist()
plt.title("overview of Review Length")
plt.xlabel('Review length')
plt.ylabel('Number of Reviews')

plt.figure(figsize = (14,14))
sns.heatmap(data1.corr(method="pearson"), cmap='Blues', annot = True)

sns.pairplot(data1)

data1['review_text'].describe()

def length(text):
    length = len([w for w in nltk.word_tokenize(text)])
    return length

# Apply length function to create review length feature
data1['review_length'] = data1['review_text'].apply(length)
data1.head(3)

data1['review_length_bin'] = pd.cut(data1['review_length'], np.arange(0,700,50))
data1.head()

def sentiments(n):
    return 1 if n >= 3 else 0
data1['sentiments'] = data1['rating'].apply(sentiments)
data1.head()        

# combine feautures 
def combined_reviews(row):
    return row['name'] + ' '+ row['review_text']
data1['all_reviews'] = data.apply(combined_reviews, axis=1)
data1.head()

data1['sentiments'].value_counts()


# Good rating percentages for each length bin with 50's
per_pos_length = data1.groupby(['review_length_bin'])['sentiments'].mean()
data1['review_length_bin'] = data1.review_length_bin.astype(str)
per_pos_length = per_pos_length*50
per_pos_length

# Plot the graph for good rating class percentage and review length bin
plt.figure(figsize = (14,6))
per_pos_length.plot(kind='bar')
plt.title('Review Lenght and Good Rating Class Percentage Graph', color = 'r', size = 14)
plt.xlabel('Review Length Bin')
plt.ylabel('Good Rating Review Percentage(%)')
plt.show()

# Plotting correlation matrix between numeric variables

plt.figure(figsize = (14,14))
sns.heatmap(data1.corr(method="pearson"), cmap='Blues', annot = True)

sns.pairplot(data1)

data2 = data1.drop(data1[(data1['review_length'] > 150) & (data1['rating_class'] == 'good')].index)

data1['review_length'].describe()

# Create a new data frame with clean text and rating class number
data3 = data2[["cleaned_text", "sentiments"]].reset_index()
data3.head(10)

## Modeling
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from catboost import CatBoostClassifier, Pool
from sklearn.naive_bayes import GaussianNB 
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from  sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from gensim.models import Word2Vec
from tqdm import tqdm
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.dummy import DummyClassifier

# Initialize the countervectorizer
countVec = CountVectorizer(ngram_range=(1, 2),binary=True)

data3.info()

# Fit the 'clean_text' to countvectorizer
countVec.fit(data3['cleaned_text'].values.astype('U'))

# Transform the matriz
transformed_matrix = countVec.transform(data3["cleaned_text"].values.astype('U'))


# Convert matrix to array
transformed_matrix.toarray()

# Extracting the feature names
names = countVec.get_feature_names()

# Adding a 'rating' column from previous dataframe's rating value
data4 = pd.DataFrame(transformed_matrix.toarray(), columns=names)
data4['rating'] = data3['sentiments']

data4['rating'].head()

# Create lists for forming a dataframe summary
feature_names = []
avg_ratings = [] 
rating_counts = []
for name in names:
    if name != 'rating':    
        avg_rating = data4[data4[name]== 1]['rating'].mean()
        rating_count = data4[data4[name]== 1]['rating'].count()
        feature_names.append(name)
        avg_ratings.append(avg_rating)
        rating_counts.append(rating_count)  
    else:
        pass

# Create a new dataframe from words, average ratings, and rating counts
data_summary = pd.DataFrame({'feature_name':feature_names, 'avg_rating': avg_ratings, 'rating_count':rating_counts})

# Let's see the new dataframe
data_summary

# Words that are commonly used in the reviews which have good ratings
data_good = data_summary.query("rating_count > 20").sort_values(by='avg_rating', ascending=False)[4:50]
data_good.head(50)
print(data_good)
wordcloud_good = dict(zip(data_good['feature_name'].tolist(), data_good['avg_rating'].tolist()))

# Generate a word cloud image
wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                min_font_size = 10).generate_from_frequencies(wordcloud_good)
 
# plot the WordCloud image                       
plt.figure(figsize = (20, 20), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
plt.show()

# Words that are commonly used in the reviews which have bad ratings
data_bad = data_summary.query("rating_count > 10").sort_values(by= 'avg_rating', ascending=True)[:12]

data_bad.sort_values(by=['avg_rating'],ascending=False,inplace=True)
data_bad

wordcloud_bad = dict(zip(data_bad['feature_name'].tolist(), data_bad['avg_rating'].tolist()))

# Generate a word cloud image

wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                min_font_size = 10).generate_from_frequencies(wordcloud_bad)
 
# plot the WordCloud image                       
plt.figure(figsize = (20, 20), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
plt.show()

data1.info()
data1.isnull().sum()
data1.dropna(inplace=True)
data1.isnull().sum()

data2.to_csv('C:/Users/Moses/Dropbox/Teesside/Assignments//Reduced_Cleaned_Reviews_apple_earpods.csv', sep=',', encoding='utf-8', index = False)

data2.info()
data2.shape

data3 = pd.read_csv('C:/Users/Moses/Dropbox/Teesside/Assignments/Reduced_Cleaned_Reviews_apple_earpods.csv') 
data3.head(5)
data3.info()

# Dropping unnecessary columns
data4 = data3.drop(['rating','name','review_length','review_length_bin', 'sentiments', 'all_reviews'], axis=1)
data4.head()
data4.info()

data4['rating_class'] = data4['rating_class'].apply(lambda x: 0 if x == 'bad' else 1)

# Splitting the Data Set into Train and Test Sets
X = data4['cleaned_text']
y = data4['rating_class']

# Splitting Dataset into train and test set 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=48)


# Print train and test set shape
print ('Train Set data\t\t:{}\nTest Set data\t\t:{}'.format(X_train.shape, X_test.shape))


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title = 'Confusion matrix',
                          cmap = plt.cm.ocean):
    """
    Create a confusion matrix plot for 'good' and 'bad' rating values 
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title, fontsize = 20)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize = 20)
    plt.yticks(tick_marks, classes, fontsize = 20)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment = "center", 
                 color = "white" if cm[i, j] < thresh else "black", fontsize = 40)
    
    plt.tight_layout()
    plt.ylabel('True Label', fontsize = 30)
    plt.xlabel('Predicted Label', fontsize = 30)

    return plt

def disp_confusion_matrix(y_pred, model_name, vector = 'CounterVectorizing'):
    """
    Display confusion matrix for selected model with countVectorizer
    """
    cm = confusion_matrix(y_test, y_pred)
    fig = plt.figure(figsize=(10, 10))
    plot = plot_confusion_matrix(cm, classes=['Bad','Good'], normalize=False, 
                                 title = model_name + " " + 'with' + " " + vector + " "+ '\nConfusion Matrix')
    plt.show()

# Initialize the countervectorizer
countVec = CountVectorizer(ngram_range=(1, 2),binary=True)
count_vect_train = countVec.fit_transform(X_train)
count_vect_train = count_vect_train.toarray()
count_vect_test = countVec.transform(X_test)
count_vect_test = count_vect_test.toarray()

# Print vocabulary length
print('Vocabulary length :', len(countVec.get_feature_names()))

# Assign feature names of vector into a variable
vocab = countVec.get_feature_names()

# Dataframe for train countvectorizer dataset
pd.DataFrame(count_vect_train, columns = vocab).head()

#create a function to apply diffrent algorithms 
def modeling(Model, Xtrain = count_vect_train, Xtest = count_vect_test):
    """
    This function apply countVectorizer with machine learning algorithms. 
    """
    
    # Instantiate the classifier: model
    model = Model
    
    # Fitting classifier to the Training set (all features)
    model.fit(Xtrain, y_train)
    
    global y_pred
    # Predicting the Test set results
    y_pred = model.predict(Xtest)
    
    # Assign f1 score to a variable
    score = f1_score(y_test, y_pred, average = 'weighted')
    
    # Printing evaluation metric (f1-score) 
    print("f1 score: {}".format(score))

clf = DummyClassifier(strategy = 'stratified', random_state =42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
score = f1_score(y_test, y_pred, average = 'weighted')
    
# Printing evaluation metric (f1-score) 
print("f1 score: {}".format(score))

# Compute and print the classification report
print(classification_report(y_test, y_pred))

disp_confusion_matrix(y_pred, "Dummy")

#LG with CV
# Call the modeling function for logistic regression with countvectorizer and print f1 score
modeling(LogisticRegression(multi_class = 'multinomial', solver = 'newton-cg',
                                     class_weight = 'balanced', C = 0.1, n_jobs = -1, random_state = 42))

# Assign y_pred to a variable for further process
y_pred_cv_logreg = y_pred

# Compute and print the classification report
print(classification_report(y_test, y_pred_cv_logreg))

# Print confusion matrix for logistic regression with countvectorizer
disp_confusion_matrix(y_pred_cv_logreg, "Logistic Regression")

#RF with CV
# Call the modeling function for random forest classifier with countvectorizer and print f1 score
modeling(RandomForestClassifier(n_estimators = 200, random_state = 42))

# Assign y_pred to a variable for further process
y_pred_cv_rf = y_pred


# Compute and print the classification report
print(classification_report(y_test, y_pred_cv_rf))


# Print confusion matrix for random forest classifier with countVectorizer
disp_confusion_matrix(y_pred_cv_rf, "Random Forest")


#comparision models with CV
# Function for converting the "classification report" results to a dataframe
def pandas_classification_report(y_true, y_pred):
    metrics_summary = precision_recall_fscore_support(
            y_true=y_true, 
            y_pred=y_pred)

    avg = list(precision_recall_fscore_support(
            y_true=y_true, 
            y_pred=y_pred,
            average='weighted'))

    metrics_sum_index = ['precision', 'recall', 'f1-score', 'support']
    class_report_df = pd.DataFrame(
        list(metrics_summary),
        index=metrics_sum_index)

    support = class_report_df.loc['support']
    total = support.sum() 
    avg[-1] = total

    class_report_df['weighted avg'] = avg

    return class_report_df.T

# Function for adding explanatory columns and organizing all dataframe
def comparison_matrix(y_test, y_pred, label, vector):
    df = pandas_classification_report(y_test, y_pred)
    df['class']=['bad', 'good', 'average']
    df['accuracy']= metrics.accuracy_score(y_test, y_pred)
    df['model'] = label
    df['vectorizer'] = vector
    df = df[['vectorizer', 'model', 'accuracy', 'class', 'precision', 'recall', 'f1-score', 'support']]
    return df

#For loop for using "comparison functions" 

def comparison_table(y_preds, labels):
    
    # empty list for collecting dataframes
    frames_tv = [] 
    
    # list for y_preds
    y_preds_tv = y_preds
    
    # list for labels
    labels_tv = labels  
    
    vector_tv = 'CountVect'
    
    for y_pred, label in zip(y_preds_tv, labels_tv):
        df = comparison_matrix(y_test, y_pred, label, vector_tv)
        frames_tv.append(df)

    # concatenating all dataframes
    global df_tv
    df_tv = pd.concat(frames_tv)
    
    global df_tv2
    df_tv2 = df_tv.set_index(['vectorizer', 'model', 'accuracy', 'class'])

def f1_score_bar_plot(df, category, title):
    df = df[df['class']==category]
    x = list(df['model'])
    y = list(df['f1-score'])
    y_round = list(round(df['f1-score'],2))
    a = (list(df['f1-score'])).index(max(list(df['f1-score'])))
    z = (list(df['f1-score'])).index(min(list(df['f1-score'])))
    y_mean = round(df['f1-score'].mean(),2)
    
    plt.rcParams['figure.figsize']=[15,5]
    b_plot = plt.bar(x=x,height=y)
    b_plot[a].set_color('g')
    b_plot[z].set_color('r')
    
    for i,v in enumerate(y_round):
        plt.text(i-.15,0.018,str(v), color='black', fontsize=15, fontweight='bold')
    
    plt.axhline(y_mean,ls='--',color='k',label=y_mean)
    plt.title(title)
    plt.legend()
    
    return plt.show()

comparison_table(y_preds = [y_pred, y_pred_cv_logreg, y_pred_cv_rf], 
                labels = ['Dummy','LogReg', 'Random Forest'])

df_tv2
# Plotting f1 score with "f1_score_bar_plot" function
f1_score_bar_plot(df=df_tv, category='average', title= "Average f1 Score")

CV = CountVectorizer()
ctmTr = CV.fit_transform(X_train)
X_test_dtm = CV.transform(X_test)

from sklearn.metrics import accuracy_score
model = LogisticRegression()
model.fit(ctmTr, y_train)

y_pred_class = model.predict(X_test_dtm)

accuracy_score(y_test, y_pred_class)

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.rcParams['figure.figsize'] = (6, 6)
sns.heatmap(cm ,annot = True)