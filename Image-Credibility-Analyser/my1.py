#Importing required libraries
# import json
import os
import io
import sys
from bs4 import BeautifulSoup
import requests
import array as arr
import six as six
import gensim
import json

from google.cloud import vision
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types

from nltk.corpus import stopwords
from nltk import download
download('stopwords')


# print("-------------------------------------------------------------")
# print("                   image credibility analyser                ")
# print("                            ICA                              ")
# print("-------------------------------------------------------------")

#Path to the api key to use google Vision and Language API
credential_path = "credential.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path
# print("DONE")
# #Title that is associated to the image in question

#Full Path to the image in question
image_path = "/home/shubham/Desktop/HACAK/Image Credibility Analyser/ashok.jpg"
given_title = input("Enter the description of the image.\n");
print("Given Title",given_title,end=" ")
# image_path = "/home/shubham/Desktop/HACAK/Fake-Image-Analyzer--FIA--master/as.jpg"


res = {
    'image_paths'    : image_path,
    'description'   : given_title,
    # 'matching_label': [],
    'final'         : 'FINAL',
    # 'url'           :['url1','url2','url3'],
    # 'visual_similar':['url1','url2','url3'],
    # 'dist'          :['url1','url2','url3'],
    # 'web_entities'  : ['entity1','entity2','entity3'],      
    'unsurety'      : '99.89',
    # 'comment'       :'Fake'
}

res["matching_label"]=[]
res["url"]=[]
res["visual_similar"]=[]
res["dist"]=[]
res["credible_title"]=[]

# Loading the google text corpus trained on word2vec
model = gensim.models.KeyedVectors.load_word2vec_format('/home/shubham/Documents/GoogleNews-vectors-negative300.bin.gz', binary=True, limit = 500000)

# Credible list of URLs - can be chnged by user
credible = ['investopedia','timesofindia.indiatimes','indianexpress.','indiatoday.',
'pib.gov.in','ndtv.','aajtak.intoday.in','zeenews.india','economictimes.', 'huffingtonpost.', 'theprint.', 'thelogicalindian.','wsj.s', 'nypost.', 'nytimes.','reuters.', 'economist.', 'pbs.','theatlantic.', 'theguardian.', 'edition.cnn','cnbc.', 'scroll.in', 'financialexpress.', 'npr.', 'usatoday.','hindustantimes','thehindu','indiaspend.','altnews.','boomlive.','smhoaxslayer.',
'indiatoday.','snopes.','politifact.','mediabiasfactcheck.','business-standard',
'thelogicalindian.']


# credible = ['economictimes.', 'huffingtonpost.', 'theprint.', 'thelogicalindian.', 'thequint.', 'altnews.', 'wsj.', 'nypost.', 'nytimes.', 'bbc.', 'reuters.', 'economist.', 'pbs.', 'aljazeera.', 'thewire.', 'theatlantic.', 'theguardian.', 'edition.cnn',
#             'cnbc.', 'scroll.in', 'financialexpress.', 'npr.', 'usatoday.', 'snopes.', 'politifact.','hindustantimes','thehindu','timesofindia','cnn.']



#Function for google's clous vision API
def detect_web(path):
    list = []
    i = 0
    """Detects web annotations given an image."""
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()
    # print("IMAGE: ",path)
    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)

    response = client.web_detection(image=image)
    annotations = response.web_detection
    # print("PRINTING ANNOATATIONS:===========")
    # for ass in annotations:
    #     print(ass)
    # print("\nJSON---=====================")
    # parsed = json.loads(annotations)
    # print(json.dumps(parsed, indent=4, sort_keys=True))
    # print("---=====================\n")
    if annotations.best_guess_labels:
        for label in annotations.best_guess_labels:
            res["matching_label"].append(label.label)
            print('\nBest guess for the image: {}'.format(label.label))
            print("--------------------------------------------------------------------------------")


    if annotations.pages_with_matching_images:
        print('\n{} Pages with matching images found:'.format(
            len(annotations.pages_with_matching_images)))

        for page in annotations.pages_with_matching_images:
            res["url"].append(page.url)
            # res.url.append(page.url)
            print('\n\tPage url   : {}'.format(page.url))
            list.append(page.url)

    if annotations.web_entities:
        print('\n{} Web entities found in the image: '.format(
            len(annotations.web_entities)))

        for entity in annotations.web_entities:
            print('\n\tScore      : {}'.format(entity.score))
            print(u'\tDescription: {}'.format(entity.description))

    if annotations.visually_similar_images:
        print('\n{} visually similar images found:\n'.format(
            len(annotations.visually_similar_images)))
        for image in annotations.visually_similar_images:
            res["visual_similar"].append(image.url)
            # res.visual_similar.append(image.url)
            print('\tImage url    : {}'.format(image.url))
    print("--------------------------------------------------------------------------------")
    # print("PRINTING COMPLETE LIST=============")
    # for i in list: 
    #     print(i) 
    return(list)

#---------------------#--------------------#---------------------#--------------------#

#Function to check which URLs belong to credible news sources
def credible_list(list_of_page_urls):

    c_list = []

    c_length = len(credible)

    url_length = len(list_of_page_urls)

    f = [[0 for j in range(c_length)] for i in range(url_length)]
    for i in range(url_length):
        for j in range(c_length):
            f[i][j] = list_of_page_urls[i].find(credible[j])
            if((list_of_page_urls[i].find(credible[j])) > 0):
                c_list.append(list_of_page_urls[i])
    if c_list == []:
        # res.final='No 
        # print(No credible sources have used this image, please perform human verification.'
        res['final']='No credible sources have used this image, please perform human verification.'
        res['credible_list']='None matched.'
        res['dist']='99.89'
        # print(res)
        print("No credible sources have used this image, please perform human verification.")
        print("--------------------------------------------------------------------------------")
        exit(1)
    return(c_list)
#---------------------#--------------------#---------------------#--------------------#

#Function to scrape titles off the given URLs
def titles(credible_from_url_list):

    title_list = []

    for urls in credible_from_url_list:
        if urls != []:
            r = requests.get(urls)
            html = r.content
            soup = BeautifulSoup(html, 'html.parser')
            title_list.append(soup.title.string)

    return(title_list)

#---------------------#--------------------#---------------------#--------------------#

#Function to print the scraped titles
def print_article_title(title_list):
    print("Credible article titles which use the same image: ")
    print("--------------------------------------------------------------------------------")
    for title in title_list:
        res["credible_title"].append(title)
        # res.credible_title.append(title)
        print(title)
        print("--------------------------------------------------------------------------------")
#---------------------#--------------------#---------------------#--------------------#

#Function to call google's language API for entity analysis
def entity_analysis(title_list):
    for title in title_list:
        entity_sentiment_text(title)
        # print()

#---------------------#--------------------#---------------------#--------------------#
#Function to compute the WM distances between titles and associated title and the average distance

def wmdist(title_list):
    print("Word Mover's Distance for Titles:")
    print("--------------------------------------------------------------------------------")
    distances = []
    # stop_words=stopwords.words('english')
    # given_titles = [w for w in given_title if w not in stop_words]
    for title in title_list:
        # title = [w for w in title if w not in stop_words]
        model.init_sims(replace=True)
        dist = model.wmdistance(given_title, title) #determining WM distance
        distances.append(dist)
        # distance = model.WmdSimilarity(given_title, title)

    sum_dist = 0
    for distance in distances:
        sum_dist = sum_dist + distance
        print ('distance = %.3f' % distance)
        res["dist"].append(distance)
        # res.dist.append(distance)
        print("--------------------------------------------------------------------------------")

    avg_dist = sum_dist/len(distances)
    res['unsurety']=avg_dist
    # res.unsurety=avg_dist
    print("Average Distance: {}".format(avg_dist))
    print("--------------------------------------------------------------------------------")
    return(avg_dist)
# def wmdist(title_list):
#     print("Word Mover's Distance for Titles:")
#     print("--------------------------------------------------------------------------------")
#     distances = []
#     title_list=['fake', 'spam', 'bogus','false','counterfeit','fraudulent']
#     for title in title_list:
#         dist = model.wmdistance(given_title, title) #determining WM distance
#         distances.append(dist)
#         #distance = model.WmdSimilarity(given_title, title)

#     sum_dist = 0
#     for distance in distances:
#         sum_dist = sum_dist + distance
#         print ('distance = %.3f' % distance)
#         print("--------------------------------------------------------------------------------")

#     avg_dist = sum_dist/len(distances)
#     print("Average Distance: {}".format(avg_dist))
#     print("--------------------------------------------------------------------------------")
#     print("Fake ...............")
#     print(avg_dist)
#     return(avg_dist)
#---------------------#--------------------#---------------------#--------------------#


#Function to decide whether human verification is required
def human_ver(avg_dist):
    # print(100.0*avg_dist);
    if(avg_dist > 1):
        res['final']='No credible sources have used this image, please perform human verification.'
        # res.final='The title and image are flagged. Please use human verification!'
        print("The title and image are flagged. Please use human verification!")
        print("--------------------------------------------------------------------------------")

    else:
        res['final']='The title associated with this image seems to be right. Human verification is NOT required.'
        # print(100.0*avg_dist);
        print("The title associated with this image seems to be right. Human verification is NOT required.")
        print("--------------------------------------------------------------------------------")

#---------------------#--------------------#---------------------#--------------------#
#Main function to call the rest of the above functions
def main():
    list_of_page_urls = []
    credible_from_url_list = []
    title_list = []
    list_of_page_urls = detect_web(image_path)
    credible_from_url_list = credible_list(list_of_page_urls)
    title_list = titles(credible_from_url_list)
    print_article_title(title_list)
    # print("LEN: ",len(title_list))
    # print("WM")
    avg_dist = wmdist(title_list)
    human_ver(avg_dist)
    # print("WM END")
    res2=json.dumps(res)
    load_res=json.loads(res2)
    # print(res2)
#---------------------#--------------------#---------------------#--------------------#

if __name__ == "__main__":
    main()
