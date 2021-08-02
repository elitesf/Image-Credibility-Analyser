#Importing required libraries
import os
import io
from bs4 import BeautifulSoup
import requests
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
# print("                   fake image analyser                ")
# print("                          FIA                              ")
# print("-------------------------------------------------------------")

# Path to the api key to use google Vision and Language API
credential_path = "credential.json"

# see os.environ in docs
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

#Full Path to the image in question
image_path = "/home/aayush/Desktop/HACAK/Image Credibility Analyser/ashok.jpg"
given_title = input("Enter the description of the image.\n");
print("Given Title", given_title, end= " ")


res = {
    'image_paths'    : image_path,
    'description'   :  given_title,
    'final'         : 'FINAL',
    'unsurety'      : '99.89',
}

res["matching_label"] = []
res["url"] = []
res["visual_similar"] = []
res["dist"] = []
res["credible_title"] = []

# Loading the google text corpus trained on word2vec
model = gensim.models.KeyedVectors.load_word2vec_format('/home/aayush/Documents/GoogleNews-vectors-negative300.bin.gz', binary=True, limit = 500000)

# Credible list of URLs - can be chnged by user
credible = ['investopedia','timesofindia.indiatimes','indianexpress.','indiatoday.',
'pib.gov.in','ndtv.','aajtak.intoday.in','zeenews.india','economictimes.', 'huffingtonpost.', 'theprint.', 'thelogicalindian.','wsj.s', 'nypost.', 'nytimes.','reuters.', 'economist.', 'pbs.','theatlantic.', 'theguardian.', 'edition.cnn','cnbc.', 'scroll.in', 'financialexpress.', 'npr.', 'usatoday.','hindustantimes','thehindu','indiaspend.','altnews.','boomlive.','smhoaxslayer.',
'indiatoday.','snopes.','politifact.','mediabiasfactcheck.','business-standard',
'thelogicalindian.']



#Function for google's cloud vision API
def detect_web(path):
    list = []
    i = 0
    """Detects web annotations given an image."""
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()
    
    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)

    response = client.web_detection(image=image)
    annotations = response.web_detection
    

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
                                                        # .find returns the idx of first occur of str
            f[i][j] = list_of_page_urls[i].find(credible[j])
            if(list_of_page_urls[i].find(credible[j]) >= 0):
                c_list.append(list_of_page_urls[i])
    
    # if URL not matched with any credible sources, return ans here
    if c_list == []:
        
        res['final']= 'No credible sources have used this image, please perform human verification.'
        res['credible_list']= 'None matched.'
        res['dist']= '99.89'
        
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
            r = requests.get(urls)  #=
            soup = BeautifulSoup(r.content, 'html.parser')
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

def remove_stopwords(sentence, stopwords):
    return [w for w in sentence.lower().split() if w not in stop_words]


#Function to compute the WM distances between titles and associated title and the average distance
def wmdist(title_list):
    
    print("Word Mover's Distance for Titles:")
    print("--------------------------------------------------------------------------------")
    distances = []
    stop_words= stopwords.words('english')
    
    for title in title_list:
        given_title = remove_stopwords(given_title, stopwords)
        title = remove_stopwords(title, stopwords)

        #normalizing vectors 
        model.init_sims(replace= True)
        dist = model.wmdistance(given_title, title)         #determining WM distance
        distances.append(dist)
        
    sum_dist = 0
    for distance in distances:
        sum_dist = sum_dist + distance
        print ('distance = %.3f' % distance)
        res["dist"].append(distance)
        print("--------------------------------------------------------------------------------")

    # avg_dist= sum of distance from each title / no. of titles 
    avg_dist = sum_dist/len(distances)
    res['unsurety']= avg_dist
    print("Average Distance: {}".format(avg_dist))
    print("--------------------------------------------------------------------------------")
    return(avg_dist)


#Function to decide whether human verification is required
def human_ver(avg_dist):
    
    if(avg_dist > 1):
        res['final']='No credible sources have used this image, please perform human verification.'
        print("The title and image are flagged. Please use human verification!")
        print("--------------------------------------------------------------------------------")

    else:
        res['final']='The title associated with this image seems to be right. Human verification is NOT required.'
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

    avg_dist = wmdist(title_list)
    human_ver(avg_dist)
    
    res2= json.dumps(res)
    load_res= json.loads(res2)
    
#---------------------#--------------------#---------------------#--------------------#

if __name__ == "__main__":
    main()



