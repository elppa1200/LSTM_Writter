import os, sys, json, re
import urllib.request
from newspaper import Article
from gensim.summarization.summarizer import summarize


def crowl_link(user_word, serching_time, serching_type):
    crowler_id = "Vy_eJLK5sQtHqp1OgRTy"
    crowler_pw = "NGiOZCvq1c"
    searching_word = urllib.parse.quote(user_word)
    url = "https://openapi.naver.com/v1/search/news?query=" + searching_word + "&display=" + serching_time + "&sort=" + serching_type
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id", crowler_id)
    request.add_header("X-Naver-Client-Secret", crowler_pw)
    response = urllib.request.urlopen(request)
    rescode = response.getcode()
    if(rescode==200):
        response_body = response.read()
        url_info = json.loads(response_body.decode('utf-8'))
        url_result = url_info['items']
        url_link = []
        url_desc = []
        for i in range (len(url_result)):
            url_temp = url_result[i]
            url_link.append(url_temp['link'])
            url_desc.append(url_temp['description'])
        return(url_link)
    else:
        print("Crowl Error\nError Code:" + rescode)



def crowl_text(link_list):
    i = 0
    url_text = []
    for i in range(len(link_list)):
        url = link_list[i]
        
        news = Article(url, language='ko')
        news.download()
        news.parse()
        text = summarize(news.text)
        url_text.append(text)
    url_sum = ' '.join(url_text)
    url_sum = re.sub(r' +', ' ', url_sum)
    url_sum = re.sub(r'ⓒ', ' ', url_sum)
    url_sum = re.sub(r'▲', ' ', url_sum)
    url_sum = summarize(url_sum)
    url_sum = list(url_sum)
    return (url_sum)
