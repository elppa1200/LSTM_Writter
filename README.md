# LSTM_Writter
이 프로젝트의 모델(lstm_writter.h5)은 국립국어원의 모두의 말뭉치 데이터셋을 이용해 학습되었습니다.  
이 프로젝트는 교육,학습 목적으로 제작되었습니다.
단어를 입력받고 그에 대한 정보를 입력받아 글짓기를 하는 프로젝트입니다.
 
# 모두의 말뭉치
https://corpus.korean.go.kr/main.do  
에서 데이터셋을 다운받을 수 있으며 본 프로젝트는 이 데이터셋을 사용하여 학습했습니다.    
이 자료를 사용한 작품의 공개 허가를 받았습니다.

# Naver Crawler
https://developers.naver.com/docs/search/blog/  
에서 코드를 확인할 수 있으며 crawler.py의 crowl_link함수는 이를 사용했습니다.

# 작동 방식
1. 네이버 크롤러에서 사용자가 지정한 방법으로 사용자가 지정한 만큼 사용자가 제시한 단어에 대한 정보를 담은 링크를 수집한다.  
2. 수집된 n개의 링크를 newspaper 라이브러리를 사용하여 본문을 크롤링한다.  
3. 이후 gensim 라이브러리로 본문을 요약한 다음 n개의 요약본을 한 본문으로 합한다.  
4. 합한 본문을 다시 요약한 다음 모델로 넘긴다.
5. 모델에서 그 본문 중 랜덤한 지점에서 연속된 10글자를 추출하고 Input으로 삼는다.  
6. 그 10글자에 대해서 작문을 해 Output으로 삼는다.  

# lstm_writter.h5와 jsontext.txt
모델과 자료파일은 100mb에 근사하거나 초과하므로 깃헙에 업로드가 불가함.  
아래 링크의 구글 드라이브에서 따로 다운받아 코드와 같은 위치에 저장할 것.  
https://drive.google.com/drive/folders/1pWu29J9kVjizgeuL3KnR2sm1YDxC9Vct?usp=sharing

# 필요로 하는 라이브러리
Keras - Cuda v.11  
Newspaper3k  
gensim  
Python 3.6  
