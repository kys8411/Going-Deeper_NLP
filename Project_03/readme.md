AIFFEL Campus Online 4th Code Peer Review Templete
코더 : 김용석
리뷰어 : 남희정


PRT(PeerReviewTemplate)
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.

1.코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
  아직 돌리는 중이라 정삭 동작 및 문제 해결 여부는 확인하지 못하였습니다만 아래 코드로 문제를 해결하고자 하신 의도는 파악하였습니다.   
  ''' python
  from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from konlpy.tag import Okt


def read_token(file_name):
    okt = Okt()
    result = []
    with open('.../synopsis/'+file_name, 'r') as fread: 
        print(file_name, '파일을 읽고 있습니다.')
        while True:
            line = fread.readline() 
            if not line: break 
            tokenlist = okt.pos(line, stem=True, norm=True) 
            for word in tokenlist:
                if word[1] in ["Noun"]:#, "Adjective", "Verb"]:
                    result.append((word[0])) 
    return ' '.join(result)
    '''
2.주석을 보고 작성자의 코드가 이해되었나요?
  주석은 보기 쉽게 잘 달아졌습니다. 
3.코드가 에러를 유발할 가능성이 있나요?
  
4.코드 작성자가 코드를 제대로 이해하고 작성했나요?
  노드에 대한 기본적인 이해는 하신걸로 판단됩니다.
5.코드가 간결한가요?
  네.
