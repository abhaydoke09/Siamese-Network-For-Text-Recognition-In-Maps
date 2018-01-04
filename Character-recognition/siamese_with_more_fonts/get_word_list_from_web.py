import urllib2
import re
from random import shuffle


alphabets = [chr(i) for i in range(97, 97+26)]
word_list = []

for ch in alphabets:
    url = 'https://www.thefreedictionary.com/words-that-start-with-'+ch
    page = urllib2.urlopen(url)
    html_content = page.read()
    temp_list = []
    for match in re.finditer(r'\<a href=\"([a-zA-Z]+)\"\>', html_content):
        word = match.group(1)
        if len(word) in range(3,15):
            temp_list.append(word)
    shuffle(temp_list)
    temp_list = temp_list[:40]
    word_list.extend(temp_list)

f=open('balanced_words.txt','w')
s1='\n'.join(word_list)
f.write(s1)
f.close()