import difflib
import string

SEED = 0

def random_char_control(s1, s2, shuffle):
    '''
    Pass shuffle as a 26 length string where 
    shuffle[i] = transformation for 'a' + i (in lowercase)
    '''
    ret = ""
    for i,s in enumerate(difflib.ndiff(s1, s2)):
        if s[0]==' ': 
            ret += s2[i]
        elif s[0]=='-':
            raise ValueError(f'{s1}-> {s2} involves deletions and random_char_control cant handle that')
        elif s[0]=='+':
            c = s[-1]
            print(s[0], s[-1], i)
            if c.isalpha():
                if c.isupper(): #Assuming: we need to preserve upper-case letters
                    c = shuffle[string.ascii_uppercase.index(c)]
                    c = c.upper()
                elif c.islower():
                    c = shuffle[string.ascii_lowercase.index(c)]
            ret += c
    return s1, ret

# import random

# s = string.ascii_lowercase[:26]
# l = list(s)
# random.Random(SEED).shuffle(l)
# result = ''.join(l)
# print(result)

# s1 = "I do play"
# s2 = "I don't play"
# t1, t2 = random_char_control(s1, s2, result)
# print(t1, t2)