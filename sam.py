'''

print("I am aarush")
print(" ")
string = "Python Programming"
print(string[::-1])

#counting vowel in sentences 

vowel = ['a','e','i','o','u']
count =0
word = "programming"

for i in word:
    if i in vowel:
        count+=1
print(count)        


vowel = ['a','e','i','o','u']
count = 0
word = "programming"

for i in word:
    if i not in vowel:
        count+=1
print(count)        

numberlist =[15,20,35,80,90]

maxnub = numberlist[0]

for num in numberlist:
    if maxnub < num:
        maxnub = num
print(maxnub)        

numberlist = [2,3,1,4,5]

mininub = numberlist[0]

for num in numberlist:
    if mininub > num:
        mininub = num
print(mininub)        


numberlist = [1,2,3,4,5,6]

middle = int((len(numberlist)/2))

print([numberlist[middle]])


list1 = [1,2,3,4]
list2 = [5,6,7,8]

resu_list =[]

for i in range(0,len(list1)):
    resu_list.append(list1[i]+list2[i])
print(resu_list)
'''

str1 = "Kiran".lower()
str2 = "Nagargoje".lower()

if(str1 == str1[::-1]):
    print("Same")
else:
    print("Differ")    






























