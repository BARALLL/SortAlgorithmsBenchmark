from array import array
import time
import matplotlib.pyplot as plt
import numpy as np
from random import randint, randrange
import sys



def randArray(size):
    return array('i', [randint(0, 100) for i in range(size)])    #return a random array of size given in argument 



def isSorted(A):      #test if array is sorted (in ascending order)
    arrSorted = True
    i = 0
    while i < len(A)-1 and arrSorted:
        if A[i] > A[i+1]:
            arrSorted = False
        i += 1
    assert arrSorted, "Array is not sorted correctly"   #otherwise exit program



def bubbleSort(A): 
    length = len(A)
    timeStart = time.perf_counter()
    for i in range(length):                 #loop through the array until each element is correctly sorted
        for j in range(0, length-i-1): 
            if A[j] > A[j + 1]: 
                A[j], A[j + 1] = A[j + 1], A[j] #swap adjacent elements if in the wrong order
    return time.perf_counter()-timeStart
  


def insertionSort(A):
    temp = 0
    length = len(A)
    timeStart = time.perf_counter()         #python translation of the one seen in the lesson
    for j in range(1, length):
        key = A[j]
        i = j-1
        while i >= 0 and A[i] > key:
            A[i+1] = A[i]
            i-= 1
        A[i+1] = key
    return time.perf_counter()-timeStart



def fasterInsertion(A):                 #faster because does less comparaison using for break than while
    length = len(A)                     #and faster swap
    timeStart = time.perf_counter()     #worst case for insertion is a great case for this insertion sort
    for i in range(1, length):          #close to bubble sort in writing
        for j in range(i):
            if A[i] < A[j]:
                A[j], A[j+1:i+1] = A[i], A[j:i]
                break
    return time.perf_counter()-timeStart


def mergeSort(A):
    if len(A) <= 1:
        return A

    mid = len(A) // 2
    left = A[:mid]
    right = A[mid:]

    left = mergeSort(left)          #divide and conquer
    right = mergeSort(right)

    i = j = 0
    sortedA = array('i')
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            sortedA.append(left[i])     #sort left and right arrays
            i += 1
        else:
            sortedA.append(right[j])
            j += 1

    sortedA += left[i:]     #concatenate them together
    sortedA += right[j:]
    return sortedA





def quickSort(A, pivot):
    if A == []: 
        return []
    else:
        inf = quickSort([x for x in A[1:] if x < pivot], pivot) #sort list of elements smaller than pivot
        sup = quickSort([x for x in A[1:] if x >= pivot], pivot) #sort list of elements greater than pivot
        return inf + [pivot] + sup #combine both lists




def testFunction(str, rand):                  #test time taken by the algorithm given by str in the worst case
    times = []
    x = []
    A = array('i')
    
    if str == insertionSort or str == bubbleSort or str == fasterInsertion:
        for i in range(arraySize, 0, -pas):      
            if rand:
                A += randArray(pas)                 #average case, with random array
            else:
                for j in range(pas):                #construct worst case array entry for insertion sort or bubble sort (reversed sorted)
                    A.append(i-j)                   #construct the array by adding pas * element for each test of this algorithm
            
            B = A[:]
            times.append(str(B))                #add the time taken returned to the list of time
            x.append(arraySize-i)               #for this size of array
            isSorted(B)                         #check if array correctly sorted

    elif str == mergeSort:
        A.append(1)
        index1 = 0
        index2 = 1
        if rand:
            A += randArray(1)

        for j in range(2, arraySize, pas):
            if rand:
                A += randArray(pas)
            else:
                for i in range(j+j%2, j+pas, 2):      #construct worst case array entry for merge sort
                    if i % 4 != 0:
                        A.insert(index2, i)          #insert(index, value)
                        A.insert(index1+1, i+1)        #insert+1 to emulate an append at index middle of array
                        index2 = A.index(2)
                    else:
                        A.insert(index2+1, i)       #worst case for  [1, 2, 3, 4, 5, 6, 7, 8, 9] is [5, 9, 1, 7, 3, 6, 2, 8, 4]
                        A.insert(index1, i+1)
                        index1 = A.index(1)
                        index2 = A.index(2)
            
            B = A[:]
            timeStart = time.perf_counter()
            B = str(B)
            times.append(time.perf_counter()-timeStart)
            x.append(j)
            isSorted(B)
    

    elif str == quickSort:
        for i in range(0, arraySize, pas):  #worst case is array already sorted
            if rand:
                A += randArray(pas)
            else:
                for j in range(pas):  #construct worst case array for quicksort
                    A.append(i+j)

            B = A[:]
            if rand:
                pivot = A[randint(0, len(A)-1)]    #for average case pivot is a random element of the array (tends to avoid worst case results)
            else:    
                pivot = A[0]        #worst pivot for worst case is the first element of the array

            timeStart = time.perf_counter()
            B = str(B, pivot)
            times.append(time.perf_counter()-timeStart)
            x.append(i)
            isSorted(B)

    elif str == sorted:
        for i in range(arraySize, 0, -pas):      
            if rand:
                A += randArray(pas)                 #average case, with random array
            else:
                for j in range(pas):                #construct reversed sorted array (worst case) for timsort (algorithm used by sorted function)
                    A.append(i-j)                  
            
            B = A[:]
            timeStart = time.perf_counter()
            B = sorted(B)
            times.append(time.perf_counter()-timeStart)
            x.append(arraySize-i)
            isSorted(B)

    return times, x




def test(str, rand):        #test the algorithm defined by str in the average case (rand array) or worst case
    times = []
    moy = []
    times, x = testFunction(str, rand)
    for i in range(curveAveraging):             #smoothes the curve by averaging the time for each array size
        moy, x = testFunction(str, rand)
        times = [times[i] + moy[i] for i in range(len(moy))]
    return [elem/curveAveraging for elem in times], x






def polyPlot(x, times, win, st):
    poly = np.polyfit(x, times, 2)      #calculate then plot the polynomial fit of the (x, times) data
    polyCurve = np.poly1d(poly)

    win.plot(x, times, 'r')
    win.plot(x, polyCurve(x), 'b', linewidth=1)

    win.set_xlabel('Array size', fontsize=10)
    win.set_ylabel('Execution time of ' + st, fontsize=10)
    win.set_title('Execution time according to the size of the array of ' + st, fontsize=10)
    win.legend([st, 'Polynomial fit'], fontsize=10) # + str(poly[0]) + 'x^2' can be added to show the x^2 coefficient on each graph


    
arraySize = 503             #data entry -- can be modified according to the needs --
pas = 20
curveAveraging = 10
rand = False                 #rand set to False => worst case, to True => average case (random array)

#print(sys.getrecursionlimit())
#sys.setrecursionlimit(2000) #python default recursion limit is 1000, needed to be changed if arraySize too big > 1000





figure, ax = plt.subplots()         #algorithm comparaison figure
fig, axs = plt.subplots(3, 2, sharey=True)  # each subplot share the same y axis

fig.tight_layout()
fig.subplots_adjust()


#for each algorithm it will test the time taken to sort the array of length from 0 to arraySize
#Each test is run curveAveraging times to average sorting time in order to minimize
#indirect time (occuped processor...)

#2 figure are then showed, 1 for the comparaison of each function and the other one for a
#comparaison between the curve given and the 2nd degree polynomial approximation of it


times, x = test(bubbleSort, rand)
ax.plot(x, times, 'r')              #plot the dataset to the comparative figure

polyPlot(x, times, axs[0,0], 'Bubble Sort') #calculate polynomial fit of (x, times) dataset and plot it on axs[0,0] (subplot 0, 0 of axis axs)



times, x = test(insertionSort, rand)
ax.plot(x, times, 'b')

polyPlot(x, times, axs[0,1], 'Insertion Sort')



times, x = test(fasterInsertion, rand)
ax.plot(x, times, 'c')

polyPlot(x, times, axs[1, 0], 'Faster Insertion Sort')



times, x = test(sorted, rand)
ax.plot(x, times, 'm')

polyPlot(x, times, axs[1, 1], 'TimSort')



times, x = test(quickSort, rand)
ax.plot(x, times, 'g')

polyPlot(x, times, axs[2, 0], 'Quick Sort')



times, x = test(mergeSort, rand)
ax.plot(x, times, 'k')

polyPlot(x, times, axs[2, 1], 'Merge Sort')


#plot info for comparaison figure
ax.set_xlabel('Array size')
ax.set_ylabel('Execution time of the function used')
ax.set_title('Comparison of the execution time according to the size of the array of several sorting algorithms')
ax.legend(['Bubble Sort','Insertion Sort', 'Faster Insertion Sort', 'TimSort (python sorted function)', 'Quick Sort', 'Merge Sort'])
plt.show()