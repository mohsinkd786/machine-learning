rawNums = [
    [13,4,5],
    [11,6,9],
    [7,3,2]
]
def sortMatrix(rawMatrix):
    nOfRows = len(rawMatrix)
    nOfColumns = len(rawMatrix[0])
    flattenMatrix = []
    i = 0
    tempVal = 0
    while i < nOfRows:
        j = 0
        while j < nOfColumns:
            flattenMatrix.append(rawMatrix[i][j])
            j= j + 1
        i = i + 1
    # sort the 1d list
    flattenMatrix.sort()
    
    i = 0
    k = 0
    while i < nOfRows :
        j = 0 
        while j < nOfColumns:
            rawMatrix[i][j] = flattenMatrix[k]    
            k = k + 1
            j = j + 1
        i = i + 1

    print(rawMatrix)     
     
sortMatrix(rawNums)