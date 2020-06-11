
# coding: utf-8

# In[102]:

file=open("E:/E/IIIT Delhi/IR/Assign6/CollegeMsg/CollegeMsg.txt")
lines=file.readlines()
nodes={}
totalNodes=[]
numEdges=0
numNodes=0
totalEdges=0

for line in lines:
    temp=line.split(" ")
    u=int(temp[0])
    v=int(temp[1])
    
    if u not in nodes:
        nodes[u]=[]
        nodes[u].append(v)
        numEdges+=1
        totalEdges+=1
    else:
        if v not in nodes[u]:
            nodes[u].append(v)
            numEdges+=1
        totalEdges+=1
        
    totalNodes.append(u)
    totalNodes.append(v)
    
totalNodes=list(set(totalNodes))

numNodes=len(totalNodes)
print("Total number of nodes ",numNodes)
print("Total Number of static edges ",numEdges)
print("Total Number of network edges ",totalEdges)
totalNodes.sort()


# In[49]:

maxNode=max(totalNodes)
minNode=min(totalNodes)
#print(minNode,maxNode)
matrix = [[0 for i in range(maxNode)] for j in range(maxNode)]
for node in nodes:
    for n in nodes[node]:
        matrix[node-1][n-1]=1
        
#matrix.pop(0)
#print(len(matrix))


# In[84]:

import numpy as np
inDegree=0
outDegree=0
maxIn=0
maxOut=0

inDegreeDict={}
outDegreeDict={}
centralityMeasureIn={}
centralityMeasureOut={}


i=0
for rows in matrix:
    t=rows.count(1)
    outDegree+=t
    if t in outDegreeDict:
        outDegreeDict[t]+=1
    else:
        outDegreeDict[t]=1
        
    if t>matrix[maxOut].count(1):
        maxOut=i
        #print(maxOut)
    i+=1
    centralityMeasureOut[i]=t
        

matrix1=np.array(matrix)
matrix1=matrix1.transpose().tolist()

i=0
for rows in matrix1:
    t=rows.count(1)
    inDegree+=t
    
    if t in inDegreeDict:
        inDegreeDict[t]+=1
    else:
        inDegreeDict[t]=1
        
        
    if t>matrix1[maxIn].count(1):
        maxIn=i
    i+=1
    centralityMeasureIn[i]=t #Only in-degree is considered in Degree Centrality.
    
    
print("Avg In-Degree ",inDegree/len(matrix))
print("Avg Out-Degree ",outDegree/len(matrix))
print("Node with max in-degree ",maxIn+1)
print("Node with max out-degree ",maxOut+1)
density=numEdges/(numNodes*(numNodes-1))
print("Density of Network is ",density)


# In[85]:

import matplotlib.pyplot as plt
import operator

inDegreeDict=dict( sorted(inDegreeDict.items(), key=operator.itemgetter(1),reverse=True))
outDegreeDict=dict( sorted(outDegreeDict.items(), key=operator.itemgetter(1),reverse=True))
print(len(inDegreeDict),len(outDegreeDict))

x=[]
y=[]
for degree in inDegreeDict:
    x.append(degree)
    val=inDegreeDict[degree]/numNodes
    y.append(val)
    

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(x,y)
plt.ylabel("Fraction of nodes")
plt.xlabel("Degree")
plt.title("In-Degree Distribution")
plt.legend()
plt.show()



fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(x[:15],y[:15])
plt.ylabel("Fraction of nodes")
plt.xlabel("Degree")
plt.title("In-Degree Distribution of first 15 degrees")
plt.legend()
plt.show()

x=[]
y=[]
for degree in outDegreeDict:
    x.append(degree)
    val=outDegreeDict[degree]/numNodes
    y.append(val)
    

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(x,y)
plt.ylabel("Fraction of nodes")
plt.xlabel("Degree")
plt.title("Out-Degree Distribution")
plt.legend()
plt.show()



fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(x[:15],y[:15])
plt.ylabel("Fraction of nodes")
plt.xlabel("Degree")
plt.title("Out-Degree Distribution of first 15 degrees")
plt.legend()
plt.show()


# In[103]:

x=[]
y=[]
i=1
for keys in centralityMeasureIn:
    x.append(keys)
    y.append(centralityMeasureIn[keys])
    if i<21:
        print("Node =",keys," In-Degree Centrality Measure =",centralityMeasureIn[keys])
        i+=1
    
import matplotlib.pyplot as plt
plt.plot(x,y)
plt.title("In-Degree Centrality Measure")
plt.ylabel("Centrality Measure Value")
plt.xlabel("Nodes")
plt.show()


x=[]
y=[]
i=1
for keys in centralityMeasureOut:
    x.append(keys)
    y.append(centralityMeasureOut[keys])
    if i<21:
        print("Node =",keys," Out-Degree Centrality Measure =",centralityMeasureIn[keys])
        i+=1
    
import matplotlib.pyplot as plt
plt.plot(x,y)
plt.title("Out-Degree Centrality Measure")
plt.ylabel("Centrality Measure Value")
plt.xlabel("Nodes")
plt.show()


# In[54]:

clusteringCoefficient={}
for i in range(len(matrix)):
    neighbours=[]
    neighEdges=0
    #neighbours.append(i)
    for j in range(len(matrix)):
        if matrix[i][j]==1:
            neighbours.append(j)
        elif matrix[j][i]==1:
            neighbours.append(j)
    
    for n1 in neighbours:
        for n2 in neighbours:
            if matrix[n1][n2]==1:
                neighEdges+=1
    
    length=len(neighbours)
    #print(length,neighEdges)
    if length==0 or length==1:
        clusteringCoeffVal=0
    else:
        clusteringCoeffVal=neighEdges/(length*(length-1))
    clusteringCoefficient[i+1]=clusteringCoeffVal
    
print(len(clusteringCoefficient))


# In[104]:

for i in range(1,21):
    print("Node ",i," clustering coeff = ",clusteringCoefficient[i])

#print("max= ",clusteringCoefficient[min(clusteringCoefficient,key=clusteringCoefficient.get)])
clusteringCoeffDist={}
clusteringCoeffDist[0.0]=0
clusteringCoeffDist[0.2]=0
clusteringCoeffDist[0.4]=0
clusteringCoeffDist[0.6]=0
clusteringCoeffDist[0.8]=0
clusteringCoeffDist[1.0]=0

one=0
for i in clusteringCoefficient:
    if clusteringCoefficient[i]<0.2:
        clusteringCoeffDist[0.0]+=1
    elif 0.2<=clusteringCoefficient[i]<0.4:
        clusteringCoeffDist[0.2]+=1
    elif 0.4<=clusteringCoefficient[i]<0.6:
        clusteringCoeffDist[0.4]+=1
    elif 0.6<=clusteringCoefficient[i]<0.8:
        clusteringCoeffDist[0.6]+=1
    elif 0.8<=clusteringCoefficient[i]<1.0:
        clusteringCoeffDist[0.8]+=1
    else:
        clusteringCoeffDist[1.0]+=1
        
y=[]
x=[]

for keys in clusteringCoeffDist:
    x.append(keys)
    y.append(clusteringCoeffDist[keys]/numNodes)

print(x,y)
import matplotlib.pyplot as plt

# fig = plt.figure()
# ax = fig.add_axes([1,1,1,1])
# ax.bar(x,y)
# plt.ylabel("Fraction of nodes")
# plt.xlabel("Degree")
# plt.title("Clustering Coefficient Distribution")
# plt.legend()

plt.bar(x, y, tick_label = x, width = 0.2, color = ['red', 'green'])
plt.ylabel("Fraction of Node")
plt.xlabel("Clustering Coeff")
plt.show()
plt.plot(x, y, color='green', linestyle='dashed', linewidth = 3, marker='o', markerfacecolor='blue', markersize=12)
plt.ylabel("Fraction of Node")
plt.xlabel("Clustering Coeff")
plt.show()


# In[56]:

listTupleEdge=[]
for node in nodes:
    for n in nodes[node]:
        listTupleEdge.append((node,n))
    #listTupleEdge.append(tuple(nodes[node]))
    
#print(listTupleEdge)


# In[91]:

import networkx as nx
G=nx.DiGraph()
G.add_nodes_from(totalNodes)
G.add_edges_from(listTupleEdge)


# In[92]:

pr = nx.pagerank(G)
h,a=nx.hits(G)
for i in range(1,11):
    print("page rank ",pr[i],"   Hub ",h[i],"   Authority ",a[i])


# In[101]:

yin=[]
yout=[]
x=[]

i=1
for keys in centralityMeasureIn:
    yin.append(centralityMeasureIn[keys])
    yout.append(centralityMeasureOut[keys])
    x.append(keys)
    i+=1
    if i>51:
        break
    
plt.plot(x,yin,label="In-Degree")
plt.plot(x,yout,label="Out-Degree")
plt.xlabel("Nodes")
plt.ylabel("Degree")
plt.legend()
plt.show()

i=1
ypr=[]
yh=[]
ya=[]
x=[]
for keys in pr:
    ypr.append(pr[keys])
    yh.append(h[keys])
    ya.append(a[keys])
    x.append(keys)
    i+=1
    if i>51:
        break
    
plt.plot(x,ypr,label="Page Rank")
plt.plot(x,yh,label="Hub")
plt.plot(x,ya,label="Authority")
plt.xlabel("Nodes")
plt.ylabel("Scores")
plt.legend()
plt.show()


# In[107]:

ypr=[]
yh=[]
ya=[]
x=[]
for keys in pr:
    ypr.append(pr[keys])
    yh.append(h[keys])
    ya.append(a[keys])
    x.append(keys)
    
plt.xlabel("Nodes")
plt.ylabel("PR Scores")
plt.plot(x,ypr,label="Page Rank")
plt.show()
plt.xlabel("Nodes")
plt.ylabel("HUB Scores")
plt.plot(x,yh,label="Hub")
plt.show()
plt.xlabel("Nodes")
plt.ylabel("Authorith Scores")
plt.plot(x,ya,label="Authority")
plt.show()



# In[ ]:



