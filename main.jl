using Distances

MAX_ITER = 100
CONVERGE_THRESHOLD  = 1.0

rawdata = readdlm("p654.txt");
n = rawdata[1,1];
data = Array{Float64}(rawdata[2:n+1,1:2])

function kmeans(data,k)
    centroids = findCentroids(data, initlabels(size(data,1), k), k)
    oldCentroids = nothing
    iteration = 0
    while iteration < MAX_ITER && !converged(centroids, oldCentroids)
        iteration = iteration + 1
        oldCentroids = centroids
        labels = findLabels(data, centroids)
        centroids = findCentroids(data, labels, k)
    end
    centroids
end

converged(newcent,oldcent) = oldcent != nothing && mean(colwise(Euclidean(), newcent', oldcent')) < CONVERGE_THRESHOLD
initlabels(n,k) = rand(1:k, n)
findCentroids(data, labels, k) = reduce(vcat,[sum(labels.==ki)>0 ? mean(data[labels.==ki,:],1) : rand(2)' for ki in 1:k])
findLabels(data, centroids) = begin M = pairwise(Euclidean(),data', centroids'); [indmin(M[i,:]) for i in indices(M,1)] end
evaluate(data,facilities) = sum(minimum(pairwise(Euclidean(),data', facilities'),2))
