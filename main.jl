using Distances

# Maximum number of iterations
MAX_ITER = 100
# Maximum number of iterations without improvement
MAX_PATIENCE = 10
# minimum euclidean distance for convergence
CONVERGE_THRESHOLD  = 1.0
# change up to three customers
MAX_NEIGHBORHOOD = 3
# 1 : Basic (with local search)
# 2 : Reduced (with shaking only)
MODE_SELECT = 2
MODE = [:Basic, :Reduced][MODE_SELECT]

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

initlabels(n,k) = rand(1:k, n)
findCentroids(data, labels, k) = reduce(vcat,[sum(labels.==ki)>0 ? mean(data[labels.==ki,:],1) : rand(2)' for ki in 1:k])
findLabels(data, centroids) = begin M = pairwise(Euclidean(),data', centroids'); [indmin(M[i,:]) for i in indices(M,1)] end
converged(newcent,oldcent) = oldcent != nothing && mean(colwise(Euclidean(), newcent', oldcent')) < CONVERGE_THRESHOLD
evaluate(data,facilities) = sum(minimum(pairwise(Euclidean(),data', facilities'),2))

function vns(data,numFacility)
    N = [neighbourhood(numFacility,k) for k in 1:MAX_NEIGHBORHOOD]
    centroids, oldCentroids = kmeans(data,numFacility), nothing
    iteration = iteration_wo_improvement = 0
    while iteration < MAX_ITER && iteration_wo_improvement < MAX_PATIENCE #!improved(data, centroids, oldCentroids)
        @show iteration, iteration_wo_improvement = iteration + 1, iteration_wo_improvement + 1
        oldCentroids = centroids
        k = 1
        while k <= MAX_NEIGHBORHOOD
            newCentroids = getSomeSolution(data,centroids,N[k])
            if improved(data, newCentroids, centroids)
                @show centroids = newCentroids
                k = 1
                iteration_wo_improvement = 0
            else
                k = k + 1
            end
        end
    end
    centroids
end

getSomeSolution(d,c,p) = findCentroids(d, p(findLabels(d,c)), size(c,1))
improved(d,nc,oc) = evaluate(d,nc) < evaluate(d,oc)
neighbourhood(n,k) = ll->begin l = copy(ll); s = randperm(length(l))[1:k]; l[s] = reassign(l[s],n); l end
reassign(seq,n) = begin c = rand(1:n,length(seq)); reduce(|,(c .== seq))?reassign(seq,n):c; end
