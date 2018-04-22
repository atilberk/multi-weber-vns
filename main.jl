using Distances

# Maximum number of iterations
MAX_ITER = 10
# Maximum number of iterations without improvement
MAX_PATIENCE = 1
# minimum euclidean distance for convergence
CONVERGE_THRESHOLD = 1.0
# change up to three customers
MAX_NEIGHBORHOOD = 3
# 1 : Basic (with local search)
# 2 : Reduced (with shaking only)
MODE_SELECT = 1
MODE = [:Basic, :Reduced][MODE_SELECT]

# how many kmeans trial for vns initial solution
INIT_TRIAL = 10;

rawdata = readdlm("p654.txt")
n = rawdata[1,1]
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
    centroids, oldCentroids = minimum((c->(evaluate(data,c),c)).([kmeans(data,numFacility) for i in 1:INIT_TRIAL]))[2], nothing
    iteration = iteration_wo_improvement = 0
    while iteration < MAX_ITER && iteration_wo_improvement < MAX_PATIENCE
        @show iteration, iteration_wo_improvement = iteration + 1, iteration_wo_improvement + 1
        oldCentroids = centroids
        println("vn:",evaluate(data,centroids))
        k = 1
        while k <= MAX_NEIGHBORHOOD
            newCentroids = getSomeSolution(data,centroids,N[k])
            if improved(data, newCentroids, centroids)
                println("v$k:", evaluate(data,newCentroids))
                centroids = newCentroids
                k = 1
                iteration_wo_improvement = 0
            else
                println("no better :(")
                k = k + 1
            end
        end
    end
    centroids
end

getSomeSolution(d,c,p) = begin l = p(findLabels(d,c)); l = MODE==:Basic?localSearch(d,l,size(c,1)):l; findCentroids(d, l, size(c,1)); end
neighbourhood(n,k) = ll->begin l = copy(ll); s = randperm(length(l))[1:k]; l[s] = reassign(l[s],n); l; end
reassign(seq,n) = begin c = rand(1:n,length(seq)); reduce(|,(c .== seq))?reassign(seq,n):c; end
improved(d,nc,oc) = evaluate(d,nc) < evaluate(d,oc)

function localSearch(data,labels,numFacility)
    oldLabels = nothing
    iteration = 0
    while iteration < MAX_ITER && labels != oldLabels
        iteration = iteration + 1
        oldLabels = labels
        N = allLocalNeigbours(labels,numFacility)
        bestNeighbour = N[indmin((l->evaluate(data,findCentroids(data,l,numFacility))).(N))]
        if improved(data, findCentroids(data,bestNeighbour,numFacility), findCentroids(data,labels,numFacility))
            newCentroids = findCentroids(data,bestNeighbour,numFacility)
            println("ls:",evaluate(data,newCentroids))
            labels = bestNeighbour
        else
            println("local minima!")
        end
    end
    labels
end

#expandN(v)=(els)->reduce(hcat,[[[e i] for i in v] for e in els])[:]
#allChangeSet(n,k)= k>1 ? expandN(1:n)(allChangeSet(n,k-1)) : [[i] for i in 1:n]
allLocalNeigbours(l,n)= filter(ll->ll!=l,[[l[1:i-1];j;l[i+1:end]] for j in 1:n, i in indices(l,1)][:])
Base.isless(a::Array,b::Array)=true
