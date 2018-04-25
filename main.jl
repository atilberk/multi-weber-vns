using Distances

function kmeans(data,k,initlabels)
    labels = initlabels # != nothing ? initcentroids : findCentroids(data, initlabels(size(data,1), k), k)
    oldlabels = nothing
    iteration = 0
    while iteration < MAX_ITER && labels != oldlabels #!converged(findCentroids(centroids, oldCentroids)
        iteration = iteration + 1
        oldlabels = labels
        centroids = findCentroids(data, labels, k)
        labels = findLabels(data, centroids)
    end
    labels
end

initlabels(n,k) = rand(1:k, n)
randomPoint(data)=mean(data,1) .+ (std(data,1).*(rand(2).-0.5)')
findCentroids(data, labels, k) = reduce(vcat,[sum(labels.==ki)>0 ? mean(data[labels.==ki,:],1) : randomPoint(data) for ki in 1:k])
findLabels(data, centroids) = begin M = pairwise(Euclidean(),data', centroids'); [indmin(M[i,:]) for i in indices(M,1)] end
converged(newcent,oldcent) = oldcent != nothing && mean(colwise(Euclidean(), newcent', oldcent')) < CONVERGE_THRESHOLD
evaluate(data,facilities) = sum(minimum(pairwise(Euclidean(),data', facilities'),2))

function vns(data,numFacility,initlabels)
    N = [neighbourhood(numFacility,k) for k in 1:MAX_NEIGHBORHOOD]
    #centroids = initcentroids # != nothing ? initcentroids : minimum((c->(evaluate(data,c),c)).([kmeans(data,numFacility) for i in 1:INIT_TRIAL]))[2]
    labels = initlabels
    oldlabels = nothing
    iteration = iteration_wo_improvement = 0
    while iteration < MAX_ITER && iteration_wo_improvement < MAX_PATIENCE
        @show iteration, iteration_wo_improvement = iteration + 1, iteration_wo_improvement + 1
        oldlabels = labels
        #println(evaluate(data,centroids))
        k = 1
        while k <= MAX_NEIGHBORHOOD
            newlabels = getSomeSolution(data,labels,N[k],numFacility)
            if improved(data, findCentroids(data, newlabels, numFacility), findCentroids(data, labels, numFacility))
                labels=newlabels
                println("HERE")
                break
            else
                k = k + 1
            end
        end
        if k <= MAX_NEIGHBORHOOD
            iteration_wo_improvement = 0
        end
        append!(KVALUES, k)
    end
    labels
end

getSomeSolution(d,ll,p,n) = begin l = p(ll); l = MODE==:Basic?localSearch(d,l,n):l; end
neighbourhood(n,k) = ll->begin l = copy(ll); s = randperm(length(l))[1:k]; l[s] = reassign(l[s],n); l; end
reassign(seq,n) = begin c = rand(1:n,length(seq)); reduce(|,(c .== seq))?reassign(seq,n):c; end
improved(d,nc,oc) = evaluate(d,nc) < evaluate(d,oc)

function localSearch(data,labels,numFacility)
    oldlabels = nothing
    iteration = 0
    centroids = findCentroids(data,labels,numFacility)
    while iteration < MAX_ITER && labels != oldlabels
        iteration = iteration + 1
        oldlabels = labels
        N = allLocalNeigbours(labels,numFacility)
        bestNeighbour = N[indmin((l->evaluate(data,findCentroids(data,l,numFacility))).(N))]
        bestCentroids=findCentroids(data,bestNeighbour,numFacility)
        if improved(data, bestCentroids, centroids)
            labels = bestNeighbour
            centroids = bestCentroids
            println(evaluate(data,centroids))
        else
            println("local optimum!")
        end
    end
    labels
end

#expandN(v)=(els)->reduce(hcat,[[[e i] for i in v] for e in els])[:]
#allChangeSet(n,k)= k>1 ? expandN(1:n)(allChangeSet(n,k-1)) : [[i] for i in 1:n]
allLocalNeigbours(l,n)= filter(ll->ll!=l,[[l[1:i-1];j;l[i+1:end]] for j in 1:n, i in indices(l,1)][:])
Base.isless(a::Array,b::Array)=true

# Maximum number of iterations
MAX_ITER = 200
# Maximum number of iterations without improvement
MAX_PATIENCE = 3
# minimum euclidean distance for convergence
CONVERGE_THRESHOLD = 0.0
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

KARRAY=[]
KVALUES=[]

function main(numFacilities=[3,5,8], numTrials=5)
    global KARRAY=[]
    for nf in numFacilities
        for i in 1:numTrials
            global KVALUES=[]
            il=initlabels(size(data,1),nf)
            #ic=findCentroids(data,il,nf)
            t0=time()
            reskms=evaluate(data,findCentroids(data,kmeans(data,nf,il),nf))
            t1=time()
            resvns=evaluate(data,findCentroids(data,vns(data,nf,il),nf))
            t2=time()
            println("$nf\t$i\t$reskms\t$(@sprintf("%.3f",1000*(t1-t0))) ms\t$resvns\t$(@sprintf("%.3f",t2-t1)) s")
            append!(KARRAY,[KVALUES])
        end
    end
end
