module restart_strategies

export restart_known_fstar,restart_mono,restart_extra_eps_algo,no_restart

struct Infos
    funvals::Array{Float64,1}
    grads::Array{Float64,2} #find a good type for matrices
    xs::Array{Float64,2} #same
    restarts::Array{Int64,1}
end

function eps_algo(u,k)
    n = length(u)
    res_ = zeros(n)
    inter = Array(u)
    res = Array(u)
    for i in 1:(2*k)
        for j in 1:(n-i)
            z =  1/(inter[j+1] - inter[j])
            #if z == Inf
                #res[j] = res_[j+1]
            #else
                res[j] = res_[j+1] + z
            #end
        end
        res_ = Array(inter)
        inter = Array(res)
    end
    return res[1:(n-2*k)]
end

function extra_damien(s::Array{Float64,1},k::Int64,lambda::Float64)::Float64
    # s should be of length 2k+1
    # vector of 2k delta_s  
    delta_s = s[2:end].-s[1:(end-1)]
    #n = Int(length(delta_s)/2) 
    S = zeros(k,k)
    for i = 1:k
        S[:,i] = delta_s[i:(i+k-1)] 
    end
    delta_S = S'*S
    #if abs(det(delta_S)) < 10^-20
    #    a = (delta_S + lambda*Matrix(I,k,k))\(S'*delta_s[(k+1):end])
    #else 
            a = (S+lambda*randn(k,k))\(delta_s[(k+1):end])
    #end
    #a = a./sum(a)
    #a = -(S'*S + lambda*Matrix(I,k,k))\(S'*delta_s[(k+1):end])
        return (s[k+1] -s[1:k]'*a )/(1-sum(a))
end

function extra_damien_reborn(info::Infos,k::Int64,alpha::Float64,lambda::Float64)::Float64
    # s should be of length 2k+1
    # vector of 2k delta_s 
    n = length(info.funvals) -1
    S_ = [(alpha*info.grads[:,i] + (1-alpha)*(info.grads[:,i+1]))'*(info.xs[:,i+1]-info.xs[:,i]) + lambda/2*norm(info.xs[:,i+1]-info.xs[:,i])^2 for i in 1:n ]
    #n = Int(length(delta_s)/2) 
    S = zeros(k,k)
    for i = 1:k
        S[:,i] = S_[i:(i+k-1)] 
    end
    a = (S\(S_[(k+1):end]))
    return (info.funvals[k+1] -info.funvals[1:k]'*a )/(1-sum(a))
    #return (S_[k+1] - S_[1:k]'*a )/(1-sum(a))
end

no_restart(info::Infos,eps::Float64) = false

function restart_known_fstar(info::Infos,eps::Float64,fstar::Float64)::Bool
    return info.funval[end] - fstar < eps
end
function restart_mono(info::Infos,eps::Float64)::Bool
   return length(info.funval)>=2 && info.funval[end-1]<info.funval[end]
end
function restart_extra_eps_algo(info::Infos,max_length_btw_restarts::Int64,eps::Float64,rank::Int64,)::Tuple{Bool,Float64}
    last = info.restarts[end]
    current_funval = info.funval[last:2:end]
    if length(current_funval) >= max(2*rank+1,max_length_btw_restarts)
            extra = eps_algo(current_funval[(end-2*rank):end],rank)
            #extra = extra_damien(current_funval[(end-2*rank):end],rank,1e-14)
            return (info.funval[end]-extra[1] < eps && extra[1] < minimum(info.funval),min(extra[1],minimum(info.funval)))
    else 
        return (false,minimum(info.funval))
    end
end

function restart_extra_reborn(info::Infos,max_length_btw_restarts::Int64,eps::Float64,rank::Int64,alpha::Float64,lambda::Float64)::Tuple{Bool,Float64}
    last = info.restarts[end]
    current_funval = info.funval[last:1:end]
    if length(current_funval) >= max(2*rank+1,max_length_btw_restarts)
            #extra = eps_algo(current_funval[(end-2*rank):end],rank)
            extra = extra_damien_reborn(info,alpha,lambda)
            return (info.funval[end]-extra[1] < eps && extra[1] < minimum(info.funval),min(extra[1],minimum(info.funval)))
    else 
        return (false,minimum(info.funval))
    end
end

end