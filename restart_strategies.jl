module restart_strategies

export restart_known_fstar,restart_mono,restart_extra_eps_algo

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



function restart_known_fstar(last,funval,x,eps,fstar)::Bool
    return funval[end] - fstar < eps
end
function restart_mono(last,funval,x,eps)::Bool
   return length(funval)>=2 && funval[end-1]<funval[end]
end
function restart_extra_eps_algo(last::Int64,funval::Array{Float64,1},x::Union{Array{Float64,1},Array{Float64,2}},eps::Float64,rank::Int64)::Tuple{Bool,Float64}
    current_funval = funval[last:2:end]
    if length(current_funval) >= 2*rank+1
            extra = eps_algo(current_funval[(end-2*rank):end],rank)
            return (funval[end]-extra[1] < eps && extra[1] < minimum(funval),min(extra[1],minimum(funval)))
    else 
        return (false,-Inf)
    end
end

end