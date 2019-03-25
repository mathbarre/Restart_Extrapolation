module Restart_Algo
using LinearAlgebra
export functional,parameters,output,Nesterov_Acc,Fista,UFastGradient,AccGradientMtx,CovSelect

struct Infos
    funvals::Array{Float64,1}
    grads::Array{Float64,2} #find a good type for matrices
    xs::Array{Float64,2} #same
    restarts::Array{Int64,1}
end

struct functional
    f::Function #smooth function value
    g::Function #nonsmooth function value
    grad_f::Function #gradient of smooth part
    prox_g::Function #proximal for proximable part
    argmin_phi::Function #argminphi resume to prox of g when where in the euclidean setup
    pb_norm::Function #pb_norm is the norm in which the strong convexityt of the function in the bregman divergence is considered
end

struct parameters
    x0::Union{Array{Float64,1},Array{Float64,2}} #initial value
    q::Float64 #q=0 => nesterov normal, q=1 => gradient descent, q= mu\L optimal
    step::Float64 #stepsize for gradient step
    beta::Float64 #constant momentum if -1 follow the recursive rule for momentum 
    n_iter::Int64 # number of iterations
    gamma::Float64 #for the restart
    L_0::Float64
    tol::Float64
    max_linesearch::Int64
end

function parameters(x0::Array{Float64,1},q::Float64,step::Float64,beta::Float64,n_iter::Int64,gamma::Float64) 
     parameters(x0,q,step,beta,n_iter,gamma,0.0,0.0,0)
end
function parameters(x0::Array{Float64,2},q::Float64,step::Float64,beta::Float64,n_iter::Int64,gamma::Float64) 
    parameters(x0,q,step,beta,n_iter,gamma,0.0,0.0,0)
end
function parameters(x0::Array{Float64,1},n_iter::Int64,gamma::Float64,L_0::Float64,tol::Float64,max_linesearch::Int64) 
    parameters(x0,0.0,0.0,0.0,n_iter,gamma,L_0,tol,max_linesearch)
end
function parameters(x0::Array{Float64,2},n_iter::Int64,gamma::Float64,L_0::Float64,tol::Float64,max_linesearch::Int64) 
    parameters(x0,0.0,0.0,0.0,n_iter,gamma,L_0,tol,max_linesearch)
end
struct output
    xn::Union{Array{Float64,1},Array{Float64,2}} #final point
    funval::Array{Float64,1} #function values
    restart_index::Array{Int64,1} #iterations where restarts occured
    extrapolation::Array{Float64,1} #extrapolations of f* when its computed 
end



function Nesterov_Acc(f::functional,params::parameters,restart_strategy::Function,extra::Bool)::output
    theta = 1
    x = Array(params.x0)
    x_ = Array(x)
    y = Array(params.x0)
    funval = [f.f(x)+f.g(x)]
    grads = f.grad_f(x)
    xs = x
    last_restart = Int64(1)
    restarts = []
    extras=[]
    eps = 10*f.f(x)
    for k in 1:params.n_iter
        
        x_ = y-params.step*f.grad_f(y)
        theta_ = ((params.q-theta^2)+sqrt((params.q-theta^2)^2 + 4*theta^2))/2
        if params.beta == -1.0
            beta = theta*(1-theta)/(theta^2 + theta_)
        else 
            beta = params.beta 
        end
        y = x_ +beta*(x_-x)
        x = Array(x_)
        theta = theta_
        funval = [funval;f.f(x)+f.g(x)]
        grads = [grads f.grad_f(x)]
        xs = [xs x]
        info = Infos(funval,grads,xs,restarts)
        ex = -Inf
        if !extra
            b = restart_strategy(info,eps)
        else 
            (b,ex) = restart_strategy(info,eps)
        end
        # if !extra
        #     b = restart_strategy(last_restart,funval,x,eps)
        # else 
        #     (b,ex) = restart_strategy(last_restart,funval,x,eps)
        # end
        if b
            last_restart = k
            theta = 1
            y = Array(x)
            restarts = [restarts;k]
            eps *= exp(-params.gamma)
            
        end
        extras = [extras;ex]
    end
    out = output(x,funval,restarts,extras)
    return out
end



function Fista(f::functional,params::parameters,restart_strategy::Function,extra::Bool)::output
    t= 1
    x = Array(params.x0)
    x_ = Array(x)
    y = Array(params.x0)
    funval = [f.f(x)+f.g(x)]
    last_restart=1
    restarts = []
    extras=[]
    eps = 10*f.f(x)
    for k in 1:params.n_iter
        x_ = f.prox_g(y-params.step*f.grad_f(y))
        t_ = 0.5*(1+sqrt(1+4*t^2))
        beta = (t-1)/t_
        y = x_ +beta*(x_-x)
        x = Array(x_)
        t = t_
        funval = [funval;f.f(x)+f.g(x)]
        ex = -Inf
        if !extra
            b = restart_strategy(last_restart,funval,x,eps)
        else 
            (b,ex) = restart_strategy(last_restart,funval,x,eps)
        end
        if b
            last_restart = k
            t=1
            y = Array(x_)
            restarts = [restarts;k]
            eps *= exp(-params.gamma)
            
        end
        extras = [extras;ex]
    end
    out = output(x,funval,restarts,extras)
    return out
end



function UFastGradient(f::functional,params::parameters,restart_strategy::Function,extra::Bool)::output
    #universal fast gradient method from nesterov
    #argminphi resume to prox of g when where in the euclidean setup
    #v_k = argmin \phi_k = argmin \eta(x_0,x) +\sum_i=1^k a_i(<grad_f(x_i),x> + g(x))
   
    k =1
    x_0 = params.x0
    x_k = Array(x_0)
    y_k = Array(x_0)
    L_k = params.L_0
    A_k = 0
    a_k = 0
    if size(x_0,2) == 1
        sum_gradient = zeros(size(x_0,1)) #play the role of \phi when euclidiean setup
    else
        sum_gradient = zeros(size(x_0,1),size(x_0,2)) #play the role of \phi when euclidiean setup
    end
    funval = [f.f(x_0)+f.g(x_0)]
    extras=[f.f(x_0)+f.g(x_0)]
    last_restart=1
    restarts = []
    eps = 10*(f.f(x_0)+f.g(x_0))
    while k < params.n_iter
        v_k = f.argmin_phi(x_0,A_k,sum_gradient)

        k_ = 1
        a_ik = 0.5/L_k*(1+sqrt(1+4*A_k*L_k))
        A_ik = A_k+a_ik
        tau_ik = a_ik/A_ik
        x_ik = tau_ik*v_k + (1-tau_ik)*y_k
        x_hat_ik = f.argmin_phi(v_k,a_ik,a_ik*f.grad_f(x_ik))
        y_ik = tau_ik*x_hat_ik+(1-tau_ik)*y_k 
        while k_ < params.max_linesearch && f.f(y_ik) > f.f(x_ik)+f.grad_f(x_ik)[:]'*(y_ik[:] - x_ik[:]) + 0.5*L_k*(f.pb_norm(y_ik-x_ik))^2 + 0.5*params.tol*tau_ik
            L_k *= 2
            a_ik = 0.5/L_k*(1+sqrt(1+4*A_k*L_k))
            A_ik = A_k+a_ik
            tau_ik = a_ik/A_ik
            x_ik = tau_ik*v_k + (1-tau_ik)*y_k
            x_hat_ik = f.argmin_phi(v_k,a_ik,a_ik*f.grad_f(x_ik))
            y_ik = tau_ik*x_hat_ik+(1-tau_ik)*y_k 
            k_ +=1
        end
        if k_ == params.max_linesearch
            println("max linesearch")
        end
        L_k /=2
        x_k = Array(x_ik)

        #test mono
        # X = [x_k y_k y_ik]
        # fs = [f.f(x_k)+f.g(x_k);f.f(y_k)+f.g(y_k);f.f(y_ik)+f.g(y_ik)]
        # y_k = Array(X[:,argmin(fs)])

        y_k = Array(y_ik)
        a_k = a_ik
        A_k += a_k
        sum_gradient += a_k*f.grad_f(x_k)
        funval = [funval;f.f(y_k)+f.g(y_k)]
        ex = -Inf
        if !extra
            b = restart_strategy(last_restart,funval,y_k,eps)
        else 
            (b,ex) = restart_strategy(last_restart,funval,y_k,eps)
        end
        if b
            println(k)
            last_restart = k
            x_k= Array(y_k)
            x_0 = Array(x_k)
            L_k = params.L_0
            A_k = 0
            a_k = 0
            if size(x_0,2) == 1
                sum_gradient = zeros(size(x_0,1)) #play the role of \phi when euclidiean setup
            else
                sum_gradient = zeros(size(x_0,1),size(x_0,2)) #play the role of \phi when euclidiean setup
            end
            restarts = [restarts;k]
            eps *= exp(-params.gamma)
            
        end
        extras = [extras;ex]
        k = k+1
        
    end
    out = output(y_k,funval,restarts,extras)
    return out
end

function AccGradientMtx(f::functional,params::parameters,restart_strategy::Function,extra::Bool)
    t= 1
    x = Array(params.x0)
    x_ = Array(x)
    y = Array(params.x0)
    funval = [f.f(x)+f.g(x)]
    last_restart=1
    restarts = []
    eps = 10*f.f(x)
    extras = []
    step = params.step
    for k in 1:params.n_iter

        #adapt_step = true
        #step  *= 1.5^2
        #while adapt_step
            #step /= 1.5
            x_ = f.prox_g(y - step*f.grad_f(y),params.step)
            #adapt_step = (f.f(x_)  > (f.f(y) +(f.grad_f(y)[:]')*(x_[:] - y[:]) + 0.5/step*norm(x_[:]-y[:])^2))
        #end
        t_ = 0.5*(1+sqrt(1+4*t^2))
        beta = (t-1)/t_
        y = x_ +beta*(x_-x)
        x = Array(x_)
        t = t_
        funval = [funval;f.f(x)+f.g(x)]
        ex = -Inf
        if !extra
            b = restart_strategy(last_restart,funval,x,eps)
        else 
            (b,ex) = restart_strategy(last_restart,funval,x,eps)
        end
        if b
            last_restart = k
            t=1
            y = Array(x_)
            restarts = [restarts;k]
            eps *= exp(-params.gamma)
            step = params.step
        end
        extras = [extras;ex]
    end
    out = output(x,funval,restarts,extras)
    return out
end


function CovSelect(tol,alpha,beta,rho,Sigma,max_iter,restart_strategy,extra,gamma)
    n = size(Sigma,1)
    X0 = Array(beta*Matrix(I,n,n))
    X = Array(X0) 
    D2 = n^2/2
    M = 1/alpha^2
    L = M + D2*rho^2/(2*tol)
    sum_grad = inv(X)
    sigma1 = 1/beta^2
    U_hat = zeros(n,n)
    U = max.(min.((X)*rho^2*2*D2/tol,rho),-rho)
    dual_vals = [Inf]
    funval = [-log(det(X)) + (Sigma[:]')*X[:] + rho*sum(abs.(X))]
    Y = Array(X)
    last_restart = 1
    restarts = []
    #tol = 10*funval[1]
    extras = []
    k = 0
    for i = 1:(max_iter) 
        #step 1
        gradf = -inv(X) + Sigma + U
        sum_grad = sum_grad + (k+1)/2*gradf

        #step 2
        G = X - 1/L*gradf
        eig_G = eigen(G)
        V = eig_G.vectors
        gammas = eig_G.values
        lambdas = min.(max.(gammas,alpha),beta)
        Y = V*Diagonal(lambdas)*V'

        #step 3
        S = sigma1/L*sum_grad
        eig_S = eigen(S)
        sigmas = eig_S.values
        V_S = eig_S.vectors
        lambdas_S = min.(max.(1 ./ sigmas,alpha),beta)
        Z = V_S*Diagonal(lambdas_S)*V_S'

        #step 4
        X = 2/(k+3)*Z + (k+1)/(k+3)*Y
        U = max.(min.((X)*rho^2*2*D2/tol,rho),-rho)
        U_hat = (k*U_hat+2*U)/(k+2)

        #step 5
        eig_phi = eigen(inv(Sigma+U_hat))
        sigmas_phi = eig_phi.values
        V_phi = eig_phi.vectors
        lambdas_phi = min.(max.(sigmas_phi,alpha),beta)
        U_ = V_phi*Diagonal(lambdas_phi)*V_phi'
        phi = -log(det(U_))+(Sigma[:]+U_hat[:])'*(U_[:])
        fun = -log(det(Y)) + (Sigma[:]')*Y[:] + rho*sum(abs.(Y))
        dual_gap = fun  - phi 

        k +=1

        dual_vals = [dual_vals;dual_gap]
        funval = [funval;fun]
        ex = -Inf
        if !extra
            b = restart_strategy(last_restart,funval,Y,tol)
        else 
            (b,ex) = restart_strategy(last_restart,funval,Y,tol)
        end
        if b
            k = 0
            last_restart = i
            X = Array(Y)
            restarts = [restarts;i]
            tol *= exp(-gamma)
            L = M + D2*rho^2/(2*tol)
            sum_grad = inv(X)
            U = max.(min.((X)*rho^2*2*D2/tol,rho),-rho)
        end
        extras = [extras;ex]
        #if dual_gap < tol
        #    break
        #end
    end
    return (Y,dual_vals,funval,restarts,extras)


end


function CovSelectOtherProx(tol,alpha,beta,rho,Sigma,max_iter,restart_strategy,extra,gamma,continuation)
    n = size(Sigma,1)
    X0 = Array(beta*Matrix(I,n,n))
    X = Array(X0) 
    D2 = n^2/2
    M = 1/alpha^2
    L = M + D2*rho^2/(2*tol)
    sum_grad = zeros(n,n)
    sigma1 = 1.0
    U_hat = zeros(n,n)
    U = max.(min.((X)*rho^2*2*D2/tol,rho),-rho)
    dual_vals = [Inf]
    huber(x,e) = begin
        mu = e/(2*D2*rho)
        if abs(x) <= mu 
           return rho*x^2/(2*mu)
        else
           return rho*(abs(x) - mu/2)
        end
    end
    f(Y) = -log(det(Y)) + (Sigma[:]')*Y[:] + sum(huber.(Y,tol))
    #f(Y) = -log(det(Y)) + (Sigma[:]')*Y[:] + rho*sum(abs.(Y))
    funval = [f(X)]
    Y = Array(X)
    eps = 10*funval[end]
    last_restart = 1
    restarts = []
    extras = [funval[end]]
    k = 0
    for i = 1:(max_iter) 
        #step 1
        gradf = -inv(X) + Sigma + U
        sum_grad = sum_grad + (k+1)/2*gradf

        #step 2
        adapt_step = true
        L /= 4
        while adapt_step
            L *= 2
            G = X - 1/L*gradf
            eig_G = eigen(G)
            V = eig_G.vectors
            gammas = eig_G.values
            lambdas = min.(max.(gammas,alpha),beta)
            Y = V*Diagonal(lambdas)*V'
            adapt_step = (f(Y) > f(X) + gradf[:]'*(Y[:]-X[:]) +0.5*L*norm(Y[:]-X[:])^2)
        end

        #step 3
        S = X0 - sigma1/L*sum_grad
        eig_S = eigen(S)
        sigmas = eig_S.values
        V_S = eig_S.vectors
        lambdas_S = min.(max.(sigmas,alpha),beta)
        Z = V_S*Diagonal(lambdas_S)*V_S'

        #step 4
        X = 2/(k+3)*Z + (k+1)/(k+3)*Y
        U = max.(min.((X)*rho^2*2*D2/tol,rho),-rho)
        U_hat = (k*U_hat+2*U)/(k+2)

        #step 5
        eig_phi = eigen(inv(Sigma+U_hat))
        sigmas_phi = eig_phi.values
        V_phi = eig_phi.vectors
        lambdas_phi = min.(max.(sigmas_phi,alpha),beta)
        U_ = V_phi*Diagonal(lambdas_phi)*V_phi'
        phi = -log(det(U_))+(Sigma[:]+U_hat[:])'*(U_[:])
        
        dual_gap =  -log(det(Y)) + (Sigma[:]')*Y[:] + rho*sum(abs.(Y))- phi 

        k +=1

        dual_vals = [dual_vals;dual_gap]
        funval = [funval;f(Y)]
        ex = -Inf
        if !extra
            b = restart_strategy(last_restart,funval,Y,eps)
        else 
            (b,ex) = restart_strategy(last_restart,funval,Y,eps)
        end
        if continuation
            b = dual_gap < tol
        end
        if b
            k = 0
            last_restart = i
            X = Array(Y)
            X0 = Array(X)
            restarts = [restarts;i]
            eps *= exp(-gamma)
            #tol *= exp(-gamma)
            #L = M + D2*rho^2/(2*tol)
            sum_grad = zeros(n,n)
            U = max.(min.((X)*rho^2*2*D2/tol,rho),-rho)
        end
        extras = [extras;ex]
        #if dual_gap < tol
        #    break
        #end
    end
    return (Y,dual_vals,funval,restarts,extras)


end

end