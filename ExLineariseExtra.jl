
push!(LOAD_PATH, "/Users/mathieubarre/Documents/These/Restart_Extrapolation/") #replace it with our local path to the folder containing Restart_Algo.jl and restart_strategies.jl
using LinearAlgebra
using Restart_Algo
using restart_strategies
using PyPlot
using CSV
using StatsBase
using DelimitedFiles

function plotf(out::output,fstar,label,color)
    semilogy(1:length(out.funval),out.funval .- fstar,label=label,color=color)
    scatter(out.restart_index,out.funval[out.restart_index].-fstar,color=color)
end


#############################################################################
##                                                                         ##
##                           LEAST SQUARE                                  ##
##                                                                         ##
#############################################################################

# n = 50
# A = rand(50,n)
# A = A'*A + 5e-3*Matrix(I,n,n)
# A = Diagonal(1:100:(200+n*100))
# Sonar Dataset
A = CSV.read("./sonar_zip/data/sonar_csv.csv")
A = convert(Matrix,A)
y = A[:,end]
y = 1*(y.=="Rock")
y = 2*y.-1
A = A[:,1:(end-1)]
A = convert(Matrix{Float64},A)
A = A .-StatsBase.mean(A)
y = convert(Array{Float64},y)

#Madelon Dataset
# A = readdlm("./madelon_train.data")
# y = readdlm("./madelon_train.labels")
# y = y[:]



(m,n) = size(A)
# A = A .-(sum(A,dims=1)/m)
# A = A ./sqrt(sum(A.^2,dims=1))
L = opnorm(A'*A) #+2*0.1*(x_star'*x_star)^(2-1)
mu = eigmin(A'*A) 
x_star = randn(n)
fstar = 1.0
beta_star = (sqrt(L)-sqrt(mu))/(sqrt(L)+sqrt(mu))
#beta_star = -1.0
beta = -1.0
#beta = 0.8*1+0.2*beta_star
max_iter = 10000
gamma = 4.0
#y = randn(50)
fun(x) = begin
    return 0.5*(A*x-y)'*(A*x-y)
    #return 0.5*(x-x_star)'*A*(x-x_star)+fstar #+ 0.1*((x-x_star)'*(x-x_star))^2
end

grad_f(x) = begin
    return A'*(A*x-y)
    #return A*(x- x_star) #+ 0.1*2*((x-x_star)'*(x-x_star))^(2-1)*(x-x_star)
end

null(x) = ()
zero_fun(x) = 0
argminphi_quad(x_0,A,sum_grad) = x_0-sum_grad
f = functional(fun,zero_fun,grad_f,null,argminphi_quad,norm)
x0 =zeros(n)
params = parameters(x0,0.0,1/L,0.0,max_iter,gamma)



rank_eps = 3;
max_length_btw_restart = 10
restart_extra(info,eps) = restart_extra_reborn(info,max_length_btw_restart,eps,rank_eps,0.5,0.0)

params_bis = parameters(x0,0.0,1/L,beta,2*max_iter,gamma)
out = Nesterov_Acc(f,params_bis,restart_mono,false)
out1 = Nesterov_Acc(f,params,no_restart,false)
out3 = Nesterov_Acc(f,params,restart_mono,false)
out4 = Nesterov_Acc(f,params,restart_extra,true)





params_UFG = parameters(x0,max_iter,gamma,L,0.0,100)
out5 = UFastGradient(f,params_UFG,no_restart,false)
out7 = UFastGradient(f,params_UFG,restart_extra,true)
out8 = UFastGradient(f,params_UFG,restart_mono,false)
fstar = minimum([out.funval;out1.funval;out3.funval;out4.funval;out5.funval;out7.funval;out8.funval])
restart_fstar(last,funval,x,eps) = restart_known_fstar(last,funval,x,eps,fstar)

out6 = UFastGradient(f,params_UFG,restart_fstar,false)
out2 = Nesterov_Acc(f,params,restart_fstar,false)



figure()
plotf(out1,fstar,"Nesterov","tab:blue")
plotf(out2,fstar,"Restart f*","tab:orange")
plotf(out3,fstar,"Restart Mono","tab:green")
plotf(out4,fstar,"Restart extra ","tab:purple")
plotf(out5,fstar,"Universal Fast Gradient","tab:red")
plotf(out6,fstar,"Universal Fast Gradient Restart f*","tab:brown")
plotf(out7,fstar,"Universal Fast Gradient Restart Extra","tab:pink")
plotf(out8,fstar,"Universal Fast Gradient Restart Mono","blue")

legend()
semilogy((out7.extrapolation.-fstar),label="extra",color="black")



#############################################################################
##                                                                         ##
##                               LASSO                                     ##
##                                                                         ##
#############################################################################

#n =10
#m = 100
#A = randn(n,m)
max_iter = 10000
x_star = randn(m) 
p = 0.1
x_star[ rand(m) .> p] .= 0
#b = A*x_star + 0.1*randn(n)
lambda =1
gamma = 1.0
function lasso(x,lambda)
    return lambda*sum(abs.(x))
end

function prox_lasso(x,lambda)
    return sign.(x).*max.(abs.(x) .- lambda,0)
end
f_(x) = 0.5*(A*x-y)'*(A*x-y)
g(x) = lasso(x,lambda)
grad_f_lasso(x) = A'*(A*x-y)

L = opnorm(A'*A)
prox_g(x) = prox_lasso(x,lambda/L)
fun_lasso = functional(f_,g,grad_f_lasso,prox_g,null,norm)
x0 = ones(n)
params_fista = parameters(x0,0.0,1/L,0.0,max_iter,gamma)

rank_eps = 3;
max_length_btw_restart = 1
restart_extra(last,funval,x,eps) = restart_extra_eps_algo(Int64(last),funval,x,eps,rank_eps,max_length_btw_restart)

params_max = parameters(x0,0.0,1/L,0.0,2*max_iter,gamma)
out = Fista(fun_lasso,params_max,restart_mono,false)
out1 = Fista(fun_lasso,params_fista,no_restart,false)
out4 = Fista(fun_lasso,params_fista,restart_extra,true)
out3 = Fista(fun_lasso,params_fista,restart_mono,false)

argmin_phi_lasso(x,A,grad)= prox_lasso(x-grad,lambda*A)
fun_lasso_UFG=functional(f_,g,grad_f_lasso,null,argmin_phi_lasso,norm)
params_lasso_UFG = parameters(x0,max_iter,gamma,L,0.0,100)
out5 = UFastGradient(fun_lasso_UFG,params_lasso_UFG,no_restart,false)
out7 = UFastGradient(fun_lasso_UFG,params_lasso_UFG,restart_extra,true)
out8 = UFastGradient(fun_lasso_UFG,params_lasso_UFG,restart_mono,false)

fstar = minimum([out.funval;out1.funval;out3.funval;out4.funval;out5.funval;out7.funval;out8.funval])
restart_fstar(last,funval,x,eps) = restart_known_fstar(last,funval,x,eps,fstar)
out2 = Fista(fun_lasso,params_fista,restart_fstar,false)
out6 = UFastGradient(fun_lasso_UFG,params_lasso_UFG,restart_fstar,false)



figure()
plotf(out1,fstar,"Nesterov","tab:blue")
plotf(out2,fstar,"Restart f*","tab:orange")
plotf(out3,fstar,"Restart Mono","tab:green")
plotf(out4,fstar,"Restart extra ","tab:purple")
plotf(out5,fstar,"UFG ","tab:red")
plotf(out6,fstar,"UFG f* ","tab:brown")
plotf(out7,fstar,"UFG Extra ","tab:pink")
plotf(out8,fstar,"Universal Fast Gradient Restart Mono","blue")
legend()

#############################################################################
##                                                                         ##
##                               SVM                                       ##
##                                                                         ##
#############################################################################

svm(x) = 0.5*(A'*Diagonal(y)*x)'*(A'*Diagonal(y)*x) - sum(x)
grad_smooth_svm(x) = Diagonal(y)*A*(A'*Diagonal(y)*x).-1
prox_box(x) = min.(1,max.(0,x))
fun_svm = functional(svm,zero_fun,grad_smooth_svm,prox_box,null,norm)
argmin_phi_svm(x,A,grad) = prox_box(x-grad)
fun_svm_UFG = functional(svm,zero_fun,grad_smooth_svm,null,argmin_phi_svm,norm)
x0 = ones(m)
gamma = 2.0
max_iter = 10000
params_svm = parameters(x0,0.0,1/L,0.0,max_iter,gamma)

rank_eps = 3;

params_max = parameters(x0,0.0,1/L,0.0,2*max_iter,gamma)
out = Fista(fun_svm,params_max,restart_mono,false)
out1 = Fista(fun_svm,params_svm,no_restart,false)
out4 = Fista(fun_svm,params_svm,restart_extra,true)
out3 = Fista(fun_svm,params_svm,restart_mono,false)

params_svm_UFG = parameters(x0,max_iter,gamma,L,0.0,100)
out5 = UFastGradient(fun_svm_UFG,params_svm_UFG,no_restart,false)
out7 = UFastGradient(fun_svm_UFG,params_svm_UFG,restart_extra,true)
out8 = UFastGradient(fun_svm_UFG,params_svm_UFG,restart_mono,false)


fstar = minimum([out.funval;out1.funval;out3.funval;out4.funval;out5.funval;out6.funval;out8.funval])

restart_fstar(last,funval,x,eps) = restart_known_fstar(last,funval,x,eps,fstar)
out2 = Fista(fun_svm,params_svm,restart_fstar,false)
out6 = UFastGradient(fun_svm_UFG,params_svm_UFG,restart_fstar,false)
figure()
plotf(out1,fstar,"Nesterov","tab:blue")
plotf(out2,fstar,"Restart f*","tab:orange")
plotf(out3,fstar,"Restart Mono","tab:green")
plotf(out4,fstar,"Restart extra ","tab:purple")
plotf(out5,fstar,"UFG","tab:red")
plotf(out7,fstar,"UFG Extra ","tab:pink")
plotf(out6,fstar,"UFG f* ","tab:brown")
plotf(out8,fstar,"UFG Mono ","blue")
legend()
show()

