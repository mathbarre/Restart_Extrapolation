
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
#beta = -1.0
beta = 0.8*1+0.2*beta_star
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
restart_extra(last,funval,x,eps) = restart_extra_eps_algo(Int64(last),funval,x,eps,rank_eps,max_length_btw_restart)

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


#############################################################################
##                                                                         ##
##                            LOGISTIC                                     ##
##                                                                         ##
#############################################################################

lambda = 0.0001
fun_logit(x) = sum(log.(1 .+ exp.(-y.*(A*x))))+0.5*lambda*x'*x
grad_logit(x) = (sum(-(y.*A)'.*(exp.(-y.*(A*x))./(1 .+ exp.(-y.*(A*x))))',dims=2)')[:] +lambda*x
argmin_phi_logit(x,A,grad) = x-grad
fun_logit_UFG=functional(fun_logit,zero_fun,grad_logit,null,argmin_phi_logit,norm)
rank_eps = 1;
max_length_btw_restart = 1
restart_extra(last,funval,x,eps) = restart_extra_eps_algo(Int64(last),funval,x,eps,rank_eps,max_length_btw_restart)
x0 = zeros(n)
gamma = 5.0
max_iter = 20000
beta= -1.0
L = opnorm(A'*A)+lambda
param_logit_UFG = parameters(x0,max_iter,gamma,L,0.0,100)
params_bis = parameters(x0,0.0,1/L,beta,2*max_iter,gamma)
params = parameters(x0,0.0,1/L,beta,max_iter,gamma)
out = Nesterov_Acc(fun_logit_UFG,params_bis,restart_mono,false)
out1 = Nesterov_Acc(fun_logit_UFG,params,no_restart,false)
out2 = Nesterov_Acc(fun_logit_UFG,params,restart_mono,false)
out4 = Nesterov_Acc(fun_logit_UFG,params,restart_extra,true)
out5 =UFastGradient(fun_logit_UFG,param_logit_UFG,no_restart,false)
out6 =UFastGradient(fun_logit_UFG,param_logit_UFG,restart_mono,false)
out7 =UFastGradient(fun_logit_UFG,param_logit_UFG,restart_extra,true)
fstar = minimum([out.funval;out1.funval;out2.funval;out5.funval;out4.funval;out6.funval;out7.funval])



restart_fstar(last,funval,x,eps) = restart_known_fstar(last,funval,x,eps,fstar)
out3 = Nesterov_Acc(fun_logit_UFG,params,restart_fstar,false)
out8 =UFastGradient(fun_logit_UFG,param_logit_UFG,restart_fstar,false)
figure()
plotf(out1,fstar,"Nesterov","tab:blue")
plotf(out2,fstar,"Nesterov Mono","tab:green")
plotf(out3,fstar,"Nesterov f*","tab:orange")
plotf(out4,fstar,"Nesterov Extra","tab:purple")
plotf(out5,fstar,"UFG","tab:red")
plotf(out6,fstar,"UFG Mono","blue")
plotf(out7,fstar,"UFG Extra","tab:pink")
plotf(out8,fstar,"UFG f*","tab:brown")
legend()


#############################################################################
##                                                                         ##
##                            MATRIX COMPLETION                            ##
##                                                                         ##
#############################################################################

n = 50
m = 50
r = 35

# creates a low-rank random matrix
A = randn(m,n)
B = svd(A)
S = B.S
S[(r+1):end] .= 0

A = B.U*Diagonal(S)*B.V'

lambda = 10e-2
nb_obs = 1000
idx_obs = sample(1:(n*m),nb_obs)

fun_mtx(X) = 0.5*sum((X[idx_obs]-A[idx_obs]).^2) 
grad_mtx(X) = begin
    Res = zeros(size(X,1),size(X,2))
    Res[idx_obs] = (X[idx_obs]-A[idx_obs])
    return Res
end

tr_norm(X) = begin
    F = svd(X)
    return lambda*sum(abs.(F.S))
end

prox_tr(X,l) = begin
    F = svd(X)
    S = F.S
    S = max.(0.0,S .- lambda*l)
    return F.U*Diagonal(S)*F.V'
end

argmin_tr(X,a,grad) = prox_tr(X-grad,a*lambda)
normfro(X) = norm(X[:])

x0 = randn(m,n)

no_restart(last,funval,x,eps) = false

rank_eps = 3
max_length_btw_restart = 1
max_iter = 2000;
gamma = 3.0
L = 2.0
beta = -1.0
fun_mtx_UFG = functional(fun_mtx,tr_norm,grad_mtx,prox_tr,argmin_tr,normfro)
restart_extra(last,funval,x,eps) = restart_extra_eps_algo(Int64(last),funval,x,eps,rank_eps,max_length_btw_restart)
params = parameters(x0,0.0,1/L,beta,max_iter,gamma)
params_max = parameters(x0,0.0,1/L,beta,10*max_iter,gamma)
out = AccGradientMtx(fun_mtx_UFG,params_max,no_restart,false)
out_ = AccGradientMtx(fun_mtx_UFG,params,no_restart,false)
out__ = AccGradientMtx(fun_mtx_UFG,params,restart_mono,false)
out___= AccGradientMtx(fun_mtx_UFG,params,restart_extra,true)
params_mtx_UFG_max = parameters(x0,2*max_iter,gamma,L,0.0,100)
#out = UFastGradient(fun_mtx_UFG,params_mtx_UFG_max,no_restart)
fstar= minimum([out.funval;out_.funval;out__.funval;out__.funval])
params_mtx_UFG = parameters(x0,max_iter,gamma,L,0.0,100)
#out1 = UFastGradient(fun_mtx_UFG,params_mtx_UFG,no_restart,false)
#out2 = UFastGradient(fun_mtx_UFG,params_mtx_UFG,restart_extra,true)
#out3 = UFastGradient(fun_mtx_UFG,params_mtx_UFG,restart_mono,false)

#fstar= minimum([out.funval;out2.funval;out3.funval;out_.funval;out__.funval])
restart_fstar(last,funval,x,eps) = restart_known_fstar(last,funval,x,eps,fstar)
#out4 = UFastGradient(fun_mtx_UFG,params_mtx_UFG,restart_fstar,false)
out____= AccGradientMtx(fun_mtx_UFG,params,restart_fstar,false)
figure()
#plotf(out1,fstar,"UFG","tab:blue")
#plotf(out2,fstar,"UFG Extra","tab:purple")
#plotf(out3,fstar,"UFG Mono","tab:green")
#plotf(out4,fstar,"UFG f*","tab:orange")
plotf(out_,fstar,"Acc","tab:red")
plotf(out__,fstar,"Acc Mono","tab:brown")
plotf(out___,fstar,"Acc Extra","tab:pink")
plotf(out____,fstar,"Acc Extra","tab:orange")
legend()

#############################################################################
##                                                                         ##
##                         COVARIANCE SELECTION                            ##
##                                                                         ##
#############################################################################

n = 30
A = 1.0*Matrix(I,n,n)
nb_obs = 12
idx_obs = sample(1:(n*n),nb_obs)
A[idx_obs] = sign.(4*rand(nb_obs).-1)
A = A'+A

sigma = 0.2
V = rand(n,n)
B = inv(A) + sigma*(V+V')/2

rho = 0.2

alpha = eigen(A).values[1]
beta = eigen(A).values[end]
tol = 1e-3
max_iter = 20000

rank_eps = 3
max_length_btw_restart = 1
gamma = 2.0
no_restart(last,funval,x,eps) = false
restart_extra(last,funval,x,eps) = restart_extra_eps_algo(Int64(last),funval,x,eps,rank_eps,max_length_btw_restart)
restart_const(last,funval,x,eps) = (mod(length(funval),100) == 0)

(X,duals,funval,r,e) = CovSelectOtherProx(tol,0.5*alpha,2*beta,rho,B,max_iter,no_restart,false,gamma,false)
(X_,duals_,funval_,r_,e_) = CovSelectOtherProx(tol,0.5*alpha,2*beta,rho,B,max_iter,restart_extra,true,gamma,false)
(X__,duals__,funval__,r__,e__) = CovSelectOtherProx(tol,0.5*alpha,2*beta,rho,B,max_iter,restart_mono,false,gamma,false)
fstar = minimum([funval;funval_;funval__])
restart_fstar(last,funval,x,eps) = restart_known_fstar(last,funval,x,eps,fstar)
(X___,duals___,funval___,r___,e___) = CovSelectOtherProx(tol,0.5*alpha,2*beta,rho,B,max_iter,restart_fstar,false,gamma,false)

#thresh(X) = min.(max.((abs.(X) .> 1e0) .* X,-2.0),4.0) 
thresh(X) = X
figure()
imshow(A,cmap="Greys")
figure()
imshow(thresh(inv(B)),cmap="Greys")
figure()
imshow(thresh(X),cmap="Greys")
figure()
imshow(thresh(X_),cmap="Greys")



figure()
semilogy(funval.-fstar,label="no restart")
semilogy(funval_.-fstar,label="restart extra")
scatter(r_,funval_[r_].-fstar,color="tab:orange")
semilogy(funval__.-fstar,label = "restart mono")
scatter(r__,funval__[r__].-fstar,color="tab:green")
semilogy(funval___.-fstar,label = "restart f*")
scatter(r___,funval___[r___].-fstar,color="tab:red")
semilogy(abs.(e_.-fstar))
legend()

figure()
semilogy(duals,label="no restart")
semilogy(duals_,label="restart extra")
scatter(r_,duals_[r_],color="tab:orange")
semilogy(duals__,label = "restart mono")
scatter(r__,duals__[r__],color="tab:green")
semilogy(duals___,label="restart f*")
scatter(r___,duals___[r___],color="tab:red")
