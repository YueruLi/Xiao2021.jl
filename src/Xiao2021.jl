module Xiao2021
using MAT, Dierckx, DelimitedFiles
import Base.NamedTuple
#Load in the init values as named-tuples
function NamedTuple(d::Dict{String, T} where T)
    NamedTuple(Symbol(k) => v for (k,v) in d)
end
include("solve_eqm.jl")
include("stages_PL.jl")
#init
function getInit()
    init0 = matread("init0.mat")["init"];
    init = NamedTuple(init0);
    return init
end
#Settings 
function getSettings()
    set = (seed = 29111998, T = 12*35, N = 10000, Nx =7, Ne =3, Ny = 7, Na = 3);
    return set
end;

#Fixed parameters
function getFixParameters()
    set = getSettings();
    gamma_ya = zeros(7,3);
    gamma_ya[:,1] = [.069078 .1300653 .1109684 .1536212 .1469259 .0854673 .0538162];
    gamma_ya[:,2] = [.0134691 .0236159 .0504847 .0249374 .0186719 .0133951 .00548];
    gamma_ya[:,3] = [.0430964 .0179972 .0141437 .0130393 .0067923 .0035348 .0014];

    #from k-means clustering 
    pKmeans = vec(exp.([2.6427255, 2.9569508, 3.100542, 3.2401642, 3.3862326, 3.5540491, 3.8309034]));

    fxp = (beta = 0.988,
        lambda = 0.5,
        gamma   = 1.0/(12*20),
        chi = 1/(12*8),
        gamma2 =1.0/(12*20),
        etaf = 1/(12*3),
        etam = 1/1.05,
        phi = 1/(12*20),
        alpha = [0, 1, 2],
        eden = [0.1587, 0.6826, 0.1587],
        sdf = 0.2,
        sdm = 0.2,
        gamma_yα = vec(gamma_ya),
        p = pKmeans ./pKmeans[1],
        );

    #initial HC distribution from initial wage distribution
    dist0    = readdlm("init_dist0.csv", ',',header=true)[1];
    xm = dist0[:,1]./pKmeans[1];
    xf = dist0[:,2]./pKmeans[1];
    Gm = dist0[:,3];
    Gf = dist0[:,4];
    #males
    hc = LinRange(fxp.p[1],fxp.p[set.Ny],set.Ny);
    cdfm = Spline1D(xm, Gm; k=1)(hc);
    replace!(cdfm, missing=>1);
    dm = diff(cdfm);
    #females
    cdff = Spline1D(xf, Gf; k=1)(hc);
    replace!(cdff, missing=>1);
    df = diff(cdff);
    tempFxp = Dict("hc"=>hc,"phi0m" =>[cdfm[1]; dm],"phi0f" =>[cdff[1]; df],
                "c" => 0.0,
                "Transfer" => 0.75,
                "theta" => 0.107140016964402,
                "s1" =>0.71875,
                "s2" =>0.53125,
                "deltaf_NC"=> 0.012,
                "deltam_NC" => 0.0075,
                "deltaf_YC" => 0.016,
                "deltam_YC" => 0.0075,
                "delta" => 0.0075,
                "rho" =>-14.6947062177210,
                "mum" =>0.757706936476000,
                "d3" => 0)

    fxp = merge(fxp,NamedTuple(tempFxp))
    return fxp
end;
function getParams()
    parameters = Dict("a"=>0.956548300980000,"K"=>22.2322078018640,
                "M" => 2.02378547792400,
                "muf" =>1.07630516756000,
                "sigma" =>0.436340820205000,
                "d1m" => 0.00107456051900000,
                "d2m" => 0.0116103577370000,
                "d1f" => 0.00107456051900000,
                "d2f" => 0.0116103577370000,
                "b" => 7.79173908202000);
    par = NamedTuple(parameters)
    return par
end

export getSettings, getFixParameters,getParams, getInit, solve_eqm
end