module solve_eqm
# Write your package code here.
using NLsolve
include("solve_dist.jl")
include("valueFunctions.jl")
include("helper.jl")
using .solve_dist, .valueFunctions, .helper
function logit(x, fixedParam)
    return 1 ./ (1.0 .+ exp.(-fixedParam.lambda .* x))
end
#Solve for equilibrium
function solve(par, set, fxp, init)
    kk = set.Ny * set.Na;
    ii = set.Nx * set.Ne;

    epsf = repeat([-2, 0, 2] .* fxp.sdf .+ par.muf * fxp.mum, inner = 7)
    epsm = repeat([-2, 0, 2] .* fxp.sdm .+ fxp.mum, inner = 7)
    alpha = repeat(fxp.alpha,inner=7);
    #humanCapital = repeat(fxp.hc, outer = 3);
    #jobProductivityType = repeat(fxp.p,outer=3);
    #jobAmenityType = repeat(fxp.alpha,inner=7);
    output = par.K .* [(par.a * hc^fxp.rho + (1 - par.a) * p^fxp.rho)^(1 / fxp.rho) for hc in repeat(fxp.hc, outer = 3), p in repeat(fxp.p, outer = 3)]
    outputb = repeat([par.b * hc for hc in fxp.hc], outer = 3)
    #This stages solution from nlsolve is slightly different from that from matlab's fsolve. The difference is on the order of 1e-6
    e = Dict("stages" => nlsolve(x -> solveStages(x, fxp), 12.50 .* ones(typeof(12.50), 8)).zero, "converge" => 1, "Pineg" => 0)
    #stages = [0.0714286494397354,0.117096067390252,0.0614754501329449,0.249999769056968,0.0714293405864623,0.175877941575804,0.00269313154484208,0.250000013739020];
    #e = Dict("stages" => stages, "converge" => 1, "Pineg" => 0)
    NC_f = e["stages"][1]
    YC_f = e["stages"][2]
    PL_f = e["stages"][3]
    D_f = e["stages"][4]
    NC_m = e["stages"][5]
    YC_m = e["stages"][6]
    PL_m = e["stages"][7]
    D_m = e["stages"][8]
    gem = par.d1m .+ par.d2m .* fxp.p
    gef = par.d1f .+ par.d2f .* fxp.p


    #unemployed workers u(x), in different stages
    umNC_0 = vec(init.umNC)
    umPL_0 = vec(init.umYC / 2)
    umYC_0 = vec(init.umYC / 2)
    umD_0 = vec(init.umD)
    ufNC_0 = vec(init.ufNC)
    ufPL_0 = vec(init.ufYC / 2)
    ufYC_0 = vec(init.ufYC / 2)
    ufD_0 = vec(init.ufD)
    #employed workers h(x, y, fxp.alpha)
    hmNC_0 = init.hmNC
    hmYC_0 = init.hmYC
    hmPL_0 = init.hmPL
    hmD_0 = init.hmD
    hfNC_0 = init.hfNC
    hfYC_0 = init.hfYC
    hfPL_0 = init.hfPL
    hfD_0 = init.hfD
    #vacancies
    v0 = vec(init.v)
    #Guess Value Functions
    # value in unemployment
    UmNC_0 = vec(init.UmNC)
    UmPL_0 = vec(init.UmYC)
    UmYC_0 = vec(init.UmYC)
    UmD_0 = vec(init.UmD)
    UfNC_0 = vec(init.UfNC)
    UfPL_0 = vec(init.UfYC)
    UfYC_0 = vec(init.UfYC)
    UfD_0 = vec(init.UfD)
    #value in vacancy Π₀(y,α)
    Pi0 = vec(init.Pi)

    #value in surplus
    #Male NC
    SmNC = init.SmNC
    SmNCPos = copy(SmNC)
    SmNCPos[SmNCPos.<0] .= 0
    #Male PL
    SmPL = init.SmPL
    SmPLPos = copy(SmPL)
    SmPLPos[SmPLPos.<0] .= 0
    #Male YC
    SmYC = init.SmYC
    SmYCPos = copy(SmYC)
    SmYCPos[SmYCPos.<0] .= 0
    #Male D
    SmD = init.SmD
    SmDPos = copy(SmD)
    SmDPos[SmDPos.<0] .= 0

    #Female NC
    SfNC = init.SfNC
    SfNCPos = copy(SfNC)
    SfNCPos[SfNCPos.<0] .= 0
    #Female PL
    SfPL = init.SfPL
    SfPLPos = copy(SfPL)
    SfPLPos[SmPLPos.<0] .= 0
    #Female YC
    SfYC = init.SfYC
    SfYCPos = copy(SfYC)
    SfYCPos[SfYCPos.<0] .= 0
    #Female D
    SfD = init.SfD
    SfDPos = copy(SfD)
    SfDPos[SfDPos.<0] .= 0

    #Probably don't need this 
    PmNC_0 = SmNC + [x + y for x in UmNC_0, y in Pi0]
    PmPL_0 = SmPL + [x + y for x in UmPL_0, y in Pi0]
    PmYC_0 = SmYC + [x + y for x in UmYC_0, y in Pi0]
    PmD_0 = SmD + [x + y for x in UmD_0, y in Pi0]

    PfNC_0 = SfNC + [x + y for x in UfNC_0, y in Pi0]
    PfPL_0 = SfPL + [x + y for x in UfPL_0, y in Pi0]
    PfYC_0 = SfYC + [x + y for x in UfYC_0, y in Pi0]
    PfD_0 = SfD + [x + y for x in UfD_0, y in Pi0]

    tau = fxp.Transfer * sum(sum(hmPL_0 + hfPL_0, dims = 1) .* output, dims = 1) / (sum(sum(hmNC_0 + hfNC_0 + hmYC_0 + hfYC_0 + hmD_0 + hfD_0, dims = 1) .* output, dims = 1))
    PmNC = zeros(Float64, 21, 21)
    PmPL = zeros(Float64, 21, 21)
    PmYC = zeros(Float64, 21, 21)
    PmD = zeros(Float64, 21, 21)

    PfNC = zeros(Float64, 21, 21)
    PfPL = zeros(Float64, 21, 21)
    PfYC = zeros(Float64, 21, 21)
    PfD = zeros(Float64, 21, 21)

    UmNC = zeros(Float64, 21)
    UmPL = zeros(Float64, 21)
    UmYC = zeros(Float64, 21)
    UmD = zeros(Float64, 21)
    UfNC = zeros(Float64, 21)
    UfPL = zeros(Float64, 21)
    UfYC = zeros(Float64, 21)
    UfD = zeros(Float64, 21)
    
    maxit = 10000
    change_dist = zeros(maxit, 1)
    tol = 1e-9
    for itout = 1:maxit
        effU = sum(umNC_0 + ufNC_0 + fxp.s1 * (umYC_0 + umD_0 + ufYC_0 + ufD_0)) +
           fxp.s2 * sum(sum(hmNC_0 + hmYC_0 + hmD_0 + hfNC_0 + hfYC_0 + hfD_0, dims = 1));
        V = sum(v0)
        fxp = merge(fxp, (lambdau_NC = fxp.theta / (effU * V)^0.5, lambdau_YC = fxp.theta / (effU * V)^0.5 * fxp.s1, lambdae = fxp.theta / (effU * V)^0.5 * fxp.s2));
        Pi = zeros(Float64, 21);
        for it1 = 1:maxit
            #Value of Unemployment
            #Male NC
            UmNC = [x + fxp.beta * (fxp.gamma * y + fxp.chi * z +
                                    fxp.lambdau_NC * par.sigma * sum(v0 .* SmNCPos[i, :]) +
                                    (1 - fxp.gamma - fxp.chi) * m)
                    for (x, y, z, m, i) in zip(outputb, UmD_0, UmPL_0, UmNC_0, 1:21)]
    
            #Male PL
            UmPL = [x + fxp.beta * (fxp.gamma * y + fxp.etam * z +
                                    (1 - fxp.gamma - fxp.etam) * m)
                    for (x, y, z, m) in zip(outputb, UmD_0, UmYC_0, UmPL_0)]
    
            #Male YC
            UmYC = [x + fxp.beta * (fxp.gamma * y + fxp.chi * z +
                                    fxp.lambdau_YC * par.sigma * sum(v0 .* SmYCPos[i, :]) +
                                    (1 - fxp.gamma - fxp.chi) * m)
                    for (x, y, z, m, i) in zip(outputb, UmD_0, UmPL_0, UmYC_0, 1:21)]
    
            #Male D 
            UmD = [x + fxp.beta * ((1 - fxp.phi) * y +
                                   fxp.lambdau_YC * par.sigma * sum(v0 .* SmDPos[i, :]))
                   for (x, y, i) in zip(outputb, UmD_0, 1:21)]
    
            #Female NC
            UfNC = [x + fxp.beta * (fxp.gamma * y + fxp.chi * z +
                                    fxp.lambdau_NC * par.sigma * sum(v0 .* SfNCPos[i, :]) +
                                    (1 - fxp.gamma - fxp.chi) * m)
                    for (x, y, z, m, i) in zip(outputb, UfD_0, UfPL_0, UfNC_0, 1:21)]
    
            #Female PL
            UfPL = [x + fxp.beta * (fxp.gamma * y + fxp.etaf * z +
                                    (1 - fxp.gamma - fxp.etaf) * m)
                    for (x, y, z, m) in zip(outputb, UfD_0, UfYC_0, UfPL_0)]
            #Female YC
            UfYC = [x + fxp.beta * (fxp.gamma * y + fxp.chi * z +
                                    fxp.lambdau_YC * par.sigma * sum(v0 .* SfYCPos[i, :]) +
                                    (1 - fxp.gamma - fxp.chi) * m)
                    for (x, y, z, m, i) in zip(outputb, UfD_0, UfPL_0, UfYC_0, 1:21)]
            #Female D
            UfD = [x + fxp.beta * ((1 - fxp.phi) * y +
                                   fxp.lambdau_YC * par.sigma * sum(v0 .* SfDPos[i, :]))
                   for (x, y, i) in zip(outputb, UfD_0, 1:21)]
    
            Pi = zeros(Float64, 21)
            for j = 1:kk
                dmNC = -1 .* SmNC .+ SmNC[:, j]
                dmYC = -1 .* SmYC .+ SmYC[:, j]
                dmD = -1 .* SmD .+ SmD[:, j]
    
                dfNC = -1 .* SfNC .+ SfNC[:, j]
                dfYC = -1 .* SfYC .+ SfYC[:, j]
                dfD = -1 .* SfD .+ SfD[:, j]
    
                #Vacancy Value Calculation
                Pi[j] = -fxp.c + fxp.beta * (
                    fxp.lambdau_NC * (1 - par.sigma) * sum(umNC_0 .* SmNCPos[:, j]) +
                    fxp.lambdau_YC * (1 - par.sigma) * (sum(umYC_0 .* SmYCPos[:, j]) + sum(umD_0 .* SmDPos[:, j])) +
                    fxp.lambdau_NC * (1 - par.sigma) * sum(ufNC_0 .* SfNCPos[:, j]) +
                    fxp.lambdau_YC * (1 - par.sigma) * (sum(ufYC_0 .* SfYCPos[:, j]) + sum(ufD_0 .* SfDPos[:, j])) +
                    fxp.lambdae * (1 - par.sigma) * (
                        sum(hmNC_0 .* dmNC .* logit(dmNC, fxp)) +
                        sum(hmYC_0 .* dmYC .* logit(dmYC, fxp)) +
                        sum(hmD_0 .* dmD .* logit(dmD, fxp)) +
                        sum(hfNC_0 .* dfNC .* logit(dfNC, fxp)) +
                        sum(hfYC_0 .* dfYC .* logit(dfYC, fxp)) +
                        sum(hfD_0 .* dfD .* logit(dfD, fxp))
                    )
                    + Pi0[j]
                )
            end
    
            # joint value P
            PmNC_hat = PmNC_0 .* (SmNC .> 0) + [x + y for x in UmNC, y in Pi] .* (SmNC .< 0)
            PmPL_hat = PmPL_0 .* (SmPL .> 0) + [x + y for x in UmPL, y in Pi] .* (SmPL .< 0)
            PmYC_hat = PmYC_0 .* (SmYC .> 0) + [x + y for x in UmYC, y in Pi] .* (SmYC .< 0)
            PmD_hat = PmD_0 .* (SmD .> 0) + [x + y for x in UmD, y in Pi] .* (SmD .< 0)
    
            PfNC_hat = PfNC_0 .* (SfNC .> 0) + [x + y for x in UfNC, y in Pi] .* (SfNC .< 0)
            PfPL_hat = PfPL_0 .* (SfPL .> 0) + [x + y for x in UfPL, y in Pi] .* (SfPL .< 0)
            PfYC_hat = PfYC_0 .* (SfYC .> 0) + [x + y for x in UfYC, y in Pi] .* (SfYC .< 0)
            PfD_hat = PfD_0 .* (SfD .> 0) + [x + y for x in UfD, y in Pi] .* (SfD .< 0)
            for i in 1:ii
                for j in 1:kk
                    PmNC[i, j] = getPmNC(i,j,v0,fxp,par,Pi,gem,SmNC,UmNC,PmD_hat,PmPL_hat,PmNC_hat,epsm,alpha,output,tau)
                    PmPL[i, j] = getPmPL(i,j,Pi,fxp,UmPL,PmD_hat,PmYC_hat,PmPL_hat,epsm,alpha,output)
                    PmYC[i, j] = getPmYC(i,j,v0,Pi,fxp,par,epsm,gem,tau,output,alpha,UmYC,SmYC,PmD_hat,PmPL_hat,PmYC_hat)
                    PmD[i, j] = getPmD(i,j,v0,Pi,fxp,par,output,gem,tau,epsm,alpha,SmD,PmD_hat,UmD)
    
                    PfNC[i, j] = getPfNC(i,j,v0,fxp,par,Pi,gef,SfNC,UfNC,PfD_hat,PfPL_hat,PfNC_hat,epsf,alpha,output,tau)
                    PfPL[i, j] = getPfPL(i,j,Pi,par,fxp,UfPL,PfD_hat,PfYC_hat,PfPL_hat,epsf,alpha,output)
                    PfYC[i, j] = getPfYC(i,j,v0,Pi,fxp,par,epsf,gef,tau,output,alpha,UfYC,SfYC,PfD_hat,PfPL_hat,PfYC_hat)
                    PfD[i, j] = getPfD(i,j,v0,Pi,fxp,par,output,gef,tau,epsf,alpha,SfD,PfD_hat,UfD)
                end
            end
    
            SmNC_0 = PmNC - [x + y for x in UmNC, y in Pi]
            SmPL_0 = PmPL - [x + y for x in UmPL, y in Pi]
            SmYC_0 = PmYC - [x + y for x in UmYC, y in Pi]
            SmD_0 = PmD - [x + y for x in UmD, y in Pi]
    
            SfNC_0 = PfNC - [x + y for x in UfNC, y in Pi]
            SfPL_0 = PfPL - [x + y for x in UfPL, y in Pi]
            SfYC_0 = PfYC - [x + y for x in UfYC, y in Pi]
            SfD_0 = PfD - [x + y for x in UfD, y in Pi]
    
            change1 = sum((SmNC - SmNC_0) .^ 2 + (SmYC - SmYC_0) .^ 2 + (SmPL - SmPL_0) .^ 2 + (SmD - SmD_0) .^ 2 + (SfNC - SfNC_0) .^ 2 + (SfYC - SfYC_0) .^ 2 + (SfPL - SfPL_0) .^ 2 + (SfD - SfD_0) .^ 2)
    
            if change1 < tol
                println("converged in " * string(it1) * " iterations, with change:" * string(change1))
                break
            end
            UmNC_0 = UmNC
            UmPL_0 = UmPL
            UmYC_0 = UmYC
            UmD_0 = UmD
            UfNC_0 = UfNC
            UfPL_0 = UfPL
            UfYC_0 = UfYC
            UfD_0 = UfD
    
            Pi0 = Pi
    
            PmNC_0 = PmNC
            PmYC_0 = PmYC
            PmPL_0 = PmPL
            PmD_0 = PmD
            PfNC_0 = PfNC
            PfYC_0 = PfYC
            PfPL_0 = PfPL
            PfD_0 = PfD
    
            SmNC = SmNC_0
            SmNCPos = copy(SmNC_0)
            SmNCPos[SmNCPos.<0] .= 0
    
            SmYC = SmYC_0
            SmYCPos = copy(SmYC)
            SmYCPos[SmYCPos.<0] .= 0
    
            SmPL = SmPL_0
            SmPLPos = copy(SmPL)
            SmPLPos[SmPLPos.<0] .= 0
    
            SmD = SmD_0
            SmDPos = copy(SmD)
            SmDPos[SmDPos.<0] .= 0
    
            SfNC = SfNC_0
            SfNCPos = copy(SfNC)
            SfNCPos[SfNCPos.<0] .= 0
    
            SfYC = SfYC_0
            SfYCPos = copy(SfYC)
            SfYCPos[SfYCPos.<0] .= 0
    
            SfPL = SfPL_0
            SmPLPos = copy(SmPL)
            SmPLPos[SmPLPos.<0] .= 0
    
            SfD = SfD_0
            SfDPos = copy(SfD)
            SfDPos[SfDPos.<0] .= 0
        end
        #return e;
        x0 = vcat(umNC_0, umPL_0, umYC_0, umD_0, ufNC_0, ufPL_0, ufYC_0, ufD_0, vec(hmNC_0), vec(hmYC_0), vec(hmPL_0), vec(hmD_0),vec(hfNC_0), vec(hfYC_0), vec(hfPL_0), vec(hfD_0),vec(v0));
        x = zeros(Float64,length(x0));
        x_orig = x0;
        for it2 = 1:maxit
            x = solveDist(x0, set, fxp, par, SmNC, SmPL, SmYC, SmD, SfNC, SfPL, SfYC, SfD, e["stages"]);
            x[abs.(x) .< 1e-30] .= 0;
            dev = (x-x0)./x0;
            replace!(dev, Inf=>NaN);
            replace!(dev, NaN => 0)
            change2  = sum(dev.^2)
            if (change2 < tol)
                println("converged in " * string(it2) * " iterations, with change:" * string(change2));
                break;
            end  
            x0 = x;
        end
        #If this fixed point in distributions is different from the initial
        #distribution, then repeat the who process again
        dev                 = (x-x_orig)./x_orig;        #percentage deviation
        replace!(dev, Inf=>NaN);
        replace!(dev, NaN => 0)
        change_dist[itout]  = sum(dev.^2);
    
        #Print Progress
        println("=================="*"Outer Iteration " *string(itout)*" with change "*string(change_dist[itout])*"==================");
        #Update Solution for distributions
        v0 = x[length(x)-set.Ny*set.Na+1:length(x)];
        #delete v0 from x
        x = deleteat!(x, length(x)-set.Ny*set.Na+1:length(x));
        x = reshape(x,set.Nx*set.Ne,:);
    
        umNC_0    = x[:, 1];
        umPL_0    = x[:,2];
        umYC_0    = x[:,3];
        umD_0     = x[:,4];
        ufNC_0    = x[:,5];
        ufPL_0    = x[:,6];
        ufYC_0    = x[:,7];
        ufD_0     = x[:,8];
        hmNC_0    = x[:,9:29];
        hmYC_0    = x[:,30:50];
        hmPL_0    = x[:,51:71];
        hmD_0     = x[:,72:92];
        hfNC_0    = x[:,93:113];
        hfYC_0    = x[:,114:134];
        hfPL_0    = x[:,135:155];
        hfD_0     = x[:,156:176];
    
        #output distribution 
        get!(e, "hmNC", hmNC_0);
        get!(e, "hmPL", hmPL_0);
        get!(e, "hmYC", hmYC_0);
        get!(e, "hmD", hmD_0);
        get!(e, "hfNC", hfNC_0);
        get!(e, "hfPL", hfPL_0);
        get!(e, "hfYC", hfYC_0);
        get!(e, "hfD", hfD_0);
    
        get!(e, "umNC", umNC_0);
        get!(e, "umPL", umPL_0);
        get!(e, "umYC", umYC_0);
        get!(e, "umD", umD_0);
        get!(e, "ufNC", ufNC_0);
        get!(e, "ufPL", ufPL_0);
        get!(e, "ufYC", ufYC_0);
        get!(e, "ufD", ufD_0);
    
        get!(e, "v", v0);
    
        #output surpluses
        get!(e, "SmNC", SmNC);
        get!(e, "SmPL", SmPL);
        get!(e, "SmYC", SmYC);
        get!(e, "SmD", SmD);
        get!(e, "SfNC", SfNC);
        get!(e, "SfPL", SfPL);
        get!(e, "SfYC", SfYC);
        get!(e, "SfD", SfD);
    
        get!(e, "UmNC", UmNC);
        get!(e, "UmPL", UmPL);
        get!(e, "UmYC", UmYC);
        get!(e, "UmD", UmD);
        get!(e, "UfNC", UfNC);
        get!(e, "UfPL", UfPL);
        get!(e, "UfYC", UfYC);
        get!(e, "UfD", UfD);
    
        get!(e, "Pi", Pi);
    
        #Update Tax Rate
        tau = fxp.Transfer * sum(sum(hmPL_0 + hfPL_0, dims = 1) .* output, dims = 1) / (sum(sum(hmNC_0 + hfNC_0 + hmYC_0 + hfYC_0 + hmD_0 + hfD_0, dims = 1) .* output, dims = 1));
        get!(e, "tau", tau);
    
        #Convergence Check: bouncing between 2 iterations
        if (((itout>3) && (abs(change_dist[itout-1]-change_dist[itout-3])<tol/100)) || itout > 30)
            get!(e, "Pineg", 0);
            get!(e, "par", par);
            get!(e, "fxp", fxp);
            findneg = (Pi .< 0)
            if (sum(findneg)>0)
                get!(e, "Pineg", 1);
            end
            break;
        end
        #converged 
        if (change_dist[itout]<tol)
            print("After Iterations "*string(itout)*" converge with change "*string(change_dist[itout]));
            findneg = (Pi .< 0);
            if (sum(findneg)>0)
                get!(e, "Pineg", 1);
            end
            break;
        end
    end
end
export solve
end