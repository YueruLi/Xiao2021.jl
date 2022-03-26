module solve_eqm
# Write your package code here.
using NLsolve, LinearAlgebra
include("solve_dist.jl")
include("valueFunctions.jl")
include("helper.jl")
using .solve_dist, .valueFunctions, .helper 
function logit(x, fixedParam)
    return 1 ./ (1.0 .+ exp.(-fixedParam.lambda .* x))
end


#Solve for equilibrium
function solve(par, set, fxp, init; purpose="solve")
    kk = set.Ny * set.Na
    ii = set.Nx * set.Ne

    epsf = repeat([-2, 0, 2] .* fxp.sdf .+ par.muf * fxp.mum, inner=7)
    epsm = repeat([-2, 0, 2] .* fxp.sdm .+ fxp.mum, inner=7)
    alpha = repeat(fxp.alpha, inner=7)
    #humanCapital = repeat(fxp.hc, outer = 3);
    #jobProductivityType = repeat(fxp.p,outer=3);
    #jobAmenityType = repeat(fxp.alpha,inner=7);
    output = par.K .* [(par.a * hc^fxp.rho + (1 - par.a) * p^fxp.rho)^(1 / fxp.rho) for hc in repeat(fxp.hc, outer=3), p in repeat(fxp.p, outer=3)]
    outputb = repeat([par.b * hc for hc in fxp.hc], outer=3)
    #This stages solution from nlsolve is slightly different from that from matlab's fsolve. The difference is on the order of 1e-6
    #e = Dict("stages" => nlsolve(x -> solveStages(x, fxp), 12.50 .* ones(typeof(12.50), 8)).zero, "converge" => 1, "Pineg" => 0)
    stages = [0.0714286494397354, 0.117096067390252, 0.0614754501329449, 0.249999769056968, 0.0714293405864623, 0.175877941575804, 0.00269313154484208, 0.250000013739020]
    e = Dict("stages" => stages, "converge" => 1, "Pineg" => 0)
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

    #Joint values 
    PmNC_0 = SmNC + [x + y for x in UmNC_0, y in Pi0]
    PmPL_0 = SmPL + [x + y for x in UmPL_0, y in Pi0]
    PmYC_0 = SmYC + [x + y for x in UmYC_0, y in Pi0]
    PmD_0 = SmD + [x + y for x in UmD_0, y in Pi0]

    PfNC_0 = SfNC + [x + y for x in UfNC_0, y in Pi0]
    PfPL_0 = SfPL + [x + y for x in UfPL_0, y in Pi0]
    PfYC_0 = SfYC + [x + y for x in UfYC_0, y in Pi0]
    PfD_0 = SfD + [x + y for x in UfD_0, y in Pi0]


    #initial guess for tax rate
    tau = fxp.Transfer * sum(sum(hmPL_0 + hfPL_0, dims=1) .* output, dims=1) / (sum(sum(hmNC_0 + hfNC_0 + hmYC_0 + hfYC_0 + hmD_0 + hfD_0, dims=1) .* output, dims=1))
    #prepare to update these values
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
    Pi = zeros(Float64, 21)


    get!(e, "hmNC", hmNC_0)
    get!(e, "hmPL", hmPL_0)
    get!(e, "hmYC", hmYC_0)
    get!(e, "hmD", hmD_0)
    get!(e, "hfNC", hfNC_0)
    get!(e, "hfPL", hfPL_0)
    get!(e, "hfYC", hfYC_0)
    get!(e, "hfD", hfD_0)

    get!(e, "umNC", umNC_0)
    get!(e, "umPL", umPL_0)
    get!(e, "umYC", umYC_0)
    get!(e, "umD", umD_0)
    get!(e, "ufNC", ufNC_0)
    get!(e, "ufPL", ufPL_0)
    get!(e, "ufYC", ufYC_0)
    get!(e, "ufD", ufD_0)

    get!(e, "v", v0)

    #output surpluses
    get!(e, "SmNC", SmNC)
    get!(e, "SmPL", SmPL)
    get!(e, "SmYC", SmYC)
    get!(e, "SmD", SmD)
    get!(e, "SfNC", SfNC)
    get!(e, "SfPL", SfPL)
    get!(e, "SfYC", SfYC)
    get!(e, "SfD", SfD)

    get!(e, "UmNC", UmNC)
    get!(e, "UmPL", UmPL)
    get!(e, "UmYC", UmYC)
    get!(e, "UmD", UmD)
    get!(e, "UfNC", UfNC)
    get!(e, "UfPL", UfPL)
    get!(e, "UfYC", UfYC)
    get!(e, "UfD", UfD)
    get!(e, "tau", tau)
    get!(e, "Pi", Pi)

    #Value function iterations
    maxit = 10000
    change_dist = zeros(maxit, 1)
    tol = 1e-9
    x0 = vcat(umNC_0, umPL_0, umYC_0, umD_0, ufNC_0, ufPL_0, ufYC_0, ufD_0, vec(hmNC_0), vec(hmYC_0), vec(hmPL_0), vec(hmD_0), vec(hfNC_0), vec(hfYC_0), vec(hfPL_0), vec(hfD_0), vec(v0))
    x = zeros(Float64, length(x0))
    #outer loop
    for itout = 1:maxit

        effU = sum(umNC_0 + ufNC_0 + fxp.s1 * (umYC_0 + umD_0 + ufYC_0 + ufD_0)) +
               fxp.s2 * sum(sum(hmNC_0 + hmYC_0 + hmD_0 + hfNC_0 + hfYC_0 + hfD_0, dims=1))
        V = sum(v0)
        fxp = merge(fxp, (lambdau_NC=fxp.theta / (effU * V)^0.5, lambdau_YC=fxp.theta / (effU * V)^0.5 * fxp.s1, lambdae=fxp.theta / (effU * V)^0.5 * fxp.s2))
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
                    PmNC[i, j] = getPmNC(i, j, v0, fxp, par, Pi, gem, SmNC, UmNC, PmD_hat, PmPL_hat, PmNC_hat, epsm, alpha, output, tau)
                    PmPL[i, j] = getPmPL(i, j, Pi, fxp, UmPL, PmD_hat, PmYC_hat, PmPL_hat, epsm, alpha, output)
                    PmYC[i, j] = getPmYC(i, j, v0, Pi, fxp, par, epsm, gem, tau, output, alpha, UmYC, SmYC, PmD_hat, PmPL_hat, PmYC_hat)
                    PmD[i, j] = getPmD(i, j, v0, Pi, fxp, par, output, gem, tau, epsm, alpha, SmD, PmD_hat, UmD)

                    PfNC[i, j] = getPfNC(i, j, v0, fxp, par, Pi, gef, SfNC, UfNC, PfD_hat, PfPL_hat, PfNC_hat, epsf, alpha, output, tau)
                    PfPL[i, j] = getPfPL(i, j, Pi, par, fxp, UfPL, PfD_hat, PfYC_hat, PfPL_hat, epsf, alpha, output)
                    PfYC[i, j] = getPfYC(i, j, v0, Pi, fxp, par, epsf, gef, tau, output, alpha, UfYC, SfYC, PfD_hat, PfPL_hat, PfYC_hat)
                    PfD[i, j] = getPfD(i, j, v0, Pi, fxp, par, output, gef, tau, epsf, alpha, SfD, PfD_hat, UfD)
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
        #For development purposes, will be deleted in the future
        if (cmp(purpose, "value_fn_dev") == 0)
            ValueDict = Dict("SmNC" => SmNC, "UfNC" => UfNC)
            get!(ValueDict, "SmPL", SmPL)
            get!(ValueDict, "SmYC", SmYC)
            get!(ValueDict, "SmD", SmD)

            get!(ValueDict, "SfNC", SfNC)
            get!(ValueDict, "SfPL", SfPL)
            get!(ValueDict, "SfYC", SfYC)
            get!(ValueDict, "SfD", SfD)

            get!(ValueDict, "UfPL", UfPL)
            get!(ValueDict, "UfYC", UfYC)
            get!(ValueDict, "UfD", UfD)

            get!(ValueDict, "UmNC", UmNC)
            get!(ValueDict, "UmPL", UmPL)
            get!(ValueDict, "UmYC", UmYC)
            get!(ValueDict, "UmD", UmD)

            get!(ValueDict, "Pi", Pi)
            return ValueDict
        end


        x0 = vcat(umNC_0, umPL_0, umYC_0, umD_0, ufNC_0, ufPL_0, ufYC_0, ufD_0, vec(hmNC_0), vec(hmYC_0), vec(hmPL_0), vec(hmD_0), vec(hfNC_0), vec(hfYC_0), vec(hfPL_0), vec(hfD_0), vec(v0))
        x_orig = x0

        #For development purposes, will be deleted in the future
        if (cmp(purpose, "solve_dist_dev") == 0)
            return fxp, SmNC, SmPL, SmYC, SmD, SfNC, SfPL, SfYC, SfD, e["stages"], x0
        end
        #For development purposes, will be deleted in the future
        if (cmp(purpose, "solve_dist_matrix") == 0)
            x = solveDist(x0, set, fxp, par, SmNC, SmPL, SmYC, SmD, SfNC, SfPL, SfYC, SfD, e["stages"])
            x[x.<1e-30] .= 0
            v0 = x[length(x)-set.Ny*set.Na+1:length(x)]
            #delete v0 from x
            x = deleteat!(x, length(x)-set.Ny*set.Na+1:length(x))
            x = reshape(x, set.Nx * set.Ne, :)

            umNC_0 = x[:, 1]
            umPL_0 = x[:, 2]
            umYC_0 = x[:, 3]
            umD_0 = x[:, 4]
            ufNC_0 = x[:, 5]
            ufPL_0 = x[:, 6]
            ufYC_0 = x[:, 7]
            ufD_0 = x[:, 8]

            hmNC_0 = x[:, 9:29]
            hmYC_0 = x[:, 30:50]
            hmPL_0 = x[:, 51:71]
            hmD_0 = x[:, 72:92]
            hfNC_0 = x[:, 93:113]
            hfYC_0 = x[:, 114:134]
            hfPL_0 = x[:, 135:155]
            hfD_0 = x[:, 156:176]

            solveDistMatrix = Dict("umNC_0" => umNC_0, "hmNC_0" => hmNC_0)
            get!(solveDistMatrix, "hmPL_0", hmPL_0)
            get!(solveDistMatrix, "hmYC_0", hmYC_0)
            get!(solveDistMatrix, "hmD_0", hmD_0)

            get!(solveDistMatrix, "hfNC_0", hfNC_0)
            get!(solveDistMatrix, "hfPL_0", hfPL_0)
            get!(solveDistMatrix, "hfYC_0", hfYC_0)
            get!(solveDistMatrix, "hfD_0", hfD_0)

            get!(solveDistMatrix, "umPL_0", umPL_0)
            get!(solveDistMatrix, "umYC_0", umYC_0)
            get!(solveDistMatrix, "umD_0", umD_0)

            get!(solveDistMatrix, "ufNC_0", ufNC_0)
            get!(solveDistMatrix, "ufPL_0", ufPL_0)
            get!(solveDistMatrix, "ufYC_0", ufYC_0)
            get!(solveDistMatrix, "ufD_0", ufD_0)

            return solveDistMatrix
        end


        for it2 = 1:maxit
            x = solveDist(x0, set, fxp, par, SmNC, SmPL, SmYC, SmD, SfNC, SfPL, SfYC, SfD, e["stages"])
            x[x.<1e-30] .= 0
            dev = (x - x0) ./ x0 #percentage deviation 
            replace!(dev, Inf => NaN)
            replace!(dev, NaN => 0)
            change2 = sum(dev .^ 2)
            if (change2 < tol)
                println("converged in " * string(it2) * " iterations, with change:" * string(change2))
                break
            end
            x0 = copy(x)
        end
        #If this fixed point in distributions is different from the initial
        #distribution, then repeat the who process again
        dev = (x - x_orig) ./ x_orig        #percentage deviation
        replace!(dev, Inf => NaN)
        replace!(dev, NaN => 0)
        change_dist[itout] = sum(dev .^ 2)


        #Print Progress
        println("==================" * "Outer Iteration " * string(itout) * " with change " * string(change_dist[itout]) * "==================")
        #Update Solution for distributions
        v0 = x[length(x)-set.Ny*set.Na+1:length(x)]
        #delete v0 from x
        x = deleteat!(x, length(x)-set.Ny*set.Na+1:length(x))
        x = reshape(x, set.Nx * set.Ne, :)

        umNC_0 = x[:, 1]
        umPL_0 = x[:, 2]
        umYC_0 = x[:, 3]
        umD_0 = x[:, 4]
        ufNC_0 = x[:, 5]
        ufPL_0 = x[:, 6]
        ufYC_0 = x[:, 7]
        ufD_0 = x[:, 8]
        hmNC_0 = x[:, 9:29]
        hmYC_0 = x[:, 30:50]
        hmPL_0 = x[:, 51:71]
        hmD_0 = x[:, 72:92]
        hfNC_0 = x[:, 93:113]
        hfYC_0 = x[:, 114:134]
        hfPL_0 = x[:, 135:155]
        hfD_0 = x[:, 156:176]

        #output distribution 
        e["hmNC"] = hmNC_0
        e["hmPL"] = hmPL_0
        e["hmYC"] = hmYC_0
        e["hmD"] = hmD_0

        e["hfNC"] = hfNC_0
        e["hfPL"] = hfPL_0
        e["hfYC"] = hfYC_0
        e["hfD"] = hfD_0

        e["umNC"] = umNC_0
        e["umPL"] = umPL_0
        e["umYC"] = umYC_0
        e["umD"] = umD_0

        e["ufNC"] = ufNC_0
        e["ufPL"] = ufPL_0
        e["ufYC"] = ufYC_0
        e["ufD"] = ufD_0

        e["v"] = v0



        #output surpluses
        e["SmNC"] = SmNC
        e["SmPL"] = SmPL
        e["SmYC"] = SmYC
        e["SmD"] = SmD

        e["SfNC"] = SfNC
        e["SfPL"] = SfPL
        e["SfYC"] = SfYC
        e["SfD"] = SfD

        e["UmNC"] = UmNC
        e["UmPL"] = UmPL
        e["UmYC"] = UmYC
        e["UmD"] = UmD

        e["UfNC"] = UfNC
        e["UfPL"] = UfPL
        e["UfYC"] = UfYC
        e["UfD"] = UfD

        e["Pi"] = Pi

        #Update Tax Rate
        tau = fxp.Transfer * sum(sum(hmPL_0 + hfPL_0, dims=1) .* output, dims=1) / (sum(sum(hmNC_0 + hfNC_0 + hmYC_0 + hfYC_0 + hmD_0 + hfD_0, dims=1) .* output, dims=1))
        e["tau"] = tau

        #for development purpose, will delete later
        if (cmp(purpose, "one_outer_loop") == 0)
            if (itout == 1)
                return fxp, e
            end
        end

        if (cmp(purpose, "two_outer_loop") == 0)
            if (itout == 2)
                return fxp, e
            end
        end

        #Convergence Check: bouncing between 2 iterations
        if (((itout > 3) && (abs(change_dist[itout-1] - change_dist[itout-3]) < tol / 100)) || itout > 30)
            get!(e, "Pineg", 0)
            get!(e, "par", par)
            get!(e, "fxp", fxp)
            findneg = (Pi .< 0)
            if (sum(findneg) > 0)
                get!(e, "Pineg", 1)
            end
            break
        end
        #converged 
        if (change_dist[itout] < tol)
            println("After Iterations " * string(itout) * " converge with change " * string(change_dist[itout]))
            findneg = (Pi .< 0)
            if (sum(findneg) > 0)
                get!(e, "Pineg", 1)
            end
            break
        end
    end

    if (cmp(purpose, "solve_eqm_wo_wage") == 0)
        return fxp, e
    end


    #Solve for equilibrium wages 
    # calculate wages even when didn't converge
    # if e["converge"] == 1 && e["Pineg"] == 0

    #quilibrium wage phi0 (update with surplus share)
    v0mat = repeat(vec(v0)', outer=[ii, 1])
    phi0m_NC = zeros(Float64, ii, kk)
    phi0m_PL = zeros(Float64, ii, kk)
    phi0m_YC = zeros(Float64, ii, kk)
    phi0m_D = zeros(Float64, ii, kk)
    phi0f_NC = zeros(Float64, ii, kk)
    phi0f_PL = zeros(Float64, ii, kk)
    phi0f_YC = zeros(Float64, ii, kk)
    phi0f_D = zeros(Float64, ii, kk)

    Wm_NC = [UmNC[x] + par.sigma * SmNC[x, y] for x in 1:21, y in 1:21] .* (SmNC .> 0)
    Wm_PL = [UmPL[x] + par.sigma * SmPL[x, y] for x in 1:21, y in 1:21] .* (SmPL .> 0)
    Wm_YC = [UmYC[x] + par.sigma * SmYC[x, y] for x in 1:21, y in 1:21] .* (SmYC .> 0)
    Wm_D = [UmD[x] + par.sigma * SmD[x, y] for x in 1:21, y in 1:21] .* (SmD .> 0)

    Wf_NC = [UfNC[x] + par.sigma * SfNC[x, y] for x in 1:21, y in 1:21] .* (SfNC .> 0)
    Wf_PL = [UfPL[x] + par.sigma * SfPL[x, y] for x in 1:21, y in 1:21] .* (SfPL .> 0)
    Wf_YC = [UfYC[x] + par.sigma * SfYC[x, y] for x in 1:21, y in 1:21] .* (SfYC .> 0)
    Wf_D = [UfD[x] + par.sigma * SfD[x, y] for x in 1:21, y in 1:21] .* (SfD .> 0)

    gem = zeros(set.Nx, set.Nx, set.Ny)
    gef = zeros(set.Nx, set.Nx, set.Ny)
    for y = 1:set.Ny
        for x0 = 1:set.Nx
            for x1 = 1:set.Nx
                if x1 == x0 + 1
                    gem[x0, x1, y] = par.d1m + par.d2m * fxp.p[y]
                    gef[x0, x1, y] = par.d1f + par.d2f * fxp.p[y]
                elseif x1 == x0 + 2
                    gem[x0, x1, y] = (par.d1m + par.d2m * fxp.p[y]) * fxp.d3
                    gef[x0, x1, y] = (par.d1f + par.d2f * fxp.p[y]) * fxp.d3
                end
            end
            gem[x0, x0, y] = 1 - sum(gem[x0, :, y])
            gef[x0, x0, y] = 1 - sum(gef[x0, :, y])

        end

        gem[set.Nx, set.Nx, y] = 1
        gef[set.Nx, set.Nx, y] = 1
    end
    A = ones(set.Nx, set.Nx)
    B = kron(Matrix(1.0I, set.Ne, set.Ne), A)
    gem = repeat(gem, outer=[set.Ne, set.Ne, set.Na])
    gem = gem .* repeat(B, outer=[1, 1, ii])
    gef = repeat(gef, outer=[set.Ne, set.Ne, set.Na])
    gef = gef .* repeat(B, outer=[1, 1, ii])
    gu = Matrix(1.0I, ii, ii)

    #phase 0
    for i = 1:ii
        for j = 1:kk
            gem_j = gem[i, :, j]     # for current HC i and job j, transition matrix from x to x'
            gef_j = gef[i, :, j]     # for current HC i and job j, transition matrix from x to x'
            gu_i = gu[i, :]

            #NC stage

            #Male
            vmI = repeat(SmNC[:, j] + UmNC, outer=[1, kk])   # max value incumbent willing to give to worker
            vmP = SmNC + repeat(UmNC, outer=[1, kk])       # poaching firm value
            dm = vmP - vmI
            Am = (vmP .> vmI) .* ((vmI + par.sigma * (vmP - vmI)) .* logit(dm, fxp) +
                                  vmI .* (1 .- logit(dm, fxp))) + (vmP .<= vmI) .* (vmP .> Wm_NC[i, j]) .* ((vmP + par.sigma * (vmI - vmP)) .* (1 .- logit(dm, fxp)) +
                                                                                                            vmP .* logit(dm, fxp)) + (vmP .<= Wm_NC[i, j]) .* (Wm_NC[i, j] .* (1 .- logit(dm, fxp)) +
                                                                                                                                                               vmP .* logit(dm, fxp))

            #no wage update for HC growth
            Wmhat_NC = UmNC .* (SmNC[:, j] .<= 0) + Wm_NC[i, j] .* (SmNC[:, j] .> 0)
            Wmhat_PL = UmPL .* (SmPL[:, j] .<= 0) + Wm_NC[i, j] .* (SmPL[:, j] .> 0)
            Wmhat_D = UmD .* (SmD[:, j] .<= 0) + Wm_NC[i, j] .* (SmD[:, j] .> 0)

            Em_jk = fxp.deltam_NC * UmNC + fxp.lambdae * sum(v0mat .* (Am - repeat(Wmhat_NC, outer=[1, kk])), dims=2) .* (SmNC[:, j] .> 0) +
                    fxp.gamma * Wmhat_D + fxp.chi * Wmhat_PL + (1 - fxp.deltam_NC - fxp.gamma - fxp.chi) * Wmhat_NC

            phi0m_NC[i, j] = Wm_NC[i, j] - epsm[i] * alpha[j] - first(fxp.beta * gem_j' * Em_jk)

            #female 

            vfI = repeat(SfNC[:, j] + UfNC, outer=[1, kk])   # max value incumbent willing to give to worker
            vfP = SfNC + repeat(UfNC, outer=[1, kk])       # poaching firm value
            df = vfP - vfI

            Af = (vfP .> vfI) .* ((vfI + par.sigma * (vfP - vfI)) .* logit(df, fxp) +
                                  vfI .* (1 .- logit(df, fxp))) + (vfP .<= vfI) .* (vfP .> Wf_NC[i, j]) .* ((vfP + par.sigma * (vfI - vfP)) .* (1 .- logit(df, fxp)) +
                                                                                                            vfP .* logit(df, fxp)) + (vfP .<= Wf_NC[i, j]) .* (Wf_NC[i, j] .* (1 .- logit(df, fxp)) +
                                                                                                                                                               vmP .* logit(df, fxp))

            #now wage update for HC growth
            Wfhat_NC = UfNC .* (SfNC[:, j] .<= 0) + Wf_NC[i, j] .* (SfNC[:, j] .> 0)
            Wfhat_PL = UfPL .* (SfPL[:, j] .<= 0) + Wf_NC[i, j] .* (SfPL[:, j] .> 0)
            Wfhat_D = UfD .* (SfD[:, j] .<= 0) + Wf_NC[i, j] .* (SfD[:, j] .> 0)

            Ef_jk = fxp.deltaf_NC * UfNC + fxp.lambdae * sum(v0mat .* (Af - repeat(Wfhat_NC, outer=[1, kk])), dims=2) .* (SfNC[:, j] .> 0) +
                    fxp.gamma * Wfhat_D + fxp.chi * Wfhat_PL + (1 - fxp.deltaf_NC - fxp.gamma - fxp.chi) * Wfhat_NC

            phi0f_NC[i, j] = Wf_NC[i, j] - epsf[i] * alpha[j] - first(fxp.beta * gef_j' * Ef_jk)

            #PL stage wages never used

            #male
            Wmhat_PL = UmPL .* (SmPL[:, j] .<= 0) + Wm_PL[i, j] .* (SmPL[:, j] .> 0)
            Wmhat_YC = UmYC .* (SmYC[:, j] .<= 0) + Wm_PL[:, j] .* (SmYC[:, j] .> 0)
            Wmhat_D = UmD .* (SmD[:, j] .<= 0) + Wm_PL[:, j] .* (SmD[:, j] .> 0)

            Em_jk = fxp.deltam_YC * UmPL + fxp.gamma * Wmhat_D + fxp.etam * Wmhat_YC + (1 - fxp.deltam_YC - fxp.gamma - fxp.etam) * Wmhat_PL
            phi0m_PL[i, j] = Wm_PL[i, j] - epsm[i] * alpha[j] - first(fxp.beta * gu_i' * Em_jk)

            #female
            Wfhat_PL = UfPL .* (SfPL[:, j] .<= 0) + Wf_PL[i, j] .* (SfPL[:, j] .> 0)
            Wfhat_YC = UfYC .* (SfYC[:, j] .<= 0) + Wf_PL[i, j] .* (SfYC[:, j] .> 0)
            Wfhat_D = UfD .* (SfD[:, j] .<= 0) + Wf_PL[i, j] .* (SfD[:, j] .> 0)
            Ef_jk = fxp.deltaf_YC * UfPL + fxp.gamma * Wfhat_D + fxp.etaf * Wfhat_YC + (1 - fxp.deltaf_YC - fxp.gamma - fxp.etaf) * Wfhat_PL
            phi0f_PL[i, j] = Wf_PL[i, j] - (par.M + epsf[i]) * alpha[j] - first(fxp.beta * gu_i' * Ef_jk)

            #YC stage

            #male
            vmI = repeat(SmYC[:, j] + UmYC, outer=[1, kk])
            vmP = SmYC + repeat(UmYC, outer=[1, kk])
            dm = vmP - vmI
            Am = (vmP .> vmI) .* ((vmI + par.sigma * (vmP - vmI)) .* logit(dm, fxp) + vmI .* (1 .- logit(dm, fxp))) + (vmP .<= vmI) .* (vmP .> Wm_YC[i, j]) .* ((vmP + par.sigma * (vmI - vmP)) .* (1 .- logit(dm, fxp)) + vmP .* logit(dm, fxp)) + (vmP .<= Wm_YC[i, j]) .* (Wm_YC[i, j] .* (1 .- logit(dm, fxp)) + vmP .* logit(dm, fxp))

            #no wage update for HC growth 
            Wmhat_YC = UmYC .* (SmYC[:, j] .<= 0) + Wm_YC[i, j] .* (SmYC[:, j] .> 0)
            Wmhat_PL = UmPL .* (SmPL[:, j] .<= 0) + Wm_YC[i, j] .* (SmPL[:, j] .> 0)
            Wmhat_D = UmD .* (SmD[:, j] .<= 0) + Wm_YC[i, j] .* (SmD[:, j] .> 0)
            Em_jk = fxp.deltam_YC * UmYC + fxp.lambdae * sum(v0mat .* (Am - repeat(Wmhat_YC, outer=[1, kk])), dims=2) .* (SmYC[:, j] .> 0) + fxp.gamma * Wmhat_D + fxp.chi * Wmhat_PL + (1 - fxp.deltam_YC - fxp.gamma - fxp.chi) * Wmhat_YC
            phi0m_YC[i, j] = Wm_YC[i, j] - epsm[i] * alpha[j] - first(fxp.beta * gem_j' * Em_jk)

            #female
            vfI = repeat(SfYC[:, j] + UfYC, outer=[1, kk])
            vfP = SfYC + repeat(UfYC, outer=[1, kk])
            df = vfP - vfI
            Af = (vfP .> vfI) .* ((vfI + par.sigma * (vfP - vfI)) .* logit(df, fxp) + vfI .* (1 .- logit(df, fxp))) + (vfP .<= vfI) .* (vfP .> Wf_YC[i, j]) .* ((vfP + par.sigma * (vfI - vfP)) .* (1 .- logit(df, fxp)) + vfP .* logit(df, fxp)) + (vfP .<= Wf_YC[i, j]) .* (Wf_YC[i, j] .* (1 .- logit(df, fxp)) + vfP .* logit(df, fxp))

            Wfhat_YC = UfYC .* (SfYC[:, j] .<= 0) + Wf_YC[i, j] .* (SfYC[:, j] .> 0)
            Wfhat_PL = UfPL .* (SfPL[:, j] .<= 0) + Wf_YC[i, j] .* (SfPL[:, j] .> 0)
            Wfhat_D = UfD .* (SfD[:, j] .<= 0) + Wf_YC[i, j] .* (SfD[:, j] .> 0)
            Ef_jk = fxp.deltaf_YC * UfYC + fxp.lambdae * sum(v0mat .* (Af - repeat(Wfhat_YC, outer=[1, kk])), dims=2) .* (SfYC[:, j] .> 0) + fxp.gamma * Wfhat_D + fxp.chi * Wfhat_PL + (1 - fxp.deltaf_YC - fxp.gamma - fxp.chi) * Wfhat_YC
            phi0f_YC[i, j] = Wf_YC[i, j] - (par.M + epsf[i]) * alpha[j] - first(fxp.beta * gef_j' * Ef_jk)

            #D stage
            #male
            vmI = repeat(SmD[:, j] + UmD, outer=[1, kk])
            vmP = SmD + repeat(UmD, outer=[1, kk])
            dm = vmP - vmI
            Am = (vmP .> vmI) .* ((vmI + par.sigma * (vmP - vmI)) .* logit(dm, fxp) + vmI .* (1 .- logit(dm, fxp))) + (vmP .<= vmI) .* (vmP .> Wm_D[i, j]) .* ((vmP + par.sigma * (vmI - vmP)) .* (1 .- logit(dm, fxp)) + vmP .* logit(dm, fxp)) .+ (vmP .<= Wm_D[i, j]) .* (Wm_D[i, j] .* (1 .- logit(dm, fxp)) + vmP .* logit(dm, fxp))

            Wmhat_D = UmD .* (SmD[:, j] .<= 0) + Wm_D[i, j] .* (SmD[:, j] .> 0)
            Em_jk = fxp.delta * UmD + fxp.lambdae * sum(v0mat .* (Am - repeat(Wmhat_D, outer=[1, kk])), dims=2) .* (SmD[:, j] .> 0) + (1 - fxp.phi - fxp.delta) * Wmhat_D
            phi0m_D[i, j] = Wm_D[i, j] - epsm[i] * alpha[j] - first(fxp.beta * gem_j' * Em_jk)

            #Female
            vfI = repeat(SfD[:, j] + UfD, outer=[1, kk])
            vfP = SfD + repeat(UfD, outer=[1, kk])
            df = vfP - vfI
            Af = (vfP .> vfI) .* ((vfI + par.sigma * (vfP - vfI)) .* logit(df, fxp) + vfI .* (1 .- logit(df, fxp))) + (vfP .<= vfI) .* (vfP .> Wf_D[i, j]) .* ((vfP + par.sigma * (vfI - vfP)) .* (1 .- logit(df, fxp)) + vfP .* logit(df, fxp)) + (vfP .<= Wf_D[i, j]) .* (Wf_D[i, j] .* (1 .- logit(df, fxp)) + vfP .* logit(df, fxp))

            Wfhat_D = UfD .* (SfD[:, j] .<= 0) + Wf_D[i, j] .* (SfD[:, j] .> 0)
            Ef_jk = fxp.delta * UfD + fxp.lambdae * sum(v0mat .* (Af - repeat(Wfhat_D, outer=[1, kk])), dims=2) .* (SfD[:, j] .> 0) + (1 - fxp.phi - fxp.delta) * Wfhat_D
            phi0f_D[i, j] = Wf_D[i, j] - epsf[i] * alpha[j] - first(fxp.beta * gef_j' * Ef_jk)
        end
    end
    phi0m_NC[SmNC.<=0] .= NaN
    phi0m_PL[SmPL.<=0] .= NaN
    phi0m_YC[SmYC.<=0] .= NaN
    phi0m_D[SmD.<=0] .= NaN
    phi0f_NC[SfNC.<=0] .= NaN
    phi0f_PL[SfPL.<=0] .= NaN
    phi0f_YC[SfYC.<=0] .= NaN
    phi0f_D[SfD.<=0] .= NaN

    #e qm wages phi1(y, y', x) -- wage at JTJ transitions from y to y'
    #make JTJ transition from y to y' when S(x, y')>S(x, y)
    Wm1_NC = zeros(Float64, kk, kk, ii)
    Wm1_YC = zeros(Float64, kk, kk, ii)
    Wm1_D = zeros(Float64, kk, kk, ii)
    Wf1_NC = zeros(Float64, kk, kk, ii)
    Wf1_YC = zeros(Float64, kk, kk, ii)
    Wf1_D = zeros(Float64, kk, kk, ii)
    phi1m_NC = zeros(Float64, kk, kk, ii)
    fill!(phi1m_NC, NaN)
    phi1m_YC = zeros(Float64, kk, kk, ii)
    fill!(phi1m_YC, NaN)
    phi1m_D = zeros(Float64, kk, kk, ii)
    fill!(phi1m_D, NaN)
    phi1f_NC = zeros(Float64, kk, kk, ii)
    fill!(phi1f_NC, NaN)
    phi1f_YC = zeros(Float64, kk, kk, ii)
    fill!(phi1f_YC, NaN)
    phi1f_D = zeros(Float64, kk, kk, ii)
    fill!(phi1f_D, NaN)
    #Wm_NC(k,j,i) where j is the poaching firm that succesfully poached from k (Incumbent)
    for i = 1:ii
        Sm_yP = repeat(SmNC[i, :]', outer=[kk, 1])
        Sm_yI = repeat(SmNC[i, :], outer=[1, kk])
        #JTJ transitions only happen when poaching>incumbent, and incumbent>0
        Wm1_NC[:, :, i] = (UmNC[i] * ones(kk, kk) + Sm_yI + par.sigma * (Sm_yP - Sm_yI)) .* (Sm_yP .> Sm_yI) .* (Sm_yI .> 0)

        Sm_yP = repeat(SmYC[i, :]', outer=[kk, 1])
        Sm_yI = repeat(SmYC[i, :], outer=[1, kk])
        #JTJ transitions only happen when poaching>incumbent, and incumbent>0
        Wm1_YC[:, :, i] = (UmYC[i] * ones(kk, kk) + Sm_yI + par.sigma * (Sm_yP - Sm_yI)) .* (Sm_yP .> Sm_yI) .* (Sm_yI .> 0)

        Sm_yP = repeat(SmD[i, :]', outer=[kk, 1])
        Sm_yI = repeat(SmD[i, :], outer=[1, kk])
        #JTJ transitions only happen when poaching>incumbent, and incumbent>0
        Wm1_D[:, :, i] = (UmD[i] * ones(kk, kk) + Sm_yI + par.sigma * (Sm_yP - Sm_yI)) .* (Sm_yP .> Sm_yI) .* (Sm_yI .> 0)

        Sf_yP = repeat(SfNC[i, :]', outer=[kk, 1])
        Sf_yI = repeat(SfNC[i, :], outer=[1, kk])
        #JTJ transitions only happen when poaching>incumbent, and incumbent>0
        Wf1_NC[:, :, i] = (UfNC[i] * ones(kk, kk) + Sf_yI + par.sigma * (Sf_yP - Sf_yI)) .* (Sf_yP .> Sf_yI) .* (Sf_yI .> 0)

        Sf_yP = repeat(SfYC[i, :]', outer=[kk, 1])
        Sf_yI = repeat(SfYC[i, :], outer=[1, kk])
        #JTJ transitions only happen when poaching>incumbent, and incumbent>0
        Wf1_YC[:, :, i] = (UfYC[i] * ones(kk, kk) + Sf_yI + par.sigma * (Sf_yP - Sf_yI)) .* (Sf_yP .> Sf_yI) .* (Sf_yI .> 0)

        Sf_yP = repeat(SfD[i, :]', outer=[kk, 1])
        Sf_yI = repeat(SfD[i, :], outer=[1, kk])
        #JTJ transitions only happen when poaching>incumbent, and incumbent>0
        Wf1_D[:, :, i] = (UfD[i] * ones(kk, kk) + Sf_yI + par.sigma * (Sf_yP - Sf_yI)) .* (Sf_yP .> Sf_yI) .* (Sf_yI .> 0)
    end
    #calculate expected future payoff for worker x at poaching firm y'
    #phi1(y,y',x) = phi1(k,j,i)
    for i = 1:ii
        for j = 1:kk
            for k = 1:kk
                gem_j = gem[i, :, j]
                gef_j = gef[i, :, j]

                #NC stage
                #male
                if Wm1_NC[k, j, i] != 0
                    vmI = repeat(SmNC[:, j] + UmNC, outer=[1, kk]) #max value incumbent willing to offer
                    vmP = SmNC + repeat(UmNC, outer=[1, kk])    #poaching firm value 
                    dm = vmP - vmI
                    Am = (vmP .> vmI) .* ((vmI + par.sigma * (vmP - vmI)) .* logit(dm, fxp) + vmI .* (1 .- logit(dm, fxp))) + (vmP .<= vmI) .* (vmP .> Wm1_NC[k, j, i]) .* ((vmP + par.sigma * (vmI - vmP)) .* (1 .- logit(dm, fxp)) + vmP .* logit(dm, fxp)) + (vmP .<= Wm1_NC[k, j, i]) .* (Wm1_NC[k, j, i] .* (1 .- logit(dm, fxp)) + vmP .* logit(dm, fxp))

                    Wmhat_NC = UmNC .* (SmNC[:, j] .<= 0) + Wm1_NC[k, j, i] .* (SmNC[:, j] .> 0)
                    Wmhat_PL = UmPL .* (SmPL[:, j] .<= 0) + Wm1_NC[k, j, i] .* (SmPL[:, j] .> 0)
                    Wmhat_D = UmD .* (SmD[:, j] .<= 0) + Wm1_NC[k, j, i] .* (SmD[:, j] .> 0)
                    Em_j = fxp.deltam_NC * UmNC + fxp.lambdae * sum(v0mat .* (Am - repeat(Wmhat_NC, outer=[1, kk])), dims=2) .* (SmNC[:, j] .> 0) + fxp.gamma * Wmhat_D + fxp.chi * Wmhat_PL + (1 - fxp.deltam_NC - fxp.gamma - fxp.chi) * Wmhat_NC
                    phi1m_NC[k, j, i] = Wm1_NC[k, j, i] - epsm[i] * alpha[j] - first(fxp.beta * gem_j' * Em_j)

                end
                if Wf1_NC[k, j, i] != 0
                    vfI = repeat(SfNC[:, j] + UfNC, outer=[1, kk]) #max value incumbent willing to offer
                    vfP = SfNC + repeat(UfNC, outer=[1, kk])    #poaching firm value 
                    df = vfP - vfI

                    Af = (vfP .> vfI) .* ((vfI + par.sigma * (vfP - vfI)) .* logit(df, fxp) + vfI .* (1 .- logit(df, fxp))) + (vfP .<= vfI) .* (vfP .> Wf1_NC[k, j, i]) .* ((vfP + par.sigma * (vfI - vfP)) .* (1 .- logit(df, fxp)) + vfP .* logit(df, fxp)) + (vfP .<= Wf1_NC[k, j, i]) .* (Wf1_NC[k, j, i] .* (1 .- logit(df, fxp)) + vfP .* logit(df, fxp))


                    Wfhat_NC = UfNC .* (SfNC[:, j] .<= 0) + Wf1_NC[k, j, i] .* (SfNC[:, j] .> 0)
                    Wfhat_PL = UfPL .* (SfPL[:, j] .<= 0) + Wf1_NC[k, j, i] .* (SfPL[:, j] .> 0)
                    Wfhat_D = UfD .* (SfD[:, j] .<= 0) + Wf1_NC[k, j, i] .* (SfD[:, j] .> 0)
                    Ef_j = fxp.deltaf_NC * UfNC + fxp.lambdae * sum(v0mat .* (Af - repeat(Wfhat_NC, outer=[1, kk])), dims=2) .* (SfNC[:, j] .> 0) + fxp.gamma * Wfhat_D + fxp.chi * Wfhat_PL + (1 - fxp.deltaf_NC - fxp.gamma - fxp.chi) * Wfhat_NC
                    phi1f_NC[k, j, i] = Wf1_NC[k, j, i] - epsf[i] * alpha[j] - first(fxp.beta * gef_j' * Ef_j)

                end

                #YC stage
                #Male
                if Wm1_YC[k, j, i] != 0
                    vmI = repeat(SmYC[:, j] + UmYC, outer=[1, kk])
                    vmP = SmYC + repeat(UmYC, outer=[1, kk])
                    dm = vmP - vmI

                    Am = (vmP .> vmI) .* ((vmI + par.sigma * (vmP - vmI)) .* logit(dm, fxp) + vmI .* (1 .- logit(dm, fxp))) + (vmP .<= vmI) .* (vmP .> Wm1_YC[k, j, i]) .* ((vmP + par.sigma * (vmI - vmP)) .* (1 .- logit(dm, fxp)) + vmP .* logit(dm, fxp)) + (vmP .<= Wm1_YC[k, j, i]) .* (Wm1_YC[k, j, i] .* (1 .- logit(dm, fxp)) + vmP .* logit(dm, fxp))

                    Wmhat_YC = UmYC .* (SmYC[:, j] .<= 0) + Wm1_YC[k, j, i] .* (SmYC[:, j] .> 0)
                    Wmhat_PL = UmPL .* (SmPL[:, j] .<= 0) + Wm1_YC[k, j, i] .* (SmPL[:, j] .> 0)
                    Wmhat_D = UmD .* (SmD[:, j] .<= 0) + Wm1_YC[k, j, i] .* (SmD[:, j] .> 0)

                    Em_j = fxp.deltam_YC * UmYC + fxp.lambdae * sum(v0mat .* (Am - repeat(Wmhat_YC, outer=[1, kk])), dims=2) .* (SmYC[:, j] .> 0) + fxp.gamma * Wmhat_D + fxp.chi * Wmhat_PL + (1 - fxp.deltam_YC - fxp.gamma - fxp.chi) * Wmhat_YC
                    phi1m_YC[k, j, i] = Wm1_YC[k, j, i] - epsm[i] * alpha[j] - first(fxp.beta * gem_j' * Em_j)
                end

                #Female
                if Wf1_YC[k, j, i] != 0
                    vfI = repeat(SfYC[:, j] + UfYC, outer=[1, kk])
                    vfP = SfYC + repeat(UfYC, outer=[1, kk])
                    df = vfP - vfI

                    Af = (vfP .> vfI) .* ((vfI + par.sigma * (vfP - vfI)) .* logit(df, fxp) + vfI .* (1 .- logit(df, fxp))) + (vfP .<= vfI) .* (vfP .> Wf1_YC[k, j, i]) .* ((vfP + par.sigma * (vfI - vfP)) .* (1 .- logit(df, fxp)) + vfP .* logit(df, fxp)) + (vfP .<= Wf1_YC[k, j, i]) .* (Wf1_YC[k, j, i] .* (1 .- logit(df, fxp)) + vfP .* logit(df, fxp))

                    Wfhat_YC = UfYC .* (SfYC[:, j] .<= 0) + Wf1_YC[k, j, i] .* (SfYC[:, j] .> 0)
                    Wfhat_PL = UfPL .* (SfPL[:, j] .<= 0) + Wf1_YC[k, j, i] .* (SfPL[:, j] .> 0)
                    Wfhat_D = UfD .* (SfD[:, j] .<= 0) + Wf1_YC[k, j, i] .* (SfD[:, j] .> 0)

                    Ef_j = fxp.deltaf_YC * UfYC + fxp.lambdae * sum(v0mat .* (Af - repeat(Wfhat_YC, outer=[1, kk])), dims=2) .* (SfYC[:, j] .> 0) + fxp.gamma * Wfhat_D + fxp.chi * Wfhat_PL + (1 - fxp.deltaf_YC - fxp.gamma - fxp.chi) * Wfhat_YC
                    phi1f_YC[k, j, i] = Wf1_YC[k, j, i] - (par.M + epsf[i]) * alpha[j] - first(fxp.beta * gef_j' * Ef_j)
                end

                #D stage
                #Male
                if Wm1_D[k, j, i] != 0
                    vmI = repeat(SmD[:, j] + UmD, outer=[1, kk])
                    vmP = SmD + repeat(UmD, outer=[1, kk])
                    dm = vmP - vmI


                    Am = (vmP .> vmI) .* ((vmI + par.sigma * (vmP - vmI)) .* logit(dm, fxp) + vmI .* (1 .- logit(dm, fxp))) + (vmP .<= vmI) .* (vmP .> Wm1_D[k, j, i]) .* ((vmP + par.sigma * (vmI - vmP)) .* (1 .- logit(dm, fxp)) + vmP .* logit(dm, fxp)) + (vmP .<= Wm1_D[k, j, i]) .* (Wm1_D[k, j, i] .* (1 .- logit(dm, fxp)) + vmP .* logit(dm, fxp))

                    Wmhat_D = UmD .* (SmD[:, j] .<= 0) + Wm1_D[k, j, i] .* (SmD[:, j] .> 0)

                    Em_j = fxp.delta * UmD + fxp.lambdae * sum(v0mat .* (Am - repeat(Wmhat_D, outer=[1, kk])), dims=2) .* (SmD[:, j] .> 0) + (1 - fxp.phi - fxp.delta) * Wmhat_D
                    phi1m_D[k, j, i] = Wm1_D[k, j, i] - epsm[i] * alpha[j] - first(fxp.beta * gem_j' * Em_j)
                end

                #Female
                if Wf1_D[k, j, i] != 0
                    vfI = repeat(SfD[:, j] + UfD, outer=[1, kk])
                    vfP = SfD + repeat(UfD, outer=[1, kk])
                    df = vfP - vfI


                    Af = (vfP .> vfI) .* ((vfI + par.sigma * (vfP - vfI)) .* logit(df, fxp) + vfI .* (1 .- logit(df, fxp))) + (vfP .<= vfI) .* (vfP .> Wf1_D[k, j, i]) .* ((vfP + par.sigma * (vfI - vfP)) .* (1 .- logit(df, fxp)) + vfP .* logit(df, fxp)) + (vfP .<= Wf1_D[k, j, i]) .* (Wf1_D[k, j, i] .* (1 .- logit(df, fxp)) + vfP .* logit(df, fxp))

                    Wfhat_D = UfD .* (SfD[:, j] .<= 0) + Wf1_D[k, j, i] .* (SfD[:, j] .> 0)

                    Ef_j = fxp.delta * UfD + fxp.lambdae * sum(v0mat .* (Af - repeat(Wfhat_D, outer=[1, kk])), dims=2) .* (SfD[:, j] .> 0) + (1 - fxp.phi - fxp.delta) * Wfhat_D
                    phi1f_D[k, j, i] = Wf1_D[k, j, i] - epsf[i] * alpha[j] - first(fxp.beta * gef_j' * Ef_j)
                end
            end
        end
    end
    #eqm wages phi2(y',y, x) -- when poaching firm triggers renegotiation
    #do not make JTJ transition: S(x, y')<S(x, y)

    Wm2_NC = zeros(Float64, kk, kk, ii)
    Wm2_YC = zeros(Float64, kk, kk, ii)
    Wm2_D = zeros(Float64, kk, kk, ii)
    Wf2_NC = zeros(Float64, kk, kk, ii)
    Wf2_YC = zeros(Float64, kk, kk, ii)
    Wf2_D = zeros(Float64, kk, kk, ii)
    phi2m_NC = zeros(Float64, kk, kk, ii)
    fill!(phi2m_NC, NaN)
    phi2m_YC = zeros(Float64, kk, kk, ii)
    fill!(phi2m_YC, NaN)
    phi2m_D = zeros(Float64, kk, kk, ii)
    fill!(phi2m_D, NaN)
    phi2f_NC = zeros(Float64, kk, kk, ii)
    fill!(phi2f_NC, NaN)
    phi2f_YC = zeros(Float64, kk, kk, ii)
    fill!(phi2f_YC, NaN)
    phi2f_D = zeros(Float64, kk, kk, ii)
    fill!(phi2f_D, NaN)

    for i = 1:ii
        #Male
        Sm_yP = repeat(SmNC[i, :], outer=[1, kk])
        Sm_yI = repeat(SmNC[i, :]', outer=[kk, 1])

        Wm2_NC[:, :, i] = (UmNC[i] * ones(kk, kk) + Sm_yP + par.sigma * (Sm_yI - Sm_yP)) .* (Sm_yP .<= Sm_yI) .* (Sm_yI .> 0) .* (Sm_yP .> 0)

        Sm_yP = repeat(SmYC[i, :], outer=[1, kk])
        Sm_yI = repeat(SmYC[i, :]', outer=[kk, 1])

        Wm2_YC[:, :, i] = (UmYC[i] * ones(kk, kk) + Sm_yP + par.sigma * (Sm_yI - Sm_yP)) .* (Sm_yP .<= Sm_yI) .* (Sm_yI .> 0) .* (Sm_yP .> 0)

        Sm_yP = repeat(SmD[i, :], outer=[1, kk])
        Sm_yI = repeat(SmD[i, :]', outer=[kk, 1])

        Wm2_D[:, :, i] = (UmD[i] * ones(kk, kk) + Sm_yP + par.sigma * (Sm_yI - Sm_yP)) .* (Sm_yP .<= Sm_yI) .* (Sm_yI .> 0) .* (Sm_yP .> 0)
        #Female
        Sf_yP = repeat(SfNC[i, :], outer=[1, kk])
        Sf_yI = repeat(SfNC[i, :]', outer=[kk, 1])

        Wf2_NC[:, :, i] = (UfNC[i] * ones(kk, kk) + Sf_yP + par.sigma * (Sf_yI - Sf_yP)) .* (Sf_yP .<= Sf_yI) .* (Sf_yI .> 0) .* (Sf_yP .> 0)

        Sf_yP = repeat(SfYC[i, :], outer=[1, kk])
        Sf_yI = repeat(SfYC[i, :]', outer=[kk, 1])

        Wf2_YC[:, :, i] = (UfYC[i] * ones(kk, kk) + Sf_yP + par.sigma * (Sf_yI - Sf_yP)) .* (Sf_yP .<= Sf_yI) .* (Sf_yI .> 0) .* (Sf_yP .> 0)

        Sf_yP = repeat(SfD[i, :], outer=[1, kk])
        Sf_yI = repeat(SfD[i, :]', outer=[kk, 1])

        Wf2_D[:, :, i] = (UfD[i] * ones(kk, kk) + Sf_yP + par.sigma * (Sf_yI - Sf_yP)) .* (Sf_yP .<= Sf_yI) .* (Sf_yI .> 0) .* (Sf_yP .> 0)
    end

    for i = 1:ii
        for j = 1:kk
            for k = 1:kk
                gem_j = gem[i,:,j];
                gef_j = gef[i,:,j];
                #NC
                #Male
                if Wm2_NC[k,j,i] != 0
                    vmI         = repeat(SmNC[:,j]+UmNC, outer=[1, kk]);  
                    vmP         = SmNC + repeat(UmNC, outer=[1, kk]);    
                    dm          = vmP-vmI;
    
                     
                    Am          = (vmP .>vmI).*((vmI+ par.sigma*(vmP-vmI)).*logit(dm, fxp) + vmI.*(1 .-logit(dm, fxp))) + (vmP .<=vmI).*(vmP .>Wm2_NC[k,j,i]).*( (vmP+ par.sigma*(vmI-vmP)).*(1 .-logit(dm, fxp)) + vmP.*logit(dm, fxp)) + (vmP .<=Wm2_NC[k,j,i]).*( Wm2_NC[k,j,i].*(1 .-logit(dm, fxp)) + vmP.*logit(dm, fxp));
    
                    Wmhat_NC    = UmNC.*(SmNC[:,j].<=0)+Wm2_NC[k,j,i].*(SmNC[:,j].>0);    
                    Wmhat_PL    = UmPL.*(SmPL[:,j].<=0)+Wm2_NC[k,j,i].*(SmPL[:,j].>0);    
                    Wmhat_D     = UmD.*(SmD[:,j].<=0)+Wm2_NC[k,j,i].*(SmD[:,j].>0);   
                    Em_j        = fxp.deltam_NC*UmNC + fxp.lambdae*sum(v0mat.*(Am-repeat(Wmhat_NC,outer=[1,kk])),dims=2).*(SmNC[:,j] .>0) + fxp.gamma*Wmhat_D + fxp.chi*Wmhat_PL + (1-fxp.deltam_NC-fxp.gamma-fxp.chi)*Wmhat_NC
                    phi2m_NC[k,j,i] = Wm2_NC[k,j,i] - epsm[i]*alpha[j]- first(fxp.beta*gem_j'*Em_j);
                end
                #Feale
                if Wf2_NC[k,j,i] != 0
                    vfI         = repeat(SfNC[:,j]+UfNC, outer=[1, kk]);  
                    vfP         = SfNC + repeat(UfNC, outer=[1, kk]);    
                    df          = vfP-vfI;
    
                             
                    Af          = (vfP .>vfI).*((vfI+ par.sigma*(vfP-vfI)).*logit(df, fxp) + vfI.*(1 .-logit(df, fxp))) + (vfP .<=vfI).*(vfP .>Wf2_NC[k,j,i]).*( (vfP+ par.sigma*(vfI-vfP)).*(1 .-logit(df, fxp)) + vfP.*logit(df, fxp)) + (vfP .<=Wf2_NC[k,j,i]).*( Wf2_NC[k,j,i].*(1 .-logit(df, fxp)) + vfP.*logit(df, fxp));
    
                    Wfhat_NC    = UfNC.*(SfNC[:,j].<=0)+Wf2_NC[k,j,i].*(SfNC[:,j].>0);    
                    Wfhat_PL    = UfPL.*(SfPL[:,j].<=0)+Wf2_NC[k,j,i].*(SfPL[:,j].>0);    
                    Wfhat_D     = UfD.*(SfD[:,j].<=0)+Wf2_NC[k,j,i].*(SfD[:,j].>0);   
                    Ef_j        = fxp.deltaf_NC*UfNC + fxp.lambdae*sum(v0mat.*(Af-repeat(Wfhat_NC,outer=[1,kk])),dims=2).*(SfNC[:,j] .>0) + fxp.gamma*Wfhat_D + fxp.chi*Wfhat_PL + (1-fxp.deltaf_NC-fxp.gamma-fxp.chi)*Wfhat_NC
                    phi2f_NC[k,j,i] = Wf2_NC[k,j,i] - epsf[i]*alpha[j]- first(fxp.beta*gef_j'*Ef_j)
                end
                
                #YC
                #Male
                if Wm2_YC[k,j,i] != 0
                    vmI         = repeat(SmYC[:,j]+UmYC, outer=[1, kk]);  
                    vmP         = SmYC + repeat(UmYC, outer=[1, kk]);    
                    dm          = vmP-vmI;
    
                     
                    Am          = (vmP .>vmI).*((vmI+ par.sigma*(vmP-vmI)).*logit(dm, fxp) + vmI.*(1 .-logit(dm, fxp))) + (vmP .<=vmI).*(vmP .>Wm2_YC[k,j,i]).*( (vmP+ par.sigma*(vmI-vmP)).*(1 .-logit(dm, fxp)) + vmP.*logit(dm, fxp)) + (vmP .<=Wm2_YC[k,j,i]).*( Wm2_YC[k,j,i].*(1 .-logit(dm, fxp)) + vmP.*logit(dm, fxp));
    
                    Wmhat_YC    = UmYC.*(SmYC[:,j].<=0)+Wm2_YC[k,j,i].*(SmYC[:,j].>0);    
                    Wmhat_PL    = UmPL.*(SmPL[:,j].<=0)+Wm2_YC[k,j,i].*(SmPL[:,j].>0);    
                    Wmhat_D     = UmD.*(SmD[:,j].<=0)+Wm2_YC[k,j,i].*(SmD[:,j].>0);   
                    Em_j        = fxp.deltam_YC*UmYC + fxp.lambdae*sum(v0mat.*(Am-repeat(Wmhat_YC,outer=[1,kk])),dims=2).*(SmNC[:,j] .>0) + fxp.gamma*Wmhat_D + fxp.chi*Wmhat_PL + (1-fxp.deltam_YC-fxp.gamma-fxp.chi)*Wmhat_YC
                    phi2m_YC[k,j,i] = Wm2_YC[k,j,i] - epsm[i]*alpha[j]- first(fxp.beta*gem_j'*Em_j)
                end
                
                #Female
                if Wf2_YC[k,j,i] != 0
                    vfI         = repeat(SfYC[:,j]+UfYC, outer=[1, kk]); 
                    vfP         = SfYC + repeat(UfYC, outer=[1, kk]);  
                    df          = vfP-vfI;
    
                     
                    Af          = (vfP .>vfI).*((vfI+ par.sigma*(vfP-vfI)).*logit(df, fxp) + vfI.*(1 .-logit(df, fxp))) + (vfP .<=vfI).*(vfP .>Wf2_YC[k,j,i]).*( (vfP+ par.sigma*(vfI-vfP)).*(1 .-logit(df, fxp)) + vfP.*logit(df, fxp)) + (vfP .<=Wf2_YC[k,j,i]).*( Wf2_YC[k,j,i].*(1 .-logit(df, fxp)) + vfP.*logit(df, fxp));
    
                    Wfhat_YC    = UfYC.*(SfYC[:,j].<=0)+Wf2_YC[k,j,i].*(SfYC[:,j].>0);    
                    Wfhat_PL    = UfPL.*(SfPL[:,j].<=0)+Wf2_YC[k,j,i].*(SfPL[:,j].>0);    
                    Wfhat_D     = UfD.*(SfD[:,j].<=0)+Wf2_YC[k,j,i].*(SfD[:,j].>0);   
                    Ef_j        = fxp.deltaf_YC*UfYC + fxp.lambdae*sum(v0mat.*(Af-repeat(Wfhat_YC,outer=[1,kk])),dims=2).*(SfNC[:,j] .>0) + fxp.gamma*Wfhat_D + fxp.chi*Wfhat_PL + (1-fxp.deltaf_YC-fxp.gamma-fxp.chi)*Wfhat_YC
                    phi2f_YC[k,j,i] = Wf2_YC[k,j,i] - (par.M+epsf[i])*alpha[j]- first(fxp.beta*gef_j'*Ef_j)
                end
                
                #D stage
                #Male
                if Wm2_D[k,j,i] != 0
                    vmI         = repeat(SmD[:,j]+UmD, outer=[1, kk]);  
                    vmP         = SmD + repeat(UmD, outer=[1, kk]);    
                    dm          = vmP-vmI;
    
                     
                    Am          = (vmP .>vmI).*((vmI+ par.sigma*(vmP-vmI)).*logit(dm, fxp) + vmI.*(1 .-logit(dm, fxp))) + (vmP .<=vmI).*(vmP .>Wm2_D[k,j,i]).*( (vmP+ par.sigma*(vmI-vmP)).*(1 .-logit(dm, fxp)) + vmP.*logit(dm, fxp)) + (vmP .<=Wm2_D[k,j,i]).*( Wm2_D[k,j,i].*(1 .-logit(dm, fxp)) + vmP.*logit(dm, fxp));
    
                    Wmhat_D     = UmD.*(SmD[:,j].<=0)+Wm2_D[k,j,i].*(SmD[:,j].>0);   
                    Em_j        = fxp.delta*UmD + fxp.lambdae*sum(v0mat.*(Am-repeat(Wmhat_D,outer=[1,kk])),dims=2).*(SmD[:,j] .>0) + (1-fxp.phi-fxp.delta)*Wmhat_D;
                    phi2m_D[k,j,i] = Wm2_D[k,j,i] - epsm[i]*alpha[j]- first(fxp.beta*gem_j'*Em_j)
                end
                
                #Female
                if Wf2_D[k,j,i] != 0
                    vfI         = repeat(SfD[:,j]+UfD, outer=[1, kk]); 
                    vfP         = SfD + repeat(UfD, outer=[1, kk]);  
                    df          = vfP-vfI;
    
                     
                    Af          = (vfP .>vfI).*((vfI+ par.sigma*(vfP-vfI)).*logit(df, fxp) + vfI.*(1 .-logit(df, fxp))) + (vfP .<=vfI).*(vfP .>Wf2_D[k,j,i]).*( (vfP+ par.sigma*(vfI-vfP)).*(1 .-logit(df, fxp)) + vfP.*logit(df, fxp)) + (vfP .<=Wf2_D[k,j,i]).*( Wf2_D[k,j,i].*(1 .-logit(df, fxp)) + vfP.*logit(df, fxp));
    
      
                    Wfhat_D     = UfD.*(SfD[:,j].<=0)+Wf2_D[k,j,i].*(SfD[:,j].>0);   
                    Ef_j        = fxp.delta*UfD + fxp.lambdae*sum(v0mat.*(Af-repeat(Wfhat_D,outer=[1,kk])),dims=2).*(SfD[:,j] .>0) + (1-fxp.phi-fxp.delta)*Wfhat_D;
    
                    phi2f_D[k,j,i] = Wf2_D[k,j,i] - epsf[i]*alpha[j]- first(fxp.beta*gef_j'*Ef_j)
                end
            end
        end
    end
    get!(e,"phi0m_NC",phi0m_NC);
    get!(e,"phi0m_PL",phi0m_PL);
    get!(e,"phi0m_YC",phi0m_YC);
    get!(e,"phi0m_D",phi0m_D);

    get!(e,"phi0f_NC",phi0f_NC);
    get!(e,"phi0f_PL",phi0f_PL);
    get!(e,"phi0f_YC",phi0f_YC);
    get!(e,"phi0f_D",phi0f_D);

    get!(e,"phi1m_NC",phi1m_NC);
    get!(e,"phi1m_YC",phi1m_YC);
    get!(e,"phi1m_D",phi1m_D);

    get!(e,"phi1f_NC",phi1f_NC);
    get!(e,"phi1f_YC",phi1f_YC);
    get!(e,"phi1f_D",phi1f_D);

    get!(e,"phi2m_NC",phi2m_NC);
    get!(e,"phi2m_YC",phi2m_YC);
    get!(e,"phi2m_D",phi2m_D);

    get!(e,"phi2f_NC",phi2f_NC);
    get!(e,"phi2f_YC",phi2f_YC);
    get!(e,"phi2f_D",phi2f_D);

    return e
end
export solve
end