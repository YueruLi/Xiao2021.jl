module solve_eqm
# Write your package code here.
using NLsolve

#Calculates the Sigmoid 
function logit(x, fixedParam)
    return 1 ./ (1.0 .+ exp.(-fixedParam.lambda .* x))
end

#Solves for steady state distribution of workforce across age stages
function solveStages(var, fixedParam)
    NC_f    = var[1];
    YC_f    = var[2];
    PL_f    = var[3];
    D_f     = var[4];

    NC_m    = var[5];
    YC_m    = var[6];
    PL_m    = var[7];
    D_m     = var[8];


    # 4 equations for women
    eq1f = NC_f + YC_f + PL_f + D_f - 0.5;
    eq2f = fixedParam.chi*(NC_f+YC_f) - (fixedParam.etaf+fixedParam.gamma)*PL_f; # flows into and out of PL
    eq3f = fixedParam.etaf*PL_f - (fixedParam.chi+fixedParam.gamma)*YC_f; # flows into and out of YC
    eq4f = fixedParam.gamma*(NC_f+YC_f+PL_f) - fixedParam.phi*D_f;

    # 4 equations for men
    eq1m = NC_m + YC_m + PL_m + D_m - 0.5;
    eq2m = fixedParam.chi*(NC_m+YC_m) - (fixedParam.etam+fixedParam.gamma)*PL_m; # flows into and out of PL
    eq3m = fixedParam.etam*PL_m - (fixedParam.chi+fixedParam.gamma)*YC_m; # flows into and out of YC
    eq4m = fixedParam.gamma*(NC_m+YC_m+PL_m) - fixedParam.phi*D_m;

    return [eq1f, eq2f, eq3f, eq4f, eq1m, eq2m, eq3m, eq4m];
end

#Functions for updating male value functions
#Male No Child
function getPmNC(x, y,v0,fxp,par,Pi,gem,SmNC,UmNC,PmD_hat,PmPL_hat,PmNC_hat,epsm,alpha,output,tau)
    expectedPmNC = 0.0
    dmNC1 = SmNC[x, :] .- SmNC[x, y]
    #Calculate the value function if no hc increment in the next round
    val1 = fxp.deltam_NC * (Pi[y] + UmNC[x]) +
           fxp.lambdae * par.sigma * sum(v0 .* dmNC1 .* logit(dmNC1, fxp)) * (SmNC[x, y] > 0) +
           fxp.gamma * PmD_hat[x, y] + fxp.chi * PmPL_hat[x, y] +
           (1 - fxp.deltam_NC - fxp.gamma - fxp.chi) * PmNC_hat[x, y]
    #Check if it is possible for HC to grow
    if (x % 7 == 0)
        expectedPmNC = val1
    else
        dmNC2 = SmNC[x+1, :] .- SmNC[x+1, y]
        #Calculate the value function if hc increments in the next round
        val2 = fxp.deltam_NC * (Pi[y] + UmNC[x+1]) +
               fxp.lambdae * par.sigma * sum(v0 .* dmNC2 .* logit(dmNC2, fxp)) * (SmNC[x+1, y] > 0) +
               fxp.gamma * PmD_hat[x+1, y] + fxp.chi * PmPL_hat[x+1, y] +
               (1 - fxp.deltam_NC - fxp.gamma - fxp.chi) * PmNC_hat[x+1, y]
        #weight the expected value by probability of HC growth
        if (y % 7 == 0)
            expectedPmNC = gem[7] * val2 + (1 - gem[7]) * val1
        else
            expectedPmNC = gem[y%7] * val2 + (1 - gem[y%7]) * val1
        end
    end
    #return sum of value at current round and expected value for next perioid 
    return (1 - tau[1]) * output[x, y] + epsm[x] * alpha[y] + fxp.beta * expectedPmNC
end

#Male Parental Leave
function getPmPL(x, y,Pi,fxp,UmPL,PmD_hat,PmYC_hat,PmPL_hat,epsm,alpha,output)
    val = fxp.deltam_YC * (Pi[y] + UmPL[x]) +
          fxp.gamma * PmD_hat[x, y] +
          fxp.etam * PmYC_hat[x, y] +
          (1 - fxp.deltam_YC - fxp.gamma - fxp.etam) * PmPL_hat[x, y]

    return fxp.Transfer * output[x, y] + epsm[x] * alpha[y] + fxp.beta * val
end

#Male Young Child
function getPmYC(x, y,v0,Pi,fxp,par,epsm,gem,tau,output,alpha,UmYC,SmYC,PmD_hat,PmPL_hat,PmYC_hat)
    expectedPmYC = 0.0
    dmYC1 = SmYC[x, :] .- SmYC[x, y]
    #Calculate the value function if no hc increment in the next round
    val1 = fxp.deltam_YC * (Pi[y] + UmYC[x]) +
           fxp.lambdae * par.sigma * sum(v0 .* dmYC1 .* logit(dmYC1, fxp)) * (SmYC[x, y] > 0) +
           fxp.gamma * PmD_hat[x, y] + fxp.chi * PmPL_hat[x, y] +
           (1 - fxp.deltam_YC - fxp.gamma - fxp.chi) * PmYC_hat[x, y]
    #Check if it is possible for HC to grow
    if (x % 7 == 0)
        expectedPmYC = val1
    else
        dmYC2 = SmYC[x+1, :] .- SmYC[x+1, y]
        #Calculate the value function if hc increments in the next round
        val2 = fxp.deltam_YC * (Pi[y] + UmYC[x+1]) +
               fxp.lambdae * par.sigma * sum(v0 .* dmYC2 .* logit(dmYC2, fxp)) * (SmYC[x+1, y] > 0) +
               fxp.gamma * PmD_hat[x+1, y] + fxp.chi * PmPL_hat[x+1, y] +
               (1 - fxp.deltam_YC - fxp.gamma - fxp.chi) * PmYC_hat[x+1, y]
        if (y % 7 == 0)
            expectedPmYC = gem[7] * val2 + (1 - gem[7]) * val1
        else
            expectedPmYC = gem[y%7] * val2 + (1 - gem[y%7]) * val1
        end
    end
    return (1 - tau[1]) * output[x, y] + epsm[x] * alpha[y] + fxp.beta * expectedPmYC
end

#Male Done with Children
function getPmD(x,y,v0,Pi,fxp,par,output,gem,tau,epsm,alpha,SmD,PmD_hat,UmD)
    expectedPmD = 0.0
    dmD1 = SmD[x, :] .- SmD[x, y]
    #Calculate the value function if no hc increment in the next round
    val1 = fxp.phi * Pi[y] + fxp.delta * (Pi[y] + UmD[x]) +
           (1 - fxp.phi - fxp.delta) * PmD_hat[x, y] +
           fxp.lambdae * par.sigma * sum(v0 .* dmD1 .* logit(dmD1, fxp)) * (SmD[x, y] > 0)
    #Check if it is possible for HC to grow
    if (x % 7 == 0)
        expectedPmD = val1
    else
        dmD2 = SmD[x+1, :] .- SmD[x+1, y]
        #Calculate the value function if hc increments in the next round
        val2 = fxp.phi * Pi[y] + fxp.delta * (Pi[y] + UmD[x+1]) +
               (1 - fxp.phi - fxp.delta) * PmD_hat[x+1, y] +
               fxp.lambdae * par.sigma * sum(v0 .* dmD2 .* logit(dmD2, fxp)) * (SmD[x+1, y] > 0)
        if (y % 7 == 0)
            expectedPmD = gem[7] * val2 + (1 - gem[7]) * val1
        else
            expectedPmD = gem[y%7] * val2 + (1 - gem[y%7]) * val1
        end
    end
    return (1 - tau[1]) * output[x, y] + epsm[x] * alpha[y] + fxp.beta * expectedPmD
end

#Functions for updating female value functions
#Female No Child
function getPfNC(x, y,v0,fxp,par,Pi,gef,SfNC,UfNC,PfD_hat,PfPL_hat,PfNC_hat,epsf,alpha,output,tau)
    expectedPfNC = 0.0
    dfNC1 = SfNC[x, :] .- SfNC[x, y]
    #Calculate the value function if no hc increment in the next round
    val1 = fxp.deltaf_NC * (Pi[y] + UfNC[x]) +
           fxp.lambdae * par.sigma * sum(v0 .* dfNC1 .* logit(dfNC1, fxp)) * (SfNC[x, y] > 0) +
           fxp.gamma * PfD_hat[x, y] + fxp.chi * PfPL_hat[x, y] +
           (1 - fxp.deltaf_NC - fxp.gamma - fxp.chi) * PfNC_hat[x, y]
    #Check if it is possible for HC to grow
    if (x % 7 == 0)
        expectedPfNC = val1
    else
        dfNC2 = SfNC[x+1, :] .- SfNC[x+1, y]
        #Calculate the value function if hc increments in the next round
        val2 = fxp.deltaf_NC * (Pi[y] + UfNC[x+1]) +
               fxp.lambdae * par.sigma * sum(v0 .* dfNC2 .* logit(dfNC2, fxp)) * (SfNC[x+1, y] > 0) +
               fxp.gamma * PfD_hat[x+1, y] + fxp.chi * PfPL_hat[x+1, y] +
               (1 - fxp.deltaf_NC - fxp.gamma - fxp.chi) * PfNC_hat[x+1, y]
        if (y % 7 == 0)
            expectedPfNC = gef[7] * val2 + (1 - gef[7]) * val1
        else
            expectedPfNC = gef[y%7] * val2 + (1 - gef[y%7]) * val1
        end
    end
    #return value at current round and expected value for next perioid 
    return (1 - tau[1]) * output[x, y] + epsf[x] * alpha[y] + fxp.beta * expectedPfNC
end

#Female Parental Leave
function getPfPL(x, y,Pi,par,fxp,UfPL,PfD_hat,PfYC_hat,PfPL_hat,epsf,alpha,output)
    val = fxp.deltaf_YC * (Pi[y] + UfPL[x]) +
          fxp.gamma * PfD_hat[x, y] +
          fxp.etaf * PfYC_hat[x, y] +
          (1 - fxp.deltaf_YC - fxp.gamma - fxp.etaf) * PfPL_hat[x, y]

    return fxp.Transfer * output[x, y] + (par.M + epsf[x]) * alpha[y] + fxp.beta * val
end

#Female Young Child 
function getPfYC(x, y,v0,Pi,fxp,par,epsf,gef,tau,output,alpha,UfYC,SfYC,PfD_hat,PfPL_hat,PfYC_hat)
    expectedPfYC = 0.0
    dfYC1 = SfYC[x, :] .- SfYC[x, y]
    #Calculate the value function if no hc increment in the next round
    val1 = fxp.deltaf_YC * (Pi[y] + UfYC[x]) +
           fxp.lambdae * par.sigma * sum(v0 .* dfYC1 .* logit(dfYC1, fxp)) * (SfYC[x, y] > 0) +
           fxp.gamma2 * PfD_hat[x, y] + fxp.chi * PfPL_hat[x, y] +
           (1 - fxp.deltaf_YC - fxp.gamma - fxp.chi) * PfYC_hat[x, y]
    #Check if it is possible for HC to grow
    if (x % 7 == 0)
        expectedPfYC = val1
    else
        dfYC2 = SfYC[x+1, :] .- SfYC[x+1, y]
        #Calculate the value function if hc increments in the next round
        val2 = fxp.deltaf_YC * (Pi[y] + UfYC[x+1]) +
               fxp.lambdae * par.sigma * sum(v0 .* dfYC2 .* logit(dfYC2, fxp)) * (SfYC[x+1, y] > 0) +
               fxp.gamma2 * PfD_hat[x+1, y] + fxp.chi * PfPL_hat[x+1, y] +
               (1 - fxp.deltaf_YC - fxp.gamma - fxp.chi) * PfYC_hat[x+1, y]
        if (y % 7 == 0)
            expectedPfYC = gef[7] * val2 + (1 - gef[7]) * val1
        else
            expectedPfYC = gef[y%7] * val2 + (1 - gef[y%7]) * val1
        end
    end
    return (1 - tau[1]) * output[x, y] + (par.M + epsf[x]) * alpha[y] + fxp.beta * expectedPfYC
end



#Female Done with Children
function getPfD(x,y,v0,Pi,fxp,par,output,gef,tau,epsf,alpha,SfD,PfD_hat,UfD)
    expectedPfD = 0.0
    dfD1 = SfD[x, :] .- SfD[x, y]
    #Calculate the value function if no hc increment in the next round
    val1 = fxp.phi * Pi[y] + fxp.delta * (Pi[y] + UfD[x]) +
           (1 - fxp.phi - fxp.delta) * PfD_hat[x, y] +
           fxp.lambdae * par.sigma * sum(v0 .* dfD1 .* logit(dfD1, fxp)) * (SfD[x, y] > 0)
    #Check if it is possible for HC to grow
    if (x % 7 == 0)
        expectedPfD = val1
    else
        dfD2 = SfD[x+1, :] .- SfD[x+1, y]
        #Calculate the value function if hc increments in the next round
        val2 = fxp.phi * Pi[y] + fxp.delta * (Pi[y] + UfD[x+1]) +
               (1 - fxp.phi - fxp.delta) * PfD_hat[x+1, y] +
               fxp.lambdae * par.sigma * sum(v0 .* dfD2 .* logit(dfD2, fxp)) * (SfD[x+1, y] > 0)
        if (y % 7 == 0)
            expectedPfD = gef[7] * val2 + (1 - gef[7]) * val1
        else
            expectedPfD = gef[y%7] * val2 + (1 - gef[y%7]) * val1
        end
    end
    return (1 - tau[1]) * output[x, y] + epsf[x] * alpha[y] + fxp.beta * expectedPfD
end


#Solve for equilibrium
function solve(par, set, fxp, init)
    kk = set.Ny * set.Na
    ii = set.Nx * set.Ne

    epsf = repeat([-2, 0, 2] .* fxp.sdf .+ par.muf * fxp.mum, inner = 7)
    epsm = repeat([-2, 0, 2] .* fxp.sdm .+ fxp.mum, inner = 7)
    alpha = repeat(fxp.alpha,inner=7);
    #humanCapital = repeat(fxp.hc, outer = 3);
    #jobProductivityType = repeat(fxp.p,outer=3);
    #jobAmenityType = repeat(fxp.alpha,inner=7);
    output = par.K .* [(par.a * hc^fxp.rho + (1 - par.a) * p^fxp.rho)^(1 / fxp.rho) for hc in repeat(fxp.hc, outer = 3), p in repeat(fxp.p, outer = 3)]
    outputb = repeat([par.b * hc for hc in fxp.hc], outer = 3)
    e = Dict("stages" => nlsolve(x -> solveStages(x, fxp), 12.50 .* ones(typeof(12.50), 8)).zero, "converge" => 1, "Pineg" => 0)
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
    effU = sum(umNC_0 + ufNC_0 + fxp.s1 * (umYC_0 + umD_0 + ufYC_0 + ufD_0)) +
           fxp.s2 * sum(sum(hmNC_0 + hmYC_0 + hmD_0 + hfNC_0 + hfYC_0 + hfD_0, dims = 1))
    V = sum(v0)
    fxp = merge(fxp, (lambdau_NC = fxp.theta / (effU * V)^0.5, lambdau_YC = fxp.theta / (effU * V)^0.5 * fxp.s1, lambdae = fxp.theta / (effU * V)^0.5 * fxp.s2))
    maxit = 10000
    change_dist = zeros(maxit, 1)
    tol = 1e-9
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
        SmYC = SmYC_0
        SmPL = SmPL_0
        SmD = SmD_0
        SfNC = SfNC_0
        SfYC = SfYC_0
        SfPL = SfPL_0
        SfD = SfD_0
    end
    return PmNC
end

export solve
end