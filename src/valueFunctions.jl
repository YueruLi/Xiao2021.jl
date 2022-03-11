module valueFunctions
function logit(x, fixedParam)
    return 1 ./ (1.0 .+ exp.(-fixedParam.lambda .* x))
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
export getPmNC,getPmYC,getPmPL,getPmD,getPfNC,getPfYC,getPfPL,getPfD
end