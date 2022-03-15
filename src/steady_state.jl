module steady_state
function logit(x, fixedParam)
    return 1 ./ (1.0 .+ exp.(-fixedParam.lambda .* x))
end
#HC evolution function, g for gender, a for age 
function geth_I(x,y,geg,hga)
    #gea is the probability of hc growth for gender g
    #hga is the human capital distribution of gender g in stage a
    #get the job productivity level from y, an integer from 1 to 7
    jobLevel = y%7 == 0 ? 7 : y%7
    
    hga_I_xy = hga[x,y]*(1-geg[jobLevel])
    #check human capital level, if 1, human capital measure decreases
    if(x%7==1)
        return hga_I_xy
    elseif(x%7==0)#check human capital level, if 0, human capital measure increases
        hga_I_xy = hga[x,y] + hga[x-1,y]*geg[jobLevel]
    else #check human capital level, there's flowin and flow out
        hga_I_xy = hga_I_xy + hga[x-1,y]*geg[jobLevel]
    end
    return hga_I_xy
end
#Searching and Matching Stage
#Component of change that is attributable to change in employment 
#A for age
#g for gender
function getEmpChange(A,i,j,ugA_I,SgA,hgA_I,fxp,v)
    dgA = -SgA[i,:] .+ SgA[i,j];
    IgA = Int.(dgA .!= 0);
    #determine the age and the associated parameter
    if (cmp(A,"NC")==0)
        lambdau_a = fxp.lambdau_NC;
    else
        lambdau_a = fxp.lambdau_YC;
    end
    part1 = lambdau_a*ugA_I[i]*v[j];
    part2 = fxp.lambdae*v[j]*sum(hgA_I[i,:].*IgA.*logit(dgA, fxp).*(SgA[i,:].>0));
    part3 = -fxp.lambdae*hgA_I[i,j]*sum(v.*IgA.*logit(-dgA, fxp));
    return part1+part2+part3;
end
#get the change in distribution for human capital at NC
function get_hgNC_II_ij(g,i,j,ugNC_I,SgNC,hgNC_I,fxp,v)
    empChange = getEmpChange("NC",i,j,ugNC_I,SgNC,hgNC_I,fxp,v);
     #determine the gender and the associated parameters
    deltag_NC =  g == "m" ? fxp.deltam_NC : fxp.deltaf_NC;
    return hgNC_I[i,j]+(SgNC[i,j]>0)*(empChange)- hgNC_I[i,j]*(fxp.gamma + fxp.chi + deltag_NC);
end
#get the change in distribution for human capital at YC
function get_hgYC_II_ij(g,i,j,ugYC_I,SgYC,hgYC_I,hgPL_I,fxp,v)
    empChange = getEmpChange("YC",i,j,ugYC_I,SgYC,hgYC_I,fxp,v);
    #determine the gender and the associated parameters
    deltag_YC =  g == "m" ? fxp.deltam_YC : fxp.deltaf_YC;
    etag =  g == "m" ? fxp.etam : fxp.etaf;
    return hgYC_I[i,j]+etag*hgPL_I[i,j]+(SgYC[i,j]>0)*(empChange)- hgYC_I[i,j]*(fxp.gamma + deltag_YC + fxp.chi);
end
#get the change in distribution for human capital at D
function get_hgD_II_ij(i,j,SgD,ugD_I,hgNC_I,hgYC_I,hgPL_I,hgD_I,fxp,v)
    empChange = getEmpChange("D",i,j,ugD_I,SgD,hgD_I,fxp,v);
    return hgD_I[i,j]+ fxp.gamma*(hgNC_I[i,j]+ hgYC_I[i,j]+hgPL_I[i,j]) + (SgD[i,j]>0)*(empChange) - hgD_I[i,j]*(fxp.phi + fxp.delta);
end
export get_hgD_II_ij,get_hgYC_II_ij,get_hgNC_II_ij,geth_I
end