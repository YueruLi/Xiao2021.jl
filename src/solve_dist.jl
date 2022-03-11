module solve_dist

include("steady_state.jl")
using .steady_state

function logit(x, fixedParam)
    return 1 ./ (1.0 .+ exp.(-fixedParam.lambda .* x))
end
function solveDist(x0, set, fxp, par, SmNC, SmPL, SmYC, SmD, SfNC, SfPL, SfYC, SfD, stages)
    #setting up
    x = copy(x0);
    v = x[length(x)-set.Ny*set.Na+1:length(x)];
    x = deleteat!(x, length(x)-set.Ny*set.Na+1:length(x));
    x = reshape(x,set.Nx*set.Ne,:);
    umNC    = x[:, 1];
    umPL    = x[:,2];
    umYC    = x[:,3];
    umD     = x[:,4];
    ufNC    = x[:,5];
    ufPL    = x[:,6];
    ufYC    = x[:,7];
    ufD     = x[:,8];
    hmNC    = x[:,9:29];
    hmYC    = x[:,30:50];
    hmPL    = x[:,51:71];
    hmD     = x[:,72:92];
    hfNC    = x[:,93:113];
    hfYC    = x[:,114:134];
    hfPL    = x[:,135:155];
    hfD     = x[:,156:176];
    D_f     = stages[4];
    D_m     = stages[8];
    gem = par.d1m .+ par.d2m .* fxp.p;
    gef = par.d1f .+ par.d2f .* fxp.p;
    #HC evoluation Stage
    #Male
    
    umNC_I  = umNC;
    umPL_I  = umPL;
    umYC_I  = umYC;
    umD_I   = umD;
    hmNC_I = [geth_I(x,y,gem,hmNC) for x in 1:21, y in 1:21];
    hmPL_I  = hmPL;
    hmYC_I = [geth_I(x,y,gem,hmYC) for x in 1:21, y in 1:21];
    hmD_I = [geth_I(x,y,gem,hmD) for x in 1:21, y in 1:21];
    #Female
    ufNC_I  = ufNC;
    ufPL_I  = ufPL;
    ufYC_I  = ufYC;
    ufD_I   = ufD;
    hfNC_I = [geth_I(x,y,gef,hfNC) for x in 1:21, y in 1:21];
    hfYC_I = [geth_I(x,y,gef,hfYC) for x in 1:21, y in 1:21];
    hfPL_I  = hfPL;
    hfD_I = [geth_I(x,y,gef,hfD) for x in 1:21, y in 1:21];

    #(II) Searching and Matching Stage
    #initial distribution of (x,ϵ)
    initm = vec([x*y for x in fxp.phi0m, y in fxp.eden]);
    initf = vec([x*y for x in fxp.phi0f, y in fxp.eden]);

    umNC_II = [umNC_I[i]*(1 - fxp.gamma -fxp.chi- fxp.lambdau_NC*sum(v.*(SmNC[i,:].>0)))+fxp.phi*D_m*initm[i]+ fxp.deltam_NC*sum(hmNC_I[i,:]) for i in 1:21];
    ufNC_II = [ufNC_I[i]*(1 - fxp.gamma -fxp.chi- fxp.lambdau_NC*sum(v.*(SfNC[i,:].>0)))+fxp.phi*D_f*initf[i]+ fxp.deltaf_NC*sum(hfNC_I[i,:]) for i in 1:21];

    umPL_II = [umPL_I[i]*(1 - fxp.gamma -fxp.etam)+ fxp.chi*(umNC_I[i]+umYC_I[i]) + fxp.deltam_YC*sum(hmPL_I[i,:]) for i in 1:21];
    ufPL_II = [ufPL_I[i]*(1 - fxp.gamma -fxp.etaf)+ fxp.chi*(ufNC_I[i]+ufYC_I[i]) + fxp.deltaf_YC*sum(hfPL_I[i,:]) for i in 1:21];

    umYC_II = [umYC_I[i]*(1 - fxp.gamma -fxp.chi - fxp.lambdau_YC*sum(v.*(SmYC[i,:].>0)))+ fxp.etam*umPL_I[i] + fxp.deltam_YC*sum(hmYC_I[i,:]) for i in 1:21];   
    ufYC_II = [ufYC_I[i]*(1 - fxp.gamma -fxp.chi - fxp.lambdau_YC*sum(v.*(SfYC[i,:].>0)))+ fxp.etaf*ufPL_I[i] + fxp.deltaf_YC*sum(hfYC_I[i,:]) for i in 1:21];   

    umD_II = [umD_I[i]*(1 - fxp.phi - fxp.lambdau_YC*sum(v.*(SmD[i,:].>0)) )+ fxp.gamma*umNC_I[i] + fxp.gamma*umYC_I[i] + fxp.delta*sum(hmD_I[i,:]) for i in 1:21];
    ufD_II = [ufD_I[i]*(1 - fxp.phi - fxp.lambdau_YC*sum(v.*(SfD[i,:].>0)) )+ fxp.gamma*ufNC_I[i] + fxp.gamma*ufYC_I[i] + fxp.delta*sum(hfD_I[i,:]) for i in 1:21];

    hmNC_II = [get_hgNC_II_ij("m",i,j,umNC_I,SmNC,hmNC_I,fxp,v) for i in 1:21, j in 1:21];

    hmYC_II = [get_hgYC_II_ij("m",i,j,umYC_I,SmYC,hmYC_I,hmPL_I,fxp,v) for i in 1:21, j in 1:21];

    hmPL_II = [hmPL_I[i,j]+fxp.chi*(hmNC_I[i,j] + hmYC_I[i,j])- hmPL_I[i,j]*(fxp.gamma + fxp.deltam_YC + fxp.etam) for i in 1:21, j in 1:21];

    hmD_II = [get_hgD_II_ij(i,j,SmD,umD_I,hmNC_I,hmYC_I,hmPL_I,hmD_I,fxp,v) for i in 1:21, j in 1:21];

    hfNC_II = [get_hgNC_II_ij("f",i,j,ufNC_I,SfNC,hfNC_I,fxp,v) for i in 1:21, j in 1:21];

    hfYC_II = [get_hgYC_II_ij("f",i,j,ufYC_I,SfYC,hfYC_I,hfPL_I,fxp,v) for i in 1:21, j in 1:21];

    hfPL_II = [hfPL_I[i,j]+fxp.chi*(hfNC_I[i,j] + hfYC_I[i,j])- hfPL_I[i,j]*(fxp.gamma + fxp.deltaf_YC + fxp.etaf) for i in 1:21, j in 1:21];

    hfD_II = [get_hgD_II_ij(i,j,SfD,ufD_I,hfNC_I,hfYC_I,hfPL_I,hfD_I,fxp,v) for i in 1:21, j in 1:21];

    # Endogenous Quites
    # after endogenous quits, all measures must equal measures at the beginning
    umNCIII  = vec(umNC_II + sum(hmNC_II.*(SmNC .<=0), dims=2));
    umPLIII  = vec(umPL_II + sum(hmPL_II.*(SmPL .<=0),dims= 2));
    umYCIII  = vec(umYC_II + sum(hmYC_II.*(SmYC .<=0),dims= 2));
    umDIII   = vec(umD_II + sum(hmD_II.*(SmD .<=0), dims=2));
    ufNCIII  = vec(ufNC_II + sum(hfNC_II.*(SfNC .<=0),dims= 2));
    ufPLIII  = vec(ufPL_II + sum(hfPL_II.*(SfPL .<=0),dims= 2));
    ufYCIII  = vec(ufYC_II + sum(hfYC_II.*(SfYC .<=0),dims= 2));
    ufDIII   = vec(ufD_II + sum(hfD_II.*(SfD .<=0),dims= 2));
    hmNCIII  = hmNC_II - hmNC_II.*(SmNC .<=0);
    hmYCIII  = hmYC_II - hmYC_II.*(SmYC .<=0);
    hmPLIII  = hmPL_II - hmPL_II.*(SmPL .<=0);
    hmDIII   = hmD_II - hmD_II.*(SmD .<=0);
    hfNCIII  = hfNC_II - hfNC_II.*(SfNC .<=0);
    hfYCIII  = hfYC_II - hfYC_II.*(SfYC .<=0);
    hfPLIII  = hfPL_II - hfPL_II.*(SfPL .<=0);
    hfDIII   = hfD_II - hfD_II.*(SfD .<=0);

    #calculate new vacancies
    vnew    = fxp.gamma_yα - vec(sum(hmNC,dims=1)) - vec(sum(hmYC,dims=1)) -vec(sum(hmPL,dims=1)) - vec(sum(hmD,dims=1)) - vec(sum(hfNC,dims=1)) - vec(sum(hfYC,dims=1)) -vec(sum(hfPL,dims=1)) - vec(sum(hfD,dims=1));
    result = vcat(umNCIII, umPLIII, umYCIII, umDIII, ufNCIII, ufPLIII, ufYCIII, ufDIII, vec(hmNCIII), vec(hmYCIII), vec(hmPLIII), vec(hmDIII),vec(hfNCIII), vec(hfYCIII), vec(hfPLIII), vec(hfDIII),vnew);

    return result;
end
export solveDist
end



