module helper
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
export logit, solveStages
end