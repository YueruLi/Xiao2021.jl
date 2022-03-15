using Xiao2021
using Test
using


@testset "solve_eqm" begin
    # Write your tests here.
    set = getSettings();
    par = getParams();
    fxp = getFixParameters();
    init = getInit();
    e = solve_eqm.solve(par,set,fxp,init);
    @test e
end