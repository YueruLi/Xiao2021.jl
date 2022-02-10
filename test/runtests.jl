using Xiao2021
using Test

@testset "foo check" begin
   @test foo(0) < 1E-16
end
