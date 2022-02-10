module Xiao2021
# Write your package code here.
using Expectations, Distributions

function foo(μ = 1., σ = 2.)
    println("Modified foo definition")
    d = Normal(μ, σ)
    E = expectation(d)
    return E(x -> sin(x))
end
export foo
end
