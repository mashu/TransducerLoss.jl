# Numerically stable log(exp(a) + exp(b)); used in the transducer forward-backward.

"""
    logaddexp(a, b)

Numerically stable `log(exp(a) + exp(b))`. Internal; qualify as
`TransducerLoss.logaddexp` if needed.
"""
function logaddexp(a::T, b::T) where {T<:AbstractFloat}
    m = max(a, b)
    m == T(-Inf) && return T(-Inf)
    m + log(exp(a - m) + exp(b - m))
end
