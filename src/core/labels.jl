# Target packing for the transducer lattice (CPU-side); shared by every loss in this package.

"""
    pack_transducer_targets(targets, blank) → (labels, target_lengths)

Pack ragged label sequences into a zero-padded `(Umax, B)` Int32 matrix plus
`Vector{Int32}` lengths. Targets must not contain `blank` (the lattice
reserves it for the advance-time transition) — throws `ArgumentError` if they
do.
"""
function pack_transducer_targets(targets::Vector{Vector{Int}}, blank::Int)
    B = length(targets)
    target_lengths = Int32[length(t) for t in targets]
    Umax = Int(maximum(target_lengths; init = Int32(0)))
    labels = zeros(Int32, max(Umax, 1), B)
    for b in 1:B
        for (u, tok) in enumerate(targets[b])
            tok == blank && throw(ArgumentError(
                "targets must not contain the blank index $blank"))
            tok >= 1 || throw(ArgumentError("labels must be ≥ 1, got $tok"))
            labels[u, b] = Int32(tok)
        end
    end
    labels, target_lengths
end
