using Documenter
using TransducerLoss

makedocs(;
    modules = [TransducerLoss],
    authors = "Mateusz Kaduk <mateusz.kaduk@gmail.com>",
    repo = "https://github.com/mashu/TransducerLoss.jl/blob/{commit}{path}{line}",
    sitename = "TransducerLoss.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://mashu.github.io/TransducerLoss.jl",
        edit_link = "main",
        repolink = "https://github.com/mashu/TransducerLoss.jl",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Choosing a loss" => "variants.md",
        "Examples" => "examples.md",
        "API" => "api.md",
    ],
    checkdocs = :exports,
)

deploydocs(;
    repo = "github.com/mashu/TransducerLoss.jl.git",
    devbranch = "main",
    push_preview = true,
)
