Assist in rewriting this prompt via @prompt-engineer into a comprehensive, complete and easy to follow set of requirements for a long running agent task.

Our goal is to run REA / rea (https://github.com/morluto/rea) which is available in local skills to re-implement the https://github.com/peonist-ai/halogen-flash-server binary blobs.

To do this, we must leverage comprehensive testing and validation against the existing implementations. The language or reimplementation MUST be rust. The repository MUST leverage a nix flake to handle all development tools and tests.

Analyse these requirements, reorder the steps to ensure that nix is leveraged as a core requirement and rust is the reimplementation language, take the packaged nix version of halogen as per /home/jay/dev/personal/nix-config/packages/other/halogen-flash-server and copy it to the current flake, then offer both the halogen package as well as cold cathode as packages.

Testing must validate all CLI calls made to halogen match the calls to cold cathode in terms of UX, behaviour and outcomes. Halogen is written in cpp therefore is may be required we leverage unsafe rust. This must be kept to a minimum, thoroughly tested and clearly described.

The final package must describe clearly via a .github/README.md what this is, noting the excellent work of halogen and offering that this package is simply a validation of local model capability, rust implementation and agentic flows and skills.

If there is a lack of clarity in this doc that blocks the agent from rewriting it into full-plan.md, ask the user for clarifications, never assume in this task.

The child task must never assume, it may be worth a brainstorming / superpowers skill invoke to ensure alignment of outcomes between this task and the performed work.

All work must be done via git worktrees, all code must converge on an `init` branch to be reviewed.