# EdDynamics result reproduction

Replace every `/path/to/...` placeholder before running a command.

## Layout

```text
s1_empirical_dynamics/
├── main/
├── scripts/
└── robustness/
s2_minimal_mechanism/
├── main/
├── scripts/
└── robustness/
s3_event_ssl/
├── main/
│   ├── controls/
│   └── structural_analyses/
├── scripts/
└── robustness/
s4_cross_analysis/
├── main/
├── scripts/
└── robustness/
s5_supplementary/
└── scripts/
reference_outputs/
```

The `s1`–`s5` directory prefixes are navigation labels. In contrast, `stage*` prefixes retained in code filenames and result directories denote the actual workflow order.

`main/` contains primary experiments and downstream analyses, `robustness/` contains additional robustness workflows and audits, and `scripts/` contains result extraction and figure commands.

See [reference outputs](reference_outputs/README.md) for core numeric reports and supplementary report-generation scripts.

## Dependency order

1. Generate the empirical outputs in `s1_empirical_dynamics/main/`.
2. Run both construction-matched null analyses and empirical sensitivity from `s1_empirical_dynamics/robustness/`.
3. Run the complete ordered mechanism sequence in `s2_minimal_mechanism/main/`.
4. Run primary Event-SSL, its controls, macro sufficiency and representation geometry using `s3_event_ssl/README.md`.
5. Run the five additional Event-SSL seeds, then the state-only and objective-control workflows.
6. Complete the remaining empirical and mechanism robustness workflows.
7. Run `s4_cross_analysis/README.md`.
8. Run the `scripts/README.md` inside each component when its stated inputs exist; run `s5_supplementary/scripts/README.md` last.

Resume a wrapper with the same result root when it provides completion markers. Do not combine outputs generated from different code copies in one result root.
