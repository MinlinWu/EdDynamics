# Numerical source records

The [correction record](Numerical_correction_record.json) gives the previously reported and corrected values by supplementary table. It covers the saved residence-time draws (Table 4b), six-seed neural comparisons (Tables 7 and 8e) and interval rounding (Table 11a). The recorded archive SHA-256 identifies the source snapshot used in verification. Individual public files have their own digests in the [source manifest](../supplementary_source_manifest.json).

| Verification record | How to read it |
| --- | --- |
| [Initial comparison for Table 7](table7_initial_comparison.json) | Thirteen of 21 reported summaries matched the initial check. The eight retention summaries that differed are listed with their earlier values. The source list in this record identifies the twelve Stage 5 input CSV files. |
| [Table 7 check with the original scoring functions](table7_original_scoring_check.json) | Gives component scores for each of six seeds and the recalculated retention summaries. |
| [Table 7 retention check from the six source files](table7_retention_source_check.json) | Applies the original scoring functions and checks the eight corrected rows against an independent calculation. The six entries in each summary are the mean, sample standard deviation, lower and upper 95% Student’s t limits (five degrees of freedom), minimum and maximum. |
| [Table 11a interval rounding](table11a_interval_rounding.json) | Identifies the source row and the fixed-support lower limit, 0.7773499502396988, which rounds to 0.7773. |

The four records preserve the fields and hashes from the original numerical checks. In their path labels, `EdDynamics/` denotes the repository root and `outputs_KT4/` denotes the external results tree used for verification. The initial comparison uses `all_match` for agreement with the earlier printed table. In the retention check, `all_agree` records agreement between the original-function results and the independent calculation for all eight corrected rows.

The confirmation-sample counts in [numerical report 04](../reference_outputs/04_event_ssl_and_controls_numeric_results.md) refer to common prediction intervals and their distinct learners. Their source and the extractor’s handling of these counts are described in the [reference report guide](../reference_outputs/README.md#confirmation-sample-counts-in-report-04).
