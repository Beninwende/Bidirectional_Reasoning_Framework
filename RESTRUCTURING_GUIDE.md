# Repository Restructuring Guide

## Current Status

The repository has been analyzed and the README has been updated to reflect the current structure and align with the paper's contributions. Experiment 2 has been marked as deprecated in the latest version.

## Current Structure

```
Bidirectional_Study/
├── Bidirectional_Reasoning_SANER/     # Paper source
├── Experiment1/                        # Standard Fine-Tuning (Active)
├── Experiment2/                       # DEPRECATED (removed from paper)
├── Experiment3/                       # Prompting Strategies (Active)
├── Experiment4-Contrastive-Fine-tuning/ # Contrastive Fine-Tuning (Active)
└── Evaluation_Datasets/               # Evaluation data
```

## Recommended Future Restructuring (Optional)

If you want to create a cleaner structure for future versions, consider:

```
Bidirectional_Study/
├── paper/                              # Renamed from Bidirectional_Reasoning_SANER
│   └── Bidirectional_Code_Reasoning_in_LLMs/
│
├── code/                               # All experiment code
│   ├── experiment1_sft/              # Renamed from Experiment1
│   ├── experiment3_prompting/         # Renamed from Experiment3
│   └── experiment4_cft/               # Renamed from Experiment4-Contrastive-Fine-tuning
│
├── data/                               # All datasets
│   ├── training/                      # Training datasets
│   └── evaluation/                    # Evaluation datasets (renamed from Evaluation_Datasets)
│
├── results/                            # Experimental results (optional)
│   └── ground_truth_results/          # Move from paper folder
│
└── docs/                               # Documentation
    ├── README.md
    └── RESTRUCTURING_GUIDE.md
```

## Action Items

### Immediate (Done)
- ✅ Updated README to reflect paper contributions
- ✅ Removed Experiment2 references from main documentation
- ✅ Updated all path references to match current structure
- ✅ Added clear note about Experiment2 deprecation

### Optional (Future)
- [ ] Archive or remove `Experiment2/` folder (if not needed)
- [ ] Consider renaming experiment folders for consistency
- [ ] Move evaluation datasets to a `data/evaluation/` folder
- [ ] Create a `docs/` folder for additional documentation
- [ ] Add `.gitignore` to exclude experiment outputs and temporary files

## Notes

- **Experiment2**: The folder exists but is not part of the current paper. You may want to:
  - Archive it in a `deprecated/` or `archive/` folder
  - Remove it entirely if not needed for reproducibility
  - Keep it but add a `README.md` explaining it's deprecated

- **Consistency**: Current naming mixes `Experiment1`, `Experiment3`, and `Experiment4-Contrastive-Fine-tuning`. Consider standardizing to `experiment1_sft`, `experiment3_prompting`, `experiment4_cft` for consistency.

- **Paper Location**: The paper source is in `Bidirectional_Reasoning_SANER/` which is descriptive but could be simplified to `paper/` for clarity.

## Migration Path (If Restructuring)

If you decide to restructure, here's a safe migration path:

1. **Create new structure** alongside existing one
2. **Update code paths** gradually (test each experiment)
3. **Update README** with new paths
4. **Remove old structure** once everything is verified

**Important**: Keep the current structure working until all scripts and references are updated!

