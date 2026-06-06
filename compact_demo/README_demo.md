# Compact Demo

This compact demo is for artifact kick-the-tires and verifies the
enrollment-verification pipeline. It is not intended to reproduce the
full-scale numerical results in the paper.

The demo uses fully synthetic 64x64 RGB patterns. It exercises:

- enrollment from reference images
- precomputed source payload storage
- RGB-side verification against a claimed source
- computation of `r(I, c_s)`
- accept/reject output
- a tiny toy table with expected metrics

Run:

```powershell
python compact_demo\run_demo.py --verify
```

The prepared files are:

```text
compact_demo/demo_manifest.json
compact_demo/tiny_source_payloads.npz
compact_demo/tiny_checkpoint.pt
compact_demo/expected_outputs.json
```

Generated run outputs are written under:

```text
compact_demo/outputs/
```

