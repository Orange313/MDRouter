# MDRouter EAB reproduction artifact

This artifact separates retrieval-plan quality, positive reference-relative regret, complete measured path costs, and statistical guarantees. It contains reproducibility code and authorized numerical measurements, not a new certified router or a claim of Serving superiority.

## Reproduce (CPU; Linux/Python3.11)

```sh
python -m venv .venv-replay
. .venv-replay/bin/activate
python -m pip install -r requirements-replay-lock.txt
python reproduce_all.py --output /absolute/path/NEW_REPLAY
```

Expected: 38 core numeric CSV hashes +4 lifecycle sensitivity CSV hashes, and5 figure families. The fresh-install local clean-clone receipt from candidate010 is about26seconds for replay, excluding dependency installation. No GPU/network/private research checkout is required for this numerical replay. Actual GPU timing is archived, not remeasured by this command. Use official acquisition scripts and fixed revisions for hardware reruns; obey upstream access/license terms.

## Provenance and scope

Based on the audited783-file candidate010, ZIP SHA256 `3ca814f4e0ecac37de42c4b2db8a3b9dfad2d3a4bd53d61c7c93a6784845b266`. Public upload was authorized2026-10-29. README publication status is updated; numerical code, inputs and expected hashes are unchanged. The candidate/provenance/validation records describe their historical prepublication state. No raw document images, pretrained model weights, fullqrels, sealed partitions, visual indexes or private research Git history are included. Biomedical frozen-policy results are descriptive; ColPali extension is post-hoc; historical Dev data was repeatedly exposed; newK1 is NOT_CERTIFIED; strictServing equivalence was not established.

## Licenses and public verification

No original-code license grant has been selected or added. `LICENSE_PROPOSED.md` is only a proposal. Upstream data/model terms remain separate: consult `THIRD_PARTY_NOTICES_010.md` and `license_pin_verification_010.json`; absent standalone license endpoints remain NOT_VERIFIED. Publication is not a blanket MIT relicensing. Public/no-login clone replay must be reported separately after an actual successful push; the local candidate receipt alone does not establish public verification. No CMT submission is performed by these scripts.
